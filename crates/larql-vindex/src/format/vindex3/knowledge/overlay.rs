//! The V3 mutation overlay — logical edits over a read-only container.
//!
//! V3-LQL-3B: a VINDEX3 container is immutable on disk ("the bytes
//! executed are the bytes stored" is a construction-level property of
//! the format). Mutation therefore lives in an overlay addressed by
//! **semantic identity** — entity-keyed KNN entries today, feature-slot
//! overrides when the compose rung lands — never by byte offsets, so
//! an edit survives repacking or an alternative physical layout.
//!
//! The overlay speaks the same logical patch language as VINDEX2
//! ([`VindexPatch`] / [`PatchOp`]): one `.vlp` file applies to either
//! format, and the V2 semantics are the contract — all-or-nothing
//! apply, removal rebuilds by replaying the remaining patch list
//! (mirroring `PatchedVindex::rebuild_overrides`, session state
//! included in the reset).
//!
//! Feature-slot state (V3-LQL-3B rung 2) carries the V2 tombstone
//! contract verbatim (`PatchedVindex`, review 2026-07-30 M6): DELETE
//! pins a `None` meta override *and* tombstones the slot; a later
//! UPDATE resurrects it, and every read path — `feature_meta`, the
//! gate scan, WALK — must agree about the slot's existence.
//!
//! What this overlay refuses today: patch operations carrying gate/up/
//! down **vectors** (compose installs). Those must be observed by
//! *execution*, which needs the operand-source seam of the compose
//! rung. A patch containing them is refused **whole** — a partial
//! apply would misrepresent the patch's meaning.

use std::collections::{HashMap, HashSet};

use crate::error::VindexError;
use crate::index::types::FeatureMeta;
use crate::patch::format::{decode_gate_vector, PatchOp, VindexPatch};
use crate::patch::knn_store::KnnStore;

/// Logical mutation state over one bound VINDEX3 container.
#[derive(Default)]
pub struct KnowledgeOverlay {
    /// Entity-keyed retrieval entries (Architecture B). Shared store
    /// type with V2 — same logical semantics, same `knn_store.bin`
    /// persistence format.
    pub knn_store: KnnStore,
    /// Patches applied to this session, in application order.
    pub patches: Vec<VindexPatch>,
    /// Feature-meta overrides: `Some(meta)` replaces the base view's
    /// annotation, a pinned `None` (from DELETE) hides it.
    overrides_meta: HashMap<(usize, usize), Option<FeatureMeta>>,
    /// Tombstoned feature slots — excluded from every read path until
    /// an UPDATE resurrects them.
    deleted: HashSet<(usize, usize)>,
}

impl KnowledgeOverlay {
    pub fn new() -> Self {
        Self::default()
    }

    /// Tombstone a feature slot (V2's `PatchedVindex::delete_feature`
    /// contract: pin a `None` meta AND record the tombstone, so the
    /// meta path and the gate-scan path agree the slot is gone).
    pub fn delete_feature(&mut self, layer: usize, feature: usize) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, None);
        self.deleted.insert(key);
    }

    /// Override a feature's metadata; a prior tombstone on the slot is
    /// cleared (resurrection — updating a feature implies it exists).
    pub fn update_feature_meta(&mut self, layer: usize, feature: usize, meta: FeatureMeta) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, Some(meta));
        self.deleted.remove(&key);
    }

    /// Resolve a slot's metadata over the base view's answer — V2's
    /// exact read rule: an override wins (its pinned `None` hides the
    /// slot), a bare tombstone hides it, otherwise the base answers.
    pub fn resolve_feature_meta(
        &self,
        layer: usize,
        feature: usize,
        base: Option<FeatureMeta>,
    ) -> Option<FeatureMeta> {
        let key = (layer, feature);
        if let Some(override_meta) = self.overrides_meta.get(&key) {
            return override_meta.clone();
        }
        if self.deleted.contains(&key) {
            return None;
        }
        base
    }

    /// Whether the slot is currently tombstoned.
    pub fn is_tombstoned(&self, layer: usize, feature: usize) -> bool {
        self.deleted.contains(&(layer, feature))
    }

    /// How many slots are tombstoned at `layer` — the exact oversample
    /// a gate scan needs to stay full after filtering.
    pub fn tombstones_at(&self, layer: usize) -> usize {
        self.deleted.iter().filter(|&&(l, _)| l == layer).count()
    }

    /// Whether any feature-slot state exists (meta overrides or
    /// tombstones) — lets read paths skip the merge entirely.
    pub fn has_feature_state(&self) -> bool {
        !self.overrides_meta.is_empty() || !self.deleted.is_empty()
    }

    /// Apply the overlay onto one layer's full annotation vector
    /// (`feature_metas`-shaped reads).
    pub fn apply_meta_overrides(&self, layer: usize, metas: &mut [Option<FeatureMeta>]) {
        for (feature, slot) in metas.iter_mut().enumerate() {
            let key = (layer, feature);
            if let Some(override_meta) = self.overrides_meta.get(&key) {
                *slot = override_meta.clone();
            } else if self.deleted.contains(&key) {
                *slot = None;
            }
        }
    }

    /// Apply a patch, all-or-nothing: on `Err` no overlay state has
    /// been touched and the patch is not recorded. Errors when an
    /// embedded vector fails to decode, or when the patch contains
    /// feature-slot operations the V3 overlay cannot yet represent.
    pub fn try_apply_patch(&mut self, patch: VindexPatch) -> Result<(), VindexError> {
        validate_v3_patch(&patch)?;
        self.apply_unchecked(&patch);
        self.patches.push(patch);
        Ok(())
    }

    /// Remove a previously applied patch and rebuild the overlay by
    /// replaying the remaining patch list — V2's removal contract:
    /// session-added state outside a patch is reset too.
    pub fn remove_patch(&mut self, index: usize) {
        if index >= self.patches.len() {
            return;
        }
        self.patches.remove(index);
        self.knn_store = KnnStore::default();
        self.overrides_meta.clear();
        self.deleted.clear();
        let patches = std::mem::take(&mut self.patches);
        for patch in patches {
            self.apply_unchecked(&patch);
            self.patches.push(patch);
        }
    }

    fn apply_unchecked(&mut self, patch: &VindexPatch) {
        for op in &patch.operations {
            match op {
                PatchOp::InsertKnn {
                    layer,
                    entity,
                    relation,
                    target,
                    target_id,
                    confidence,
                    key_vector_b64,
                } => {
                    if let Ok(key_vec) = decode_gate_vector(key_vector_b64) {
                        self.knn_store.add(
                            *layer,
                            key_vec,
                            *target_id,
                            target.clone(),
                            entity.clone(),
                            relation.clone(),
                            confidence.unwrap_or(1.0),
                        );
                    }
                }
                PatchOp::DeleteKnn { entity } => {
                    self.knn_store.remove_by_entity(entity);
                }
                PatchOp::Delete { layer, feature, .. } => {
                    self.delete_feature(*layer, *feature);
                }
                // Vector-free Update — V2's resolution (overlay_apply)
                // verbatim: a carried meta becomes the override; the
                // resurrect rule drops a pinned `None` only when this
                // Update carries no replacement meta.
                PatchOp::Update {
                    layer,
                    feature,
                    down_meta,
                    ..
                } => {
                    let key = (*layer, *feature);
                    if let Some(dm) = down_meta {
                        let meta = FeatureMeta {
                            top_token: dm.top_token.clone(),
                            top_token_id: dm.top_token_id,
                            c_score: dm.c_score,
                            top_k: vec![larql_models::TopKEntry {
                                token: dm.top_token.clone(),
                                token_id: dm.top_token_id,
                                logit: dm.c_score,
                            }],
                        };
                        self.overrides_meta.insert(key, Some(meta));
                    }
                    if self.deleted.remove(&key)
                        && matches!(self.overrides_meta.get(&key), Some(None))
                    {
                        self.overrides_meta.remove(&key);
                    }
                }
                // validate_v3_patch refused these before apply.
                PatchOp::Insert { .. } => {}
                // Vector-free Update — V2's resolution (overlay_apply)
                // verbatim: a carried meta becomes the override; the
                // resurrect rule drops a pinned `None` only when this
                // Update carries no replacement meta.
                PatchOp::Update {
                    layer,
                    feature,
                    down_meta,
                    ..
                } => {
                    let key = (*layer, *feature);
                    if let Some(dm) = down_meta {
                        let meta = FeatureMeta {
                            top_token: dm.top_token.clone(),
                            top_token_id: dm.top_token_id,
                            c_score: dm.c_score,
                            top_k: vec![larql_models::TopKEntry {
                                token: dm.top_token.clone(),
                                token_id: dm.top_token_id,
                                logit: dm.c_score,
                            }],
                        };
                        self.overrides_meta.insert(key, Some(meta));
                    }
                    if self.deleted.remove(&key)
                        && matches!(self.overrides_meta.get(&key), Some(None))
                    {
                        self.overrides_meta.remove(&key);
                    }
                }
            }
        }
    }
}

/// Check every operation is representable on the V3 overlay and every
/// embedded vector decodes, before any state changes.
fn validate_v3_patch(patch: &VindexPatch) -> Result<(), VindexError> {
    for (i, op) in patch.operations.iter().enumerate() {
        match op {
            PatchOp::InsertKnn { key_vector_b64, .. } => {
                decode_gate_vector(key_vector_b64).map_err(|e| {
                    VindexError::Parse(format!("patch op {i}: corrupt key_vector_b64: {e}"))
                })?;
            }
            PatchOp::DeleteKnn { .. } | PatchOp::Delete { .. } => {}
            PatchOp::Update {
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            } => {
                if gate_vector_b64.is_some() || up_vector_b64.is_some() || down_vector_b64.is_some()
                {
                    return Err(compose_rung_refusal(i, "an Update carrying vectors"));
                }
            }
            PatchOp::Insert { .. } => {
                return Err(compose_rung_refusal(i, "an Insert (compose install)"));
            }
        }
    }
    Ok(())
}

/// The vectors these operations carry must be observed by execution,
/// which needs the operand-source seam of the compose rung.
fn compose_rung_refusal(op_index: usize, what: &str) -> VindexError {
    VindexError::Parse(format!(
        "patch op {op_index} is {what} — vector-bearing feature-slot patches on a \
         VINDEX3 container arrive with the compose rung (V3-LQL-3B compose); \
         the patch was not applied"
    ))
}
