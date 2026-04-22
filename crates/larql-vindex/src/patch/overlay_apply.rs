//! Patch application — `apply_patch`, `remove_patch`,
//! `rebuild_overrides` for `PatchedVindex`.
//!
//! Walks `VindexPatch::operations` and resolves each one into the
//! overlay's override maps (or the L0 KNN store for arch-B ops).
//! Pulled out of `overlay.rs` so the file holding `PatchedVindex`'s
//! query/mutation API stays focused.

use crate::index::FeatureMeta;

use super::format::{decode_gate_vector, PatchOp, VindexPatch};
use super::overlay::PatchedVindex;

impl PatchedVindex {
    /// Apply a patch. Operations are resolved into the override maps.
    pub fn apply_patch(&mut self, patch: VindexPatch) {
        for op in &patch.operations {
            match op {
                PatchOp::InsertKnn { layer, entity, relation, target, target_id, confidence, key_vector_b64 } => {
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
                    continue;
                }
                PatchOp::DeleteKnn { entity } => {
                    self.knn_store.remove_by_entity(entity);
                    continue;
                }
                _ => {}
            }
            let key = op.key().unwrap(); // safe: only Arch A ops reach here
            match op {
                PatchOp::Insert { target, confidence, gate_vector_b64, down_meta, .. } => {
                    let meta = if let Some(dm) = down_meta {
                        FeatureMeta {
                            top_token: dm.top_token.clone(),
                            top_token_id: dm.top_token_id,
                            c_score: dm.c_score,
                            top_k: vec![larql_models::TopKEntry {
                                token: dm.top_token.clone(),
                                token_id: dm.top_token_id,
                                logit: dm.c_score,
                            }],
                        }
                    } else {
                        FeatureMeta {
                            top_token: target.clone(),
                            top_token_id: 0,
                            c_score: confidence.unwrap_or(0.9),
                            top_k: vec![],
                        }
                    };
                    self.overrides_meta.insert(key, Some(meta));
                    self.deleted.remove(&key);
                    if let Some(b64) = gate_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_gate.insert(key, vec);
                        }
                    }
                }
                PatchOp::Update { gate_vector_b64, down_meta, .. } => {
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
                    if let Some(b64) = gate_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_gate.insert(key, vec);
                        }
                    }
                }
                PatchOp::Delete { .. } => {
                    self.overrides_meta.insert(key, None);
                    self.deleted.insert(key);
                    self.overrides_gate.remove(&key);
                }
                PatchOp::InsertKnn { .. } | PatchOp::DeleteKnn { .. } => {
                    unreachable!("KNN ops handled above");
                }
            }
        }
        self.patches.push(patch);
    }

    /// Remove the last applied patch and rebuild overrides.
    pub fn remove_patch(&mut self, index: usize) {
        if index < self.patches.len() {
            self.patches.remove(index);
            self.rebuild_overrides();
        }
    }

    /// Rebuild override maps from scratch (after removing a patch).
    fn rebuild_overrides(&mut self) {
        self.overrides_meta.clear();
        self.overrides_gate.clear();
        self.deleted.clear();
        // Clear base weight overrides so removed patches don't leak their
        // down/up vectors into subsequent apply_patch calls.
        // (Divinci-AI fork: Phase 1 unlearning depends on this being clean.)
        self.base.down_overrides.clear();
        self.base.up_overrides.clear();
        self.knn_store = super::knn_store::KnnStore::default();
        let patches: Vec<VindexPatch> = self.patches.drain(..).collect();
        for patch in patches {
            self.apply_patch(patch);
        }
    }
}

#[cfg(test)]
mod rebuild_overrides_tests {
    //! Regression guard for the Divinci-AI Phase-1 unlearning revert path.
    //!
    //! `rebuild_overrides` runs after `remove_patch` to reset the overlay
    //! state. It must clear *both* the per-PatchedVindex overlay maps
    //! (`overrides_meta`, `overrides_gate`, `deleted`) AND the base-side
    //! weight overrides on the underlying VectorIndex
    //! (`base.down_overrides`, `base.up_overrides`) — otherwise weight-level
    //! INSERT patches written via `set_down_vector` / `set_up_vector` leak
    //! across `remove_patch` calls and the next `apply_patch` sees stale
    //! base weights. Phase-1 unlearning revert depends on a clean reset.
    //!
    //! If a future refactor drops the `base.down_overrides.clear()` /
    //! `base.up_overrides.clear()` lines in `rebuild_overrides`, this test
    //! turns red.
    use super::*;
    use crate::index::core::VectorIndex;
    use crate::patch::format::{PatchOp, VindexPatch};
    use ndarray::Array2;

    fn make_pv() -> super::PatchedVindex {
        // Minimal 1-layer × 2-feature × 4-hidden synthetic vindex.
        let gate0 = Array2::<f32>::zeros((2, 4));
        let down_meta = vec![Some(vec![None, None])];
        let index = VectorIndex::new(vec![Some(gate0)], down_meta, 1, 4);
        super::PatchedVindex::new(index)
    }

    #[test]
    fn rebuild_overrides_clears_base_down_and_up_overrides() {
        let mut pv = make_pv();

        // Simulate a COMPILE-WITH-REFINE write that lands on the base
        // weight-override maps.
        pv.set_down_vector(0, 0, vec![1.0, 2.0, 3.0, 4.0]);
        pv.set_up_vector(0, 0, vec![0.5, 0.5, 0.5, 0.5]);
        assert!(!pv.base.down_overrides.is_empty(), "precondition: base.down_overrides should be populated");
        assert!(!pv.base.up_overrides.is_empty(),   "precondition: base.up_overrides should be populated");

        // Push any patch onto the overlay so `remove_patch(0)` has something
        // to remove and consequently triggers `rebuild_overrides`.
        let patch = VindexPatch {
            version: 1,
            base_model: "test".into(),
            base_checksum: None,
            created_at: "1970-01-01T00:00:00Z".into(),
            description: None,
            author: None,
            tags: vec![],
            operations: vec![PatchOp::Delete { layer: 0, feature: 1, reason: None }],
        };
        pv.apply_patch(patch);
        assert_eq!(pv.patches.len(), 1, "patch should be on the stack before remove");

        // Critical: revert. rebuild_overrides should clear *both* layers.
        pv.remove_patch(0);

        assert!(pv.base.down_overrides.is_empty(),
            "REGRESSION: rebuild_overrides did not clear base.down_overrides — \
             weight-level patches will leak across revert and Phase-1 unlearning is broken.");
        assert!(pv.base.up_overrides.is_empty(),
            "REGRESSION: rebuild_overrides did not clear base.up_overrides — \
             same leak path as above for up vectors.");
        assert!(pv.overrides_meta.is_empty(), "overlay overrides_meta should also be empty after revert");
        assert_eq!(pv.patches.len(), 0, "patch stack should be empty after remove");
    }
}
