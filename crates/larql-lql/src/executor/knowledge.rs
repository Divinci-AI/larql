//! The knowledge seam (V3-LQL-3A): one browse implementation, two
//! knowledge sources.
//!
//! Every browse statement (WALK / SELECT / DESCRIBE / SHOW …) is
//! written once, against [`BrowseCtx`]. The context resolves at the
//! session's binding:
//!
//! - **V2**: delegates to the existing `PatchedVindex` — same calls
//!   the executors made directly before this seam existed, so V2
//!   behaviour is unchanged by construction.
//! - **V3**: the container's own query surface
//!   ([`KnowledgeView`]) — semantic roles bound to the executable
//!   plan's operands. No `VectorIndex` is manufactured and no V2
//!   loader runs; in particular, embeddings come from role
//!   `embedding`, never `load_vindex_embeddings`.
//!
//! This module is deliberately the ONLY place that knows there are
//! two sources. Statement executors consume the context; they never
//! ask which format is bound.

use std::borrow::Cow;
use std::path::Path;

use larql_vindex::format::vindex3::knowledge::KnowledgeView;
use larql_vindex::ndarray::{Array1, Array2};
use larql_vindex::{FeatureMeta, LayerBands, PatchedVindex, VindexConfig, WalkTrace};

use crate::error::LqlError;
use crate::executor::{Backend, Session};

/// The knowledge source behind one bound session.
pub(crate) enum KnowledgeSource<'a> {
    V2(&'a PatchedVindex),
    V3(&'a KnowledgeView),
}

impl KnowledgeSource<'_> {
    pub(crate) fn loaded_layers(&self) -> Vec<usize> {
        match self {
            Self::V2(patched) => patched.loaded_layers(),
            Self::V3(view) => view.loaded_layers(),
        }
    }

    pub(crate) fn num_features(&self, layer: usize) -> usize {
        match self {
            Self::V2(patched) => patched.num_features(layer),
            Self::V3(view) => view.num_features(layer),
        }
    }

    pub(crate) fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        match self {
            Self::V2(patched) => patched.feature_meta(layer, feature),
            Self::V3(view) => view.feature_meta(layer, feature),
        }
    }

    /// Every annotation of one layer (raw-token views). V2 serves
    /// this only in heap mode — mmap vindexes degrade to `None`
    /// exactly as they did before the seam.
    pub(crate) fn feature_metas(&self, layer: usize) -> Option<Vec<Option<FeatureMeta>>> {
        match self {
            Self::V2(patched) => patched.down_meta_at(layer).map(|m| m.to_vec()),
            Self::V3(view) => view.feature_metas(layer).map(|m| m.to_vec()),
        }
    }

    pub(crate) fn gate_knn(
        &self,
        layer: usize,
        query: &Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        match self {
            Self::V2(patched) => patched.gate_knn(layer, query, top_k),
            Self::V3(view) => view.gate_knn(layer, query, top_k),
        }
    }

    pub(crate) fn walk(&self, query: &Array1<f32>, layers: &[usize], top_k: usize) -> WalkTrace {
        match self {
            Self::V2(patched) => patched.walk(query, layers, top_k),
            Self::V3(view) => view.walk(query, layers, top_k),
        }
    }

    /// KNN-store entries for an entity (DESCRIBE's L0 section). The
    /// V3 overlay's store arrives with mutation (3B); until then the
    /// V3 answer is honestly empty.
    pub(crate) fn knn_entries_for_entity(
        &self,
        entity: &str,
    ) -> Vec<(usize, larql_vindex::KnnEntry)> {
        match self {
            Self::V2(patched) => patched
                .knn_store
                .entries_for_entity(entity)
                .into_iter()
                .map(|(index, entry)| (index, entry.clone()))
                .collect(),
            Self::V3(_) => Vec::new(),
        }
    }
}

/// Everything a browse statement needs, resolved once per statement.
pub(crate) struct BrowseCtx<'a> {
    pub path: &'a Path,
    pub num_layers: usize,
    /// The default LIMIT for per-layer feature listings.
    pub intermediate_size: usize,
    pub bands: LayerBands,
    /// Present on V2 bindings — prompt encoding honours the extracted
    /// architecture's BOS policy. `None` (V3, no `ModelArchitecture`)
    /// encodes through the tokenizer alone.
    pub config: Option<&'a VindexConfig>,
    pub source: KnowledgeSource<'a>,
}

impl BrowseCtx<'_> {
    /// Embeddings with their scale — role `embedding` on V3, the V2
    /// disk loader on V2 (same per-call load the executors did
    /// before the seam).
    pub(crate) fn embeddings(&self) -> Result<(Cow<'_, Array2<f32>>, f32), LqlError> {
        match &self.source {
            KnowledgeSource::V2(_) => {
                let (embed, scale) = larql_vindex::load_vindex_embeddings(self.path)
                    .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
                Ok((Cow::Owned(embed), scale))
            }
            KnowledgeSource::V3(view) => {
                let (embed, scale) = view.embedding();
                Ok((Cow::Borrowed(embed), scale))
            }
        }
    }

    /// Encode a prompt the way this binding's INFER does.
    pub(crate) fn encode_prompt(
        &self,
        tokenizer: &larql_vindex::tokenizers::Tokenizer,
        prompt: &str,
    ) -> Result<Vec<u32>, LqlError> {
        match self.config {
            Some(config) => crate::executor::query::encode_vindex_prompt(config, tokenizer, prompt),
            None => {
                let encoding = tokenizer
                    .encode(prompt, true)
                    .map_err(|e| LqlError::Execution(format!("tokenize: {e}")))?;
                let ids = encoding.get_ids().to_vec();
                if ids.is_empty() {
                    return Err(LqlError::Execution("prompt tokenises to empty".into()));
                }
                Ok(ids)
            }
        }
    }
}

impl Session {
    /// Resolve the browse context for the bound backend — the single
    /// place the V2/V3 distinction exists on the browse path.
    pub(crate) fn browse(&self) -> Result<BrowseCtx<'_>, LqlError> {
        match &self.backend {
            Backend::Vindex {
                path,
                config,
                patched,
                ..
            } => Ok(BrowseCtx {
                path,
                num_layers: config.num_layers,
                intermediate_size: config.intermediate_size,
                bands: crate::executor::query::resolve_bands(config),
                config: Some(config),
                source: KnowledgeSource::V2(patched),
            }),
            Backend::Vindex3 {
                path,
                runtime,
                knowledge,
                ..
            } => {
                let view = knowledge.as_ref().ok_or_else(|| {
                    LqlError::Execution(
                        "browse needs the tokenizer capability and this container carries no \
                         tokenizer.json — feature annotations cannot be decoded"
                            .into(),
                    )
                })?;
                let last = runtime.plan().layers.len().saturating_sub(1);
                Ok(BrowseCtx {
                    path,
                    num_layers: view.num_layers(),
                    intermediate_size: view.max_features(),
                    // The plan declares no band semantics yet; all
                    // three bands honestly span the whole stack.
                    bands: LayerBands {
                        syntax: (0, last),
                        knowledge: (0, last),
                        output: (0, last),
                    },
                    config: None,
                    source: KnowledgeSource::V3(view),
                })
            }
            Backend::Weight { model_id, .. } => Err(LqlError::Execution(format!(
                "this operation requires a vindex. Extract first:\n  \
                 EXTRACT MODEL \"{}\" INTO \"{}.vindex\"",
                model_id,
                model_id.split('/').next_back().unwrap_or(model_id),
            ))),
            _ => Err(LqlError::NoBackend),
        }
    }
}
