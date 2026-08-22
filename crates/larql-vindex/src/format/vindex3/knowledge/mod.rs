//! The V3 query surface (V3-LQL-3A): the model as a database,
//! addressed by semantic role.
//!
//! Inference closure says "enough information exists to execute". LQL
//! query closure additionally says "enough information exists to
//! interrogate the model as a database". This module is that second
//! closure for a VINDEX3 container: it binds the browse-facing
//! **semantic roles** to the executable plan's own operands —
//!
//! ```text
//! role feature_gate   ← the layer's FFN detector matrix
//!                       (FfnOp.gate when the FFN is gated, else .up)
//! role feature_down   ← FfnOp.down (column f = feature f's output
//!                       direction)
//! role embedding      ← EmbeddingOp.table (+ scale)
//! role unembedding    ← OutputOp.projection
//! ```
//!
//! — and derives the feature space LQL browses: features are
//! `(layer, index)` rows of `feature_gate`, annotated with the tokens
//! their `feature_down` column promotes (see below). No
//! `VectorIndex` is manufactured, no family metadata is consulted, no
//! V2 file is read: every array here loads through the same
//! [`OperandStore`] execution uses, so the queryable view and the
//! executed program cannot name different bytes.
//!
//! Scope at this rung: dense FFN layers carry features; routed (MoE)
//! layers report zero features (their per-expert feature space is a
//! later rung of the role vocabulary).
//!
//! Annotation semantics deliberately match the V2 extractor's
//! user-visible contract (`extract/streaming/stages/down_meta.rs` and
//! `extract/build/down_meta.rs`): a feature's promoted tokens are its
//! `feature_down` column scored against the **embedding table**
//! (`embed · down_col`, unscaled), decoded with specials skipped,
//! trimmed, empty surfaces dropped; `c_score` is the top logit. On
//! tied-embedding models this equals promotion through the output
//! head; where they differ, V2's embedding-referenced reading is the
//! contract until the role vocabulary explicitly supersedes it.
//!
//! Annotations are computed eagerly at bind time (one
//! `embedding × feature_down` product per layer). Fine for the
//! conformance fixtures this rung gates on; lazy/memoised annotation
//! for large containers is deliberately later, perf-shaped work.

#[cfg(test)]
mod tests;

use larql_models::TopKEntry;
use ndarray::{Array1, Array2};

use crate::error::VindexError;
use crate::index::types::FeatureMeta;
use crate::index::types::{WalkHit, WalkTrace};

use super::opplan::exec::operands::OperandStore;
use super::opplan::{ComponentOpPlan, LayerFfn};

/// How many promoted tokens each feature annotation carries — matches
/// the V2 extractor's default `down_top_k`.
const ANNOTATION_TOP_K: usize = 8;

/// One layer's browsable feature space.
struct LayerKnowledge {
    /// Role `feature_gate`: `[num_features, hidden]`.
    gate: Array2<f32>,
    /// Per-feature annotations (role `feature_down` scored against
    /// role `embedding` — the V2 contract).
    metas: Vec<Option<FeatureMeta>>,
}

/// A VINDEX3 container's browse view: the feature/edge space LQL
/// queries, derived from the executable plan's semantic roles.
pub struct KnowledgeView {
    layers: Vec<Option<LayerKnowledge>>,
    hidden_size: usize,
    /// Largest dense FFN width — the browse surface's
    /// "intermediate size" (per-layer counts come from the gate rows).
    max_features: usize,
    /// Role `embedding`: `[vocab, hidden]`.
    embedding: Array2<f32>,
    embed_scale: f32,
    vocab_size: usize,
}

impl KnowledgeView {
    /// Bind the browse roles from the plan and derive the feature
    /// annotations. `tokenizer` decodes promoted token ids into the
    /// surface strings LQL displays.
    pub fn from_plan(
        plan: &ComponentOpPlan,
        store: &OperandStore,
        tokenizer: &crate::tokenizers::Tokenizer,
    ) -> Result<Self, VindexError> {
        let embedding_op = plan.embedding.as_ref().ok_or_else(|| {
            VindexError::Parse(
                "component carries no embedding op — role `embedding` is unbound".to_string(),
            )
        })?;
        let hidden = embedding_op.table.shape[1];
        let embedding = matrix(store, &embedding_op.table)?;
        let embed_scale = embedding_op.scale.unwrap_or(1.0);

        let mut layers = Vec::with_capacity(plan.layers.len());
        let mut max_features = 0;
        for layer in &plan.layers {
            let LayerFfn::Dense(ffn) = &layer.ffn else {
                // Routed/hybrid feature spaces are a later role rung.
                layers.push(None);
                continue;
            };
            // Role `feature_gate`: the detector matrix. A gated FFN's
            // detector is the gate projection; an ungated FFN's is the
            // up projection.
            let gate = matrix(store, ffn.gate.as_ref().unwrap_or(&ffn.up))?;
            // Role `feature_down`: `[hidden, num_features]`.
            let down = matrix(store, &ffn.down)?;
            let num_features = gate.shape()[0];
            max_features = max_features.max(num_features);

            let metas = annotate(&embedding, &down, tokenizer);
            layers.push(Some(LayerKnowledge { gate, metas }));
        }

        Ok(Self {
            layers,
            hidden_size: hidden,
            max_features,
            vocab_size: embedding_op.vocab_size,
            embedding,
            embed_scale,
        })
    }

    /// Layers that carry a browsable feature space, ascending.
    pub fn loaded_layers(&self) -> Vec<usize> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| layer.as_ref().map(|_| index))
            .collect()
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn num_features(&self, layer: usize) -> usize {
        self.layers
            .get(layer)
            .and_then(|l| l.as_ref())
            .map(|l| l.gate.shape()[0])
            .unwrap_or(0)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn max_features(&self) -> usize {
        self.max_features
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.layers
            .get(layer)?
            .as_ref()?
            .metas
            .get(feature)?
            .clone()
    }

    /// Every feature annotation of one layer (LQL's raw-token views).
    pub fn feature_metas(&self, layer: usize) -> Option<&[Option<FeatureMeta>]> {
        Some(&self.layers.get(layer)?.as_ref()?.metas)
    }

    /// Role `embedding` with its scale — the entity/query vector
    /// source for DESCRIBE / SELECT NEAREST / INSERT.
    pub fn embedding(&self) -> (&Array2<f32>, f32) {
        (&self.embedding, self.embed_scale)
    }

    /// Top-k features of `layer` by gate response to `query` — the
    /// same statistic and ranking the V2 gate KNN uses (dot product,
    /// ordered by absolute magnitude descending), so the two backends'
    /// walks are comparable row for row.
    pub fn gate_knn(&self, layer: usize, query: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        let Some(Some(knowledge)) = self.layers.get(layer) else {
            return Vec::new();
        };
        let scores = knowledge.gate.dot(query);
        let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.truncate(top_k);
        ranked
    }

    /// The browse walk: per requested layer, the annotated top-k gate
    /// hits — the same trace shape `PatchedVindex::walk` produces.
    pub fn walk(&self, query: &Array1<f32>, layers: &[usize], top_k: usize) -> WalkTrace {
        let mut trace = Vec::with_capacity(layers.len());
        for &layer in layers {
            let hits: Vec<WalkHit> = self
                .gate_knn(layer, query, top_k)
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.feature_meta(layer, feature)?;
                    Some(WalkHit::from_gate(layer, feature, gate_score, meta))
                })
                .collect();
            trace.push((layer, hits));
        }
        WalkTrace { layers: trace }
    }
}

/// Load one operand as a 2-D array through the execution loader.
fn matrix(
    store: &OperandStore,
    operand: &super::opplan::OperandRef,
) -> Result<Array2<f32>, VindexError> {
    let values = store.load(operand)?;
    let rows = operand.shape.first().copied().unwrap_or(0);
    let cols = operand.shape.get(1).copied().unwrap_or(0);
    Array2::from_shape_vec((rows, cols), values).map_err(|e| {
        VindexError::Parse(format!(
            "operand {}::{} is not [{rows}, {cols}]: {e}",
            operand.object, operand.tensor
        ))
    })
}

/// Annotate every feature with the tokens its `feature_down` column
/// promotes — the V2 extractor's statement, verbatim: scores are
/// `embedding · down_col` (unscaled), decoded skipping specials,
/// trimmed, empty surfaces dropped; `c_score` is the top logit.
fn annotate(
    embedding: &Array2<f32>,
    down: &Array2<f32>,
    tokenizer: &crate::tokenizers::Tokenizer,
) -> Vec<Option<FeatureMeta>> {
    // `[vocab, hidden] × [hidden, num_features]` → `[vocab, num_features]`.
    let logits = embedding.dot(down);
    let num_features = down.shape()[1];
    (0..num_features)
        .map(|feature| {
            let column = logits.column(feature);
            let mut ranked: Vec<(usize, f32)> = column.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ranked.truncate(ANNOTATION_TOP_K);
            let top_k: Vec<TopKEntry> = ranked
                .iter()
                .filter_map(|&(id, logit)| {
                    tokenizer
                        .decode(&[id as u32], true)
                        .ok()
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .map(|token| TopKEntry {
                            token,
                            token_id: id as u32,
                            logit,
                        })
                })
                .collect();
            let first = top_k.first()?;
            Some(FeatureMeta {
                top_token: first.token.clone(),
                top_token_id: first.token_id,
                c_score: first.logit,
                top_k,
            })
        })
        .collect()
}
