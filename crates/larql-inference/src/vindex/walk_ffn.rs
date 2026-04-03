//! WalkFfn — FFN backend that uses VectorIndex gate KNN for feature selection.
//!
//! Gate KNN from the (potentially patched) vindex selects which features fire.
//! The actual FFN computation uses the model's up/down weights for those features.
//! This means INSERT/DELETE/UPDATE to the vindex affect inference output.
//!
//! Optimized hot path: gate_knn_batch (one BLAS gemm) → sparse FFN.
//! Trace recording is deferred — only computed when take_trace() is called.

use ndarray::Array2;

use crate::ffn::FfnBackend;
use crate::ffn::sparse_compute::{sparse_ffn_forward, sparse_ffn_forward_with_overrides};
use crate::model::ModelWeights;

use larql_vindex::{GateIndex, WalkHit, WalkTrace};

/// FFN backend that uses a GateIndex for gate selection.
///
/// The gate matmul IS the KNN. `residual × gate_vectors^T` is both the gate
/// computation and the similarity search. Same operation, different framing.
///
/// Hot path (forward): gate_knn_batch → sparse/dense FFN. No trace overhead.
/// Cold path (take_trace): re-runs gate_knn on last residual to build trace.
pub struct WalkFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a dyn GateIndex,
    pub top_k: usize,
    /// Deferred trace: stores (layer, last_position_residual) for lazy trace building.
    trace_residuals: std::cell::RefCell<Vec<(usize, Vec<f32>)>>,
    /// Whether to record residuals for deferred trace. Default: true.
    record_trace: bool,
}

impl<'a> WalkFfn<'a> {
    pub fn new(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        Self {
            weights,
            index,
            top_k,
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: false,
        }
    }

    /// Create a WalkFfn with trace recording enabled.
    /// Only use this when you need to call take_trace() afterwards.
    pub fn new_with_trace(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        Self {
            weights,
            index,
            top_k,
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: true,
        }
    }

    /// Take the accumulated walk trace (clears internal state).
    /// Lazily computes gate KNN + feature_meta for each recorded layer.
    pub fn take_trace(&self) -> WalkTrace {
        let residuals = self.trace_residuals.borrow_mut().drain(..).collect::<Vec<_>>();
        let mut layers = Vec::with_capacity(residuals.len());

        for (layer, residual) in residuals {
            let r = ndarray::Array1::from_vec(residual);
            let hits = self.index.gate_knn(layer, &r, self.top_k);
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.index.feature_meta(layer, feature)?.clone();
                    Some(WalkHit { layer, feature, gate_score, meta })
                })
                .collect();
            layers.push((layer, walk_hits));
        }

        WalkTrace { layers }
    }
}

impl<'a> FfnBackend for WalkFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    fn forward_with_activation(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let num_features = self.index.num_features(layer);
        if num_features == 0 {
            // No vindex data for this layer — fall back to dense
            let dense_ffn = crate::ffn::WeightFfn { weights: self.weights };
            return dense_ffn.forward_with_activation(layer, x);
        }

        // Batched gate KNN: one BLAS gemm for all positions, union top-K
        let features = self.index.gate_knn_batch(layer, x, self.top_k);

        // Record last-position residual for deferred trace (cheap: just a vec copy)
        if self.record_trace {
            let seq_len = x.shape()[0];
            let last_row = x.row(seq_len - 1).to_vec();
            self.trace_residuals.borrow_mut().push((layer, last_row));
        }

        // Check for down vector overrides (patched features)
        let has_overrides = features.iter().any(|&f| self.index.down_override(layer, f).is_some());

        if has_overrides {
            let overrides: Vec<(usize, &[f32])> = features.iter()
                .filter_map(|&f| self.index.down_override(layer, f).map(|v| (f, v)))
                .collect();
            sparse_ffn_forward_with_overrides(self.weights, layer, x, &features, &overrides)
        } else {
            sparse_ffn_forward(self.weights, layer, x, &features)
        }
    }

    fn name(&self) -> &str {
        "walk"
    }
}
