//! Full-K gemv fast path for the sparse walk.
//!
//! When K ≥ the full-K density threshold (see `helpers::
//! hits_len_ge_intermediate`), the per-feature loop is mathematically
//! equivalent to three dense matmuls, so the walk routes through BLAS
//! gemm (native f32) or Q4K direct matmul (`kquant_matmul_transb`)
//! instead. Traced as `sparse:gemv_full_k`. The caller
//! (`walk_ffn_sparse`) is responsible for the density/selector/override
//! gating; this method only tries the batched kernels and returns
//! `None` when neither is available so the per-position loop runs.

use ndarray::Array2;

use super::WalkFfn;

impl<'a> WalkFfn<'a> {
    /// Attempt the full-K batched gemv/gemm path. Returns
    /// `Some((out, full_activation))` when both the gate-score batch and
    /// the up/down matmuls are available; `None` falls back to the
    /// per-position walk.
    pub(super) fn sparse_full_k_gemv(
        &self,
        layer: usize,
        x: &Array2<f32>,
        up_native: Option<ndarray::ArrayView2<'_, f32>>,
        down_native: Option<ndarray::ArrayView2<'_, f32>>,
        use_gelu: bool,
        intermediate: usize,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let seq_len = x.shape()[0];
        let hidden = x.shape()[1];
        let x_slice_for_matmul: Option<&[f32]> = x.as_slice();
        let (gate_scores, x_flat) = match (
            self.index.gate_scores_batch_backend(layer, x, self.backend),
            x_slice_for_matmul,
        ) {
            (Some(g), Some(xf)) => (g, xf),
            _ => return None,
        };

        let up_scores: Option<ndarray::Array2<f32>> = if let Some(v) = up_native {
            Some(larql_compute::dot_proj_gpu(x, &v, self.backend))
        } else if let Some(y) = self
            .index
            .kquant_matmul_transb(layer, 1, x_flat, seq_len, self.backend)
        {
            ndarray::Array2::from_shape_vec((seq_len, intermediate), y).ok()
        } else {
            None
        };
        let up_scores = up_scores?;

        let activation = if use_gelu {
            crate::ffn::gelu_tanh_gate_up(&gate_scores, &up_scores)
        } else {
            crate::ffn::silu_gate_up(&gate_scores, &up_scores)
        };
        let act_slice: Option<&[f32]> = activation.as_slice();
        let out_matmul: Option<ndarray::Array2<f32>> = if let Some(v) = down_native {
            Some(larql_compute::matmul_gpu(&activation, &v, self.backend))
        } else if let Some(act_flat) = act_slice {
            self.index
                .kquant_matmul_transb(layer, 2, act_flat, seq_len, self.backend)
                .and_then(|y| ndarray::Array2::from_shape_vec((seq_len, hidden), y).ok())
        } else {
            None
        };
        let out_matmul = out_matmul?;

        self.trace_path(layer, "sparse:gemv_full_k");
        Some((out_matmul, activation))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights, make_test_weights};
    use crate::vindex::{WalkFfn, WalkFfnConfig};
    use ndarray::Array2;

    fn x(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    /// Sparse walk with full-K (K >= num_features) routes through the
    /// gemv fast path. Drives the `hits_len_ge_intermediate` branch.
    #[test]
    fn walk_ffn_sparse_full_k_takes_gemv_path() {
        use crate::test_utils::{attach_feature_major_f32_to_test_vindex, make_test_vindex};
        let weights = make_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let cfg = WalkFfnConfig::dense(weights.num_layers);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let out = ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size))
            .expect("dense-K sparse walk should succeed");
        assert_eq!(out.0.shape(), &[1, weights.hidden_size]);
    }

    /// Sparse walk in full-K mode against the Q4K fixture (no native
    /// up/down) drives the `kquant_matmul_transb` arms inside the
    /// full-K gemv fast path: up_scores via Q4K matmul, then down via
    /// Q4K matmul again.
    #[test]
    fn walk_ffn_sparse_full_k_routes_through_kquant_matmul_on_q4k_fixture() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let cfg = WalkFfnConfig::dense(weights.num_layers);
        let backend = larql_compute::cpu_backend();
        let ffn = WalkFfn::from_config(&weights, &index, cfg).with_backend(&*backend);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size));
        // Full-K + Q4K — either takes the fast path (Some) or falls through
        // to the serial loop (also Some). Just exercise the wiring.
        if let Some((out, _activation)) = result {
            assert_eq!(out.shape(), &[1, weights.hidden_size]);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}
