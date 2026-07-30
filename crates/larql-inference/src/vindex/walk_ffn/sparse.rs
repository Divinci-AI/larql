//! Sparse walk path — zero matrix multiplications.
//!
//! The hot path for FFN inference on the LARQL vindex. For each position:
//!
//!   1. `gate_knn` → top-K features (HNSW / batched brute-force / gate-walk)
//!   2. For each feature:
//!      - `up_score  = dot(up_row(feat), x)`         via unified ffn_row_dot
//!      - `activated = silu(gate_score) * up_score`   (GEGLU)
//!      - `out      += activated * down_row(feat)`   via unified ffn_row_scaled_add
//!
//! The "unified" accessors in the `GateIndex` trait dispatch through
//! FP4 → native f32 → Q4K backends in priority order, so this single
//! function is **format-blind** — the same code path serves FP4, Q4K,
//! and native f32 vindexes. Adding a new storage format doesn't touch
//! this file.
//!
//! Four specialisations are layered on top, each in its own module:
//!
//! - **Full-K gemv fast path** (`sparse_gemv.rs`): when K ≥ 80% of
//!   num_features, the per-feature loop is mathematically equivalent to
//!   three dense matmuls — route through BLAS gemm / Q4K direct matmul.
//! - **Route selection** (`sparse_route.rs`): the per-position decision
//!   tree picking WHICH features the walk visits (cell router / pools /
//!   selector dispatch).
//! - **Gather-contiguous Q4K kernel** (`sparse_gather.rs`): known-pool
//!   routes gather gate/up/down bytes contiguous and run fused kernels.
//! - **Parallel Q4K down-cache path** (`sparse_parallel.rs`): for
//!   medium-K on Q4K-only vindexes, cache the dequantised down layer
//!   and parallelise feature chunks over rayon.
//!
//! This file keeps the entry point (`walk_ffn_sparse`), its preflight,
//! and the serial per-feature loop — the canonical correctness
//! baseline; it always works because `ffn_row_*` always has *some*
//! backend.

use ndarray::Array2;

use super::helpers::hits_len_ge_intermediate;
use super::thresholds::GATHER_MIN_FEATURES;
use super::WalkFfn;
use crate::vindex::walk_config::FeatureSelector;

impl<'a> WalkFfn<'a> {
    /// Sparse walk FFN — see module docs.
    pub(super) fn walk_ffn_sparse(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];
        let intermediate = self.index.num_features(layer);

        // Prefer native f32 mmap (zero-copy). When no native mmap is
        // available we still run — the inner loops dispatch per-row
        // through `ffn_row_dot` / `ffn_row_scaled_add`, which the
        // GateIndex trait routes to FP4 or Q4K or last-resort native
        // as appropriate. The only thing we can't do with neither
        // native f32 mmap, Q4K storage, nor FP4 storage is the serial
        // per-feature loop — those all fail and bail.
        let up_native = self.index.up_layer_matrix(layer);
        let down_native = self.index.down_layer_matrix(layer);
        let row_fallback = up_native.is_none() || down_native.is_none();
        if row_fallback
            && self.index.interleaved_kquant_layer_data(layer).is_none()
            && !self.index.has_fp4_storage()
        {
            return None;
        }

        let arch = &*self.weights.arch;
        let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // Hint the kernel to start streaming layer N+1's Q4_K/Q6_K bytes
        // into the page cache while we work on N. No-op when there's no
        // Q4_K mmap, no manifest, or `layer+1` is out of range.
        self.index.prefetch_interleaved_kquant_layer(layer + 1);

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        let layer_has_overrides = self.index.has_overrides_at(layer);
        let up_bias_for_layer = if !is_gated {
            arch.ffn_up_bias_key(layer)
                .and_then(|bk| self.weights.vectors.get(&bk).cloned())
        } else {
            None
        };
        let activation_floor = self.config.effective_activation_floor();

        // ── Full-K gemv fast path (`sparse_gemv.rs`) ─────────────────────
        // Skipped when a non-default selector is configured or a per-layer
        // pool restriction is set: in both cases gemv would bypass the
        // alternative selection criterion, so we force the walk.
        let selector_forces_walk = !matches!(self.config.selector, FeatureSelector::GateOnly)
            || self.config.pool_per_layer.is_some()
            || self.config.cell_router.is_some();
        let k_is_full =
            !selector_forces_walk && hits_len_ge_intermediate(&self.config, layer, intermediate);
        if !layer_has_overrides && is_gated && k_is_full {
            if let Some(r) =
                self.sparse_full_k_gemv(layer, x, up_native, down_native, use_gelu, intermediate)
            {
                return Some(r);
            }
        }

        // ── Per-position sparse loop ─────────────────────────────────────
        for s in 0..seq_len {
            let x_row = x.row(s);
            let x_owned = x_row.to_owned();
            let x_slice_owned: Vec<f32>;
            let x_slice: &[f32] = if let Some(sl) = x_row.as_slice() {
                sl
            } else {
                x_slice_owned = x_owned.as_slice().unwrap().to_vec();
                &x_slice_owned
            };

            let top_k = self.top_k_for(layer);

            // ── Gather-contiguous Q4K fast path (`sparse_gather.rs`) ─────
            // For a KNOWN-pool route (precomputed pool or cell-router, no
            // within-pool ranking) the active feature set is decided without
            // gate scores, so we skip the scattered `local_pool_gate_knn` and
            // gather gate+up+down (down from the feature-major sidecar)
            // contiguous, running the fused kernel in one cache-friendly pass.
            // Fixes the ~4× per-row overhead at faithful K; re-gathers every
            // position (the content-addressed pool moves per token). Declines
            // (→ scalar paths) unless gated, no overrides, Q4K up, the down
            // sidecar is loaded, and the route has ≥ GATHER_MIN_FEATURES.
            if is_gated
                && !layer_has_overrides
                && up_native.is_none()
                && !self.config.rank_within_pool
                && self.index.has_down_features_kquant()
            {
                if let Some(feats) = self.gather_route_feats(layer, x_slice, top_k) {
                    if feats.len() >= GATHER_MIN_FEATURES {
                        if let Some((out_vec, acts)) =
                            self.gather_q4k_accumulate(layer, &feats, x_slice, use_gelu, hidden)
                        {
                            let mut out_row = out.row_mut(s);
                            out_row.as_slice_mut().unwrap().copy_from_slice(&out_vec);
                            for (&feat, &a) in feats.iter().zip(&acts) {
                                full_activation[[s, feat]] = a;
                            }
                            self.trace_path(layer, "sparse:gather_q4k");
                            continue;
                        }
                    }
                }
            }

            let t_gate = std::time::Instant::now();
            let hits = self.select_route_hits(layer, &x_owned, x_slice, top_k);
            let gate_knn_ns = t_gate.elapsed().as_nanos() as u64;

            let mut out_row = out.row_mut(s);

            // Parallel Q4K-down-cache path (`sparse_parallel.rs`).
            if self.try_parallel_q4k_down(
                layer,
                &hits,
                x_row,
                x_slice,
                up_native,
                use_gelu,
                down_native.is_none(),
                is_gated,
                layer_has_overrides,
                gate_knn_ns,
                &mut out_row,
            ) {
                continue;
            }

            // Serial per-feature loop — the correctness baseline.
            for (feat, gate_score) in hits {
                let act = if is_gated {
                    let up_ov = if layer_has_overrides {
                        self.index.up_override(layer, feat)
                    } else {
                        None
                    };
                    let up_score = if let Some(up_ov) = up_ov.filter(|o| o.len() == hidden) {
                        ndarray::ArrayView1::from(up_ov).dot(&x_row)
                    } else if let Some(ref up_view) = up_native {
                        up_view.row(feat).dot(&x_row)
                    } else {
                        // Unified dispatch: FP4 → native → Q4K, per GateIndex.
                        self.index.ffn_row_dot(layer, 1, feat, x_slice)?
                    };
                    let activated_gate = if use_gelu {
                        crate::ffn::gelu_tanh(gate_score)
                    } else {
                        gate_score * crate::ffn::sigmoid(gate_score)
                    };
                    activated_gate * up_score
                } else {
                    let mut v = gate_score;
                    if let Some(ref bias) = up_bias_for_layer {
                        if feat < bias.len() {
                            v += bias[feat];
                        }
                    }
                    if use_gelu {
                        crate::ffn::gelu_tanh(v)
                    } else {
                        v * crate::ffn::sigmoid(v)
                    }
                };

                full_activation[[s, feat]] = act;

                if act.abs() > activation_floor {
                    let down_ov = if layer_has_overrides {
                        self.index.down_override(layer, feat)
                    } else {
                        None
                    };
                    if let Some(override_down) = down_ov.filter(|o| o.len() == hidden) {
                        out_row.scaled_add(act, &ndarray::ArrayView1::from(override_down));
                        continue;
                    }
                    if let Some(ref down_view) = down_native {
                        out_row.scaled_add(act, &down_view.row(feat));
                    } else {
                        let out_slice = out_row.as_slice_mut().unwrap();
                        // Unified dispatch: FP4 → native → Q4K-via-cache, per GateIndex.
                        if !self
                            .index
                            .ffn_row_scaled_add(layer, 2, feat, act, out_slice)
                        {
                            return None;
                        }
                    }
                }
            }
        }

        // Down bias
        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        self.trace_path(layer, "sparse:serial");
        Some((out, full_activation))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_vindex, make_test_weights,
    };
    use crate::vindex::{WalkFfn, WalkFfnConfig};
    use ndarray::Array2;

    fn x(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    /// Sparse walk over the Q4K fixture — `up_layer_matrix`/`down_layer_matrix`
    /// both return None (Q4K storage is byte-only) so the function
    /// routes through the row-fallback ladder dispatching via
    /// `ffn_row_dot` / `ffn_row_scaled_add`.
    #[test]
    fn walk_ffn_sparse_routes_through_q4k_fixture() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 8);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size));
        if let Some((out, activation)) = result {
            assert_eq!(out.shape(), &[1, weights.hidden_size]);
            assert_eq!(activation.shape()[0], 1);
        }
    }

    /// M3 regression (2026-07-30 review): `activation_floor` was
    /// documented and settable but read by nothing — the skip
    /// threshold was a hardcoded 1e-10. A floor above every
    /// activation's magnitude must now suppress every down
    /// contribution (zero output) while still recording the
    /// activations themselves.
    #[test]
    fn walk_ffn_sparse_honours_activation_floor() {
        use std::sync::Arc;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        // The fixture has no gate vectors, so plain gate-KNN selects
        // nothing — route through a precomputed pool to get real hits.
        let pool: Vec<usize> = vec![0, 1, 2, 3];
        let cfg = |floor: f32| {
            let mut c = WalkFfnConfig::sparse(weights.num_layers, 4)
                .with_pool_per_layer(Arc::new(vec![pool.clone(); weights.num_layers]))
                .with_precomputed_routing(true);
            c.activation_floor = floor;
            c
        };

        let (out_low, act_low) = WalkFfn::from_config(&weights, &index, cfg(0.0))
            .walk_ffn_sparse(0, &x(1, weights.hidden_size))
            .expect("sparse walk runs on the Q4K fixture");
        assert!(
            out_low.iter().any(|v| v.abs() > 0.0),
            "fixture must produce a non-zero FFN output at floor 0, \
             or this test can't distinguish anything"
        );

        let (out_high, act_high) = WalkFfn::from_config(&weights, &index, cfg(f32::MAX))
            .walk_ffn_sparse(0, &x(1, weights.hidden_size))
            .expect("sparse walk runs on the Q4K fixture");
        assert!(
            out_high.iter().all(|v| *v == 0.0),
            "a floor above every |activation| must suppress all down \
             contributions — the configured floor is being ignored"
        );
        // The floor gates the down accumulate, not activation capture.
        assert_eq!(act_low, act_high);
    }

    /// Sparse walk over the feature-major f32 fixture — `up_layer_matrix`
    /// + `down_layer_matrix` both return Some so the function bypasses
    ///   the row-fallback and goes through the BLAS gemm fast path.
    #[test]
    fn walk_ffn_sparse_routes_through_feature_major_f32_fixture() {
        use crate::test_utils::attach_feature_major_f32_to_test_vindex;
        let weights = make_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn
            .walk_ffn_sparse(0, &x(2, weights.hidden_size))
            .expect("feature-major f32 fixture should produce output");
        let (out, _activation) = result;
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Sparse walk against a bare vindex (no FFN data) returns None —
    /// no native f32, no Q4K, no FP4 → the `row_fallback` guard fires.
    #[test]
    fn walk_ffn_sparse_returns_none_when_no_ffn_data() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size));
        assert!(result.is_none());
    }

    /// Sparse walk against a StarCoder2-shaped arch (Standard FFN +
    /// up_bias) on a feature-major f32 fixture drives the
    /// `up_bias_for_layer = Some(...)` branch AND the non-gated
    /// activation arm of the serial loop.
    #[test]
    fn walk_ffn_sparse_non_gated_arch_uses_up_bias() {
        use crate::test_utils::{
            attach_feature_major_f32_to_test_vindex, make_starcoder2_test_weights,
        };
        let weights = make_starcoder2_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let out = ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size))
            .expect("starcoder2 + feature-major fixture should produce output");
        assert_eq!(out.0.shape(), &[1, weights.hidden_size]);
        assert!(out.0.iter().all(|v| v.is_finite()));
    }
}
