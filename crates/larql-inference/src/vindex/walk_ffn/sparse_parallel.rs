//! Parallel Q4K-down-cache branch of the sparse walk.
//!
//! For a medium-large hit count (≥ `PARALLEL_DOWN_MIN_HITS`) on a
//! Q4K-only vindex (no native f32 down), the down matrix transposition
//! cost justifies caching the whole dequantised down layer
//! (`kquant_ffn_layer(layer, 2)`) and parallelising the per-feature
//! up-dot + scaled-add scan over rayon chunks. Traced as
//! `sparse:parallel_q4k_down`; phase timings recorded when a
//! `PhaseTimingsHandle` is attached.
//!
//! Known limitation (2026-07-30 review, M2): this branch does not write
//! `full_activation` — activation output is silently zero for positions
//! it handles. Fixed by the forward/forward_observed split (Tier 2).

use rayon::prelude::*;

use super::thresholds::PARALLEL_DOWN_MIN_HITS;
use super::WalkFfn;

impl<'a> WalkFfn<'a> {
    /// Try the rayon-parallel Q4K-down-cache scan for one position.
    /// Returns `true` when the branch fired (the position's output row
    /// is complete — the caller `continue`s); `false` when the caller
    /// must run the serial per-feature loop instead.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_parallel_q4k_down(
        &self,
        layer: usize,
        hits: &[(usize, f32)],
        x_row: ndarray::ArrayView1<'_, f32>,
        x_slice: &[f32],
        up_native: Option<ndarray::ArrayView2<'_, f32>>,
        use_gelu: bool,
        down_native_absent: bool,
        is_gated: bool,
        layer_has_overrides: bool,
        gate_knn_ns: u64,
        out_row: &mut ndarray::ArrayViewMut1<'_, f32>,
    ) -> bool {
        let hidden = x_slice.len();
        let activation_floor = self.config.effective_activation_floor();

        // Only used when feature count is medium-large and no native
        // down exists.
        let parallelisable = !layer_has_overrides
            && is_gated
            && hits.len() >= PARALLEL_DOWN_MIN_HITS
            && down_native_absent;
        let t_cache = std::time::Instant::now();
        let down_cache_local: Option<std::sync::Arc<Vec<f32>>> = if parallelisable {
            self.index.kquant_ffn_layer(layer, 2)
        } else {
            None
        };
        let cache_fetch_ns = t_cache.elapsed().as_nanos() as u64;
        let Some(down_arc) = down_cache_local.as_ref().filter(|_| parallelisable) else {
            return false;
        };

        let down_data: &[f32] = down_arc.as_slice();
        let up_slices = self.index.interleaved_kquant_layer_data(layer);
        // Resolve up via the registry — accepts Q4_K, Q6_K, and
        // any future K-quant rather than hardcoding Q4_K-only.
        let up_q4k: Option<(&[u8], &larql_vindex::quant::registry::QuantFormatInfo)> =
            match (up_native.as_ref(), up_slices) {
                (Some(_), _) => None,
                (None, Some(s)) => {
                    larql_vindex::quant::registry::lookup(s[1].1).map(|info| (s[1].0, info))
                }
                _ => None,
            };
        let n_threads = rayon::current_num_threads().max(1);
        let chunk_size = hits.len().div_ceil(n_threads);
        let up_native_ref = up_native.as_ref();

        let t_scan = std::time::Instant::now();
        let partials: Vec<Vec<f32>> = hits
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut partial = vec![0.0f32; hidden];
                for &(feat, gate_score) in chunk {
                    let up_score = if let Some(up_view) = up_native_ref {
                        up_view.row(feat).dot(&x_row)
                    } else if let Some((up_bytes, info)) = up_q4k {
                        let row_dot = info.row_dot.expect("registry: row_dot");
                        let bytes_per_row = info
                            .bytes_per_row(hidden)
                            .expect("registry: bytes_per_row aligned");
                        let start = feat * bytes_per_row;
                        let end = start + bytes_per_row;
                        row_dot(&up_bytes[start..end], x_slice).unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    let activated_gate = if use_gelu {
                        crate::ffn::gelu_tanh(gate_score)
                    } else {
                        gate_score * crate::ffn::sigmoid(gate_score)
                    };
                    let act = activated_gate * up_score;
                    if act.abs() > activation_floor {
                        let row_start = feat * hidden;
                        let down_row = &down_data[row_start..row_start + hidden];
                        let mut pv = ndarray::ArrayViewMut1::from(partial.as_mut_slice());
                        let dv = ndarray::ArrayView1::from(down_row);
                        pv.scaled_add(act, &dv);
                    }
                }
                partial
            })
            .collect();
        let parallel_scan_ns = t_scan.elapsed().as_nanos() as u64;

        let t_reduce = std::time::Instant::now();
        let out_slice = out_row.as_slice_mut().unwrap();
        for p in &partials {
            for i in 0..hidden {
                out_slice[i] += p[i];
            }
        }
        let reduce_ns = t_reduce.elapsed().as_nanos() as u64;

        if let Some(h) = &self.phase_timings {
            use std::sync::atomic::Ordering::Relaxed;
            h.gate_knn_ns.fetch_add(gate_knn_ns, Relaxed);
            h.cache_fetch_ns.fetch_add(cache_fetch_ns, Relaxed);
            h.parallel_scan_ns.fetch_add(parallel_scan_ns, Relaxed);
            h.reduce_ns.fetch_add(reduce_ns, Relaxed);
            h.calls.fetch_add(1, Relaxed);
        }

        self.trace_path(layer, "sparse:parallel_q4k_down");
        true
    }
}

#[cfg(test)]
mod tests {
    //! Serial-vs-parallel parity (2026-07-30 review, roadmap item 20,
    //! test 3): the parallel Q4K-down branch shares the serial loop's
    //! math — same gate scores, same up row-dots, same dequantised down
    //! cache — so its output must match a serial walk over the same
    //! feature set. The serial baseline is obtained by *splitting the
    //! route in half*: per-feature FFN contributions are independent, so
    //! `out(P)` = `out(P1) + out(P2)` for disjoint `P1 ∪ P2 = P`, and
    //! each half stays below `PARALLEL_DOWN_MIN_HITS` → serial loop.

    use std::sync::Arc;

    use ndarray::Array2;

    use super::super::thresholds::PARALLEL_DOWN_MIN_HITS;
    use crate::test_utils::{
        attach_up_features_f32_only_to_test_vindex, make_test_q4k_vindex,
        make_test_q4k_weights_wide, Q4K_TEST_INTER_WIDE,
    };
    use crate::vindex::{WalkFfn, WalkFfnConfig};

    fn x(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    /// Precomputed-pool walk over `pool` with K = pool.len(). The pool
    /// keeps the gemv rewrite off (`pool_per_layer` forces the walk) and
    /// pins the exact feature set, so runs over different pools are
    /// comparable feature-by-feature.
    fn walk_pool<'a>(
        weights: &'a larql_models::ModelWeights,
        index: &'a larql_vindex::VectorIndex,
        pool: Vec<usize>,
    ) -> WalkFfn<'a> {
        let k = pool.len();
        let cfg = WalkFfnConfig::sparse(weights.num_layers, k)
            .with_pool_per_layer(Arc::new(vec![pool; weights.num_layers]))
            .with_precomputed_routing(true);
        WalkFfn::from_config(weights, index, cfg).with_dispatch_trace()
    }

    fn rel_l2(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        let num: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt();
        let den: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        num / den.max(f32::MIN_POSITIVE)
    }

    /// Route of `PARALLEL_DOWN_MIN_HITS` features on the wide Q4K-only
    /// fixture fires `sparse:parallel_q4k_down`; the output must equal
    /// the sum of two serial half-route walks (per-feature linearity).
    /// Also pins the M2 known limitation: the parallel branch does not
    /// write `full_activation` (all-zero) while the serial halves do —
    /// this assertion flips when the Tier-2 forward/forward_observed
    /// split fixes M2.
    #[test]
    fn parallel_q4k_down_matches_serial_half_route_sum() {
        let weights = make_test_q4k_weights_wide();
        let index = make_test_q4k_vindex(&weights);
        assert_eq!(
            index.num_features(0),
            Q4K_TEST_INTER_WIDE,
            "wide fixture must expose the full 768-feature layer"
        );
        let hidden = weights.hidden_size;
        let input = x(1, hidden);

        let full: Vec<usize> = (0..PARALLEL_DOWN_MIN_HITS).collect();
        let (first, second) = full.split_at(PARALLEL_DOWN_MIN_HITS / 2);

        let timings = Arc::new(super::super::timings::PhaseTimingsHandle::default());
        let par_ffn =
            walk_pool(&weights, &index, full.clone()).with_phase_timings(Arc::clone(&timings));
        let (out_par, act_par) = par_ffn
            .walk_ffn_sparse(0, &input)
            .expect("wide Q4K fixture supports the sparse walk");
        let par_trace = par_ffn.take_dispatch_trace();
        assert!(
            par_trace
                .iter()
                .any(|e| e.path == "sparse:parallel_q4k_down"),
            "route of {PARALLEL_DOWN_MIN_HITS} hits on a Q4K-only vindex \
             must fire the parallel down branch, got {par_trace:?}"
        );
        assert_eq!(
            timings.calls.load(std::sync::atomic::Ordering::Relaxed),
            1,
            "one position → one parallel_q4k_down call recorded"
        );

        let serial = |pool: &[usize]| {
            let ffn = walk_pool(&weights, &index, pool.to_vec());
            let r = ffn
                .walk_ffn_sparse(0, &input)
                .expect("serial half-route runs");
            let trace = ffn.take_dispatch_trace();
            assert!(
                trace.iter().all(|e| e.path != "sparse:parallel_q4k_down"),
                "half routes stay below PARALLEL_DOWN_MIN_HITS → serial"
            );
            r
        };
        let (out_a, act_a) = serial(first);
        let (out_b, act_b) = serial(second);

        let serial_sum = &out_a + &out_b;
        let err = rel_l2(&out_par, &serial_sum);
        // Same per-feature math and the same dequant cache; only the
        // floating-point accumulation order differs (rayon partials vs
        // serial scaled-adds), so parity is tight.
        assert!(
            err < 1e-4,
            "parallel vs serial-half-sum rel L2 = {err} (expected < 1e-4)"
        );

        // M2 pin (module doc "Known limitation"): parallel writes no
        // activations; the serial halves observed real ones.
        assert!(
            act_par.iter().all(|v| *v == 0.0),
            "M2: the parallel branch is documented to leave full_activation zero"
        );
        assert!(
            act_a.iter().any(|v| v.abs() > 0.0) && act_b.iter().any(|v| v.abs() > 0.0),
            "serial halves must record non-zero activations for the parity to mean anything"
        );
    }

    /// Same parity with feature-major f32 **up** attached: the parallel
    /// scan's `up_native` arm (BLAS row dot) instead of the Q4K row-dot
    /// arm. Down still comes from the Q4K dequant cache on both sides.
    #[test]
    fn parallel_q4k_down_native_up_arm_matches_serial_half_route_sum() {
        let weights = make_test_q4k_weights_wide();
        let mut index = make_test_q4k_vindex(&weights);
        attach_up_features_f32_only_to_test_vindex(&weights, &mut index);
        assert!(index.up_layer_matrix(0).is_some(), "native up attached");
        assert!(
            index.down_layer_matrix(0).is_none(),
            "down must stay Q4K-only or the parallel branch won't fire"
        );
        let hidden = weights.hidden_size;
        let input = x(1, hidden);

        let full: Vec<usize> = (0..PARALLEL_DOWN_MIN_HITS).collect();
        let (first, second) = full.split_at(PARALLEL_DOWN_MIN_HITS / 2);

        let par_ffn = walk_pool(&weights, &index, full.clone());
        let (out_par, _act) = par_ffn
            .walk_ffn_sparse(0, &input)
            .expect("wide Q4K fixture with native up supports the sparse walk");
        assert!(
            par_ffn
                .take_dispatch_trace()
                .iter()
                .any(|e| e.path == "sparse:parallel_q4k_down"),
            "native-up variant must still take the parallel down branch"
        );

        let serial = |pool: &[usize]| {
            walk_pool(&weights, &index, pool.to_vec())
                .walk_ffn_sparse(0, &input)
                .expect("serial half-route runs")
                .0
        };
        let serial_sum = &serial(first) + &serial(second);
        let err = rel_l2(&out_par, &serial_sum);
        assert!(
            err < 1e-4,
            "native-up parallel vs serial-half-sum rel L2 = {err} (expected < 1e-4)"
        );
    }
}
