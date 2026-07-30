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
