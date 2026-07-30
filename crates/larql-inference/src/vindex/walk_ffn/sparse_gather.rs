//! Gather-contiguous Q4K kernel for known-pool routes (tasks #24/#25).
//!
//! For a route whose feature set is decided WITHOUT gate scores
//! (precomputed pool or cell router, no within-pool ranking), the
//! sparse walk gathers the selected rows' gate/up/down Q4K **bytes**
//! into contiguous buffers and runs the fused row kernels in one
//! cache-friendly pass — no f32 materialisation. Traced as
//! `sparse:gather_q4k`.

use rayon::prelude::*;

use super::WalkFfn;

impl<'a> WalkFfn<'a> {
    /// The known-pool route feature set for the gather fast path:
    /// cell-router pool if a router is attached, else the precomputed
    /// per-layer pool truncated to top-K. `None` when the position has
    /// no gate-free route (the caller then runs the scored paths).
    pub(super) fn gather_route_feats(
        &self,
        layer: usize,
        x_slice: &[f32],
        top_k: usize,
    ) -> Option<Vec<usize>> {
        if let Some(router) = self.config.cell_router.as_ref() {
            router.pool_for(layer, x_slice).map(|p| p.to_vec())
        } else if self.config.precomputed_routing {
            self.config.pool_per_layer.as_ref().and_then(|ppl| {
                ppl.get(layer).map(|p| {
                    let mut v = p.clone();
                    v.truncate(top_k);
                    v
                })
            })
        } else {
            None
        }
    }

    /// Gather-contiguous Q4K accumulate — faithful-K fast-path kernel
    /// (task #24; made production-correct by the feature-major down
    /// sidecar, task #25 — see the down paragraph below).
    ///
    /// The scattered per-feature loop pays ~4× per-row overhead at large K
    /// (cache-unfriendly gather + per-hit dispatch), so it loses to dense
    /// above ~20% density. This gathers the selected rows' up/down Q4K
    /// **bytes** into contiguous buffers and runs the *same* fused NEON
    /// kernels (`row_dot` / `row_scaled_add`) — no f32 materialisation.
    /// Contiguity recovers the cost win: at K=4096 (40%) the kernel runs
    /// ~1.4× faster than dense vs the scattered path's 0.80×
    /// (`examples/walk_ffn_gather_gemm.rs`).
    ///
    /// `up` (`slices[1]`) is feature-major Q4K (gatherable). **`down`** must
    /// come from the **feature-major down sidecar**
    /// (`down_features_kquant.bin` via `down_features_q4k_layer_data`) — the
    /// interleaved down is stored *transposed* `[hidden × intermediate]`, so a
    /// feature's down vector is a strided column there, not a gatherable row.
    /// Reading the transposed slab with row striding was the task-#24
    /// "not yet correct for production down" caveat; task #25 resolved it by
    /// requiring the sidecar here (`?` below) and validating against dense
    /// (`examples/walk_ffn_gather_gemm.rs`, |err|/‖ref‖ ≈ 6e-3 — the Q4K
    /// re-quantisation of the Q6_K down, not a layout error). Returns `None`
    /// (caller falls back to the correct scalar paths) when the sidecar is
    /// absent — pinned by `gather_q4k_accumulate_declines_without_down_sidecar`.
    ///
    /// Returns `(out[hidden], acts[feats.len()])`. **Gate is recomputed** from
    /// gathered gate bytes (not taken from any prior scattered scoring) so the
    /// whole gate/up/down pass is contiguous — `feats` need only be the route's
    /// feature indices.
    #[allow(clippy::type_complexity)]
    pub(super) fn gather_q4k_accumulate(
        &self,
        layer: usize,
        feats: &[usize],
        x_slice: &[f32],
        use_gelu: bool,
        hidden: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        let slices = self.index.interleaved_kquant_layer_data(layer)?;
        let gate_info = larql_vindex::quant::registry::lookup(slices[0].1)?;
        let up_info = larql_vindex::quant::registry::lookup(slices[1].1)?;
        let gate_rd = gate_info.row_dot?;
        let up_rd = up_info.row_dot?;
        let gbpr = gate_info.bytes_per_row(hidden)?;
        let ubpr = up_info.bytes_per_row(hidden)?;
        let gate_b = slices[0].0;
        let up_b = slices[1].0;
        // Down from the feature-major sidecar (gatherable rows).
        let (down_b, down_fmt, padded_width) = self.index.down_features_q4k_layer_data(layer)?;
        let down_info = larql_vindex::quant::registry::lookup(down_fmt)?;
        let down_sa = down_info.row_scaled_add?;
        let dbpr = down_info.bytes_per_row(padded_width)?;
        let k = feats.len();
        if k == 0 {
            return None;
        }

        // Gather gate + up + down bytes for the route's rows into contiguous
        // buffers (sequential layout = cache-friendly fused kernel passes).
        let mut gg = vec![0u8; k * gbpr];
        let mut gu = vec![0u8; k * ubpr];
        let mut gd = vec![0u8; k * dbpr];
        for (i, &feat) in feats.iter().enumerate() {
            let (gs, ge) = (feat * gbpr, feat * gbpr + gbpr);
            let (us, ue) = (feat * ubpr, feat * ubpr + ubpr);
            let (ds, de) = (feat * dbpr, feat * dbpr + dbpr);
            if ge > gate_b.len() || ue > up_b.len() || de > down_b.len() {
                return None; // out-of-range feature — bail to the safe path
            }
            gg[i * gbpr..(i + 1) * gbpr].copy_from_slice(&gate_b[gs..ge]);
            gu[i * ubpr..(i + 1) * ubpr].copy_from_slice(&up_b[us..ue]);
            gd[i * dbpr..(i + 1) * dbpr].copy_from_slice(&down_b[ds..de]);
        }

        // gate + up scores: fused row-dot over contiguous rows, parallel.
        let gate_s: Vec<f32> = (0..k)
            .into_par_iter()
            .map(|i| gate_rd(&gg[i * gbpr..(i + 1) * gbpr], x_slice).unwrap_or(0.0))
            .collect();
        let up_s: Vec<f32> = (0..k)
            .into_par_iter()
            .map(|i| up_rd(&gu[i * ubpr..(i + 1) * ubpr], x_slice).unwrap_or(0.0))
            .collect();
        let acts: Vec<f32> = gate_s
            .iter()
            .zip(&up_s)
            .map(|(&g, &u)| {
                let ag = if use_gelu {
                    crate::ffn::gelu_tanh(g)
                } else {
                    g * crate::ffn::sigmoid(g)
                };
                ag * u
            })
            .collect();

        // down accumulate: fused scaled-add over contiguous rows, chunked.
        let activation_floor = self.config.effective_activation_floor();
        let n_threads = rayon::current_num_threads().max(1);
        let chunk = k.div_ceil(n_threads).max(1);
        let partials: Vec<Vec<f32>> = (0..k)
            .collect::<Vec<_>>()
            .par_chunks(chunk)
            .map(|ch| {
                let mut part = vec![0.0f32; hidden];
                for &i in ch {
                    if acts[i].abs() > activation_floor {
                        let _ = down_sa(&gd[i * dbpr..(i + 1) * dbpr], acts[i], &mut part);
                    }
                }
                part
            })
            .collect();
        let mut out = vec![0.0f32; hidden];
        for p in &partials {
            for (o, v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
        Some((out, acts))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
    use crate::vindex::{WalkFfn, WalkFfnConfig};
    use ndarray::Array2;

    fn x(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    /// Safety property (task #25): `gather_q4k_accumulate` must **decline**
    /// (return None) when the feature-major down sidecar is absent — the
    /// interleaved down is transposed and not gatherable, so the caller falls
    /// back to the correct scalar paths. The test fixture ships no sidecar.
    /// (With-sidecar correctness is validated against dense on a real vindex
    /// in `examples/walk_ffn_gather_gemm.rs`.)
    #[test]
    fn gather_q4k_accumulate_declines_without_down_sidecar() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let hidden = weights.hidden_size;
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let x1 = x(1, hidden);
        let x_slice = x1.row(0).to_vec();
        assert!(!index.has_down_features_kquant());
        assert!(
            ffn.gather_q4k_accumulate(0, &[0, 1, 2, 3], &x_slice, false, hidden)
                .is_none(),
            "gather must decline without the feature-major down sidecar"
        );
    }
}
