//! Q4_0 interleaved walk. C kernel with `vdotq_s32` for gate/up, scalar
//! kernel for down. Reads ~44 MB per layer (vs 315 MB for f32
//! interleaved) — 7× less data to page in, same BLAS speed warm.
//!
//! Batched GPU path (when the backend implements
//! `q4_matvec_pair_batch` — probed by calling it, since
//! `supports_quant` can't distinguish per-row kernels from the batch
//! API): one GPU submission for gate+up across all seq positions,
//! followed by one vecmat per position for down. C kernel path is the
//! CPU fallback, including for backends that decline the batch call.

use ndarray::Array2;

use super::WalkFfn;

impl<'a> WalkFfn<'a> {
    pub(super) fn walk_ffn_q4_interleaved(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        use larql_compute::cpu::ops::{q4_matvec, q4_vecmat};

        let q4_mmap = self.index.interleaved_q4_mmap_ref()?;
        let intermediate = self.index.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];

        let q4_bytes_per_matrix = larql_compute::QuantFormat::Q4_0
            .packed_matrix_bytes(intermediate, hidden)
            .expect("Q4_0 interleaved FFN format must have packed geometry");
        let q4_bytes_per_layer = q4_bytes_per_matrix * 3;
        let layer_start = layer * q4_bytes_per_layer;

        let gate_q4 = &q4_mmap[layer_start..layer_start + q4_bytes_per_matrix];
        let up_q4 =
            &q4_mmap[layer_start + q4_bytes_per_matrix..layer_start + 2 * q4_bytes_per_matrix];
        let down_q4 =
            &q4_mmap[layer_start + 2 * q4_bytes_per_matrix..layer_start + 3 * q4_bytes_per_matrix];

        self.index.prefetch_interleaved_q4_layer(layer + 1);

        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        // Batch capability is probed by CALLING `q4_matvec_pair_batch`,
        // never via `supports_quant`: CpuBackend advertises Q4-family
        // matvec kernels but leaves the batch API at its trait default
        // (`None`). A `None` here falls through to the C-kernel loop —
        // the pre-fix code filtered on `supports_quant(Q4_K)` and
        // unwrapped, panicking on any non-Metal backend.
        let batch = self.backend.and_then(|be| {
            let x_flat = x.as_slice()?;
            be.q4_matvec_pair_batch(gate_q4, up_q4, x_flat, seq_len, intermediate, hidden)
                .map(|pair| (be, pair))
        });

        if let Some((be, (all_gate, all_up))) = batch {
            // Metal: ONE GPU submission for all gate+up across ALL seq positions
            let mut all_activation: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
            for s in 0..seq_len {
                let mut activation = vec![0.0f32; intermediate];
                for i in 0..intermediate {
                    let g = all_gate[s][i];
                    let u = all_up[s][i];
                    activation[i] = if use_gelu {
                        crate::ffn::gelu_tanh(g) * u
                    } else {
                        g * crate::ffn::sigmoid(g) * u
                    };
                    full_activation[[s, i]] = activation[i];
                }
                all_activation.push(activation);
            }

            for (s, activation_row) in all_activation.iter().enumerate().take(seq_len) {
                // A backend that batches gate/up should also vecmat down;
                // if it declines, finish the row on the C kernel rather
                // than losing the layer (or panicking) mid-forward.
                let down_result = be
                    .q4_vecmat(activation_row, down_q4, intermediate, hidden)
                    .unwrap_or_else(|| {
                        q4_vecmat::dispatch(activation_row, down_q4, intermediate, hidden)
                    });
                let mut out_row = out.row_mut(s);
                for j in 0..hidden {
                    out_row[j] = down_result[j];
                }
            }
            self.trace_path(layer, "interleaved_q4:metal");
        } else {
            for s in 0..seq_len {
                let x_row = x.row(s);
                let x_slice = x_row.as_slice().unwrap();

                let gate_scores = q4_matvec::dispatch(gate_q4, x_slice, intermediate, hidden);
                let up_scores = q4_matvec::dispatch(up_q4, x_slice, intermediate, hidden);

                let mut activation = vec![0.0f32; intermediate];
                for i in 0..intermediate {
                    let g = gate_scores[i];
                    let u = up_scores[i];
                    activation[i] = if use_gelu {
                        crate::ffn::gelu_tanh(g) * u
                    } else {
                        g * crate::ffn::sigmoid(g) * u
                    };
                    full_activation[[s, i]] = activation[i];
                }

                let down_result = q4_vecmat::dispatch(&activation, down_q4, intermediate, hidden);
                let mut out_row = out.row_mut(s);
                for j in 0..hidden {
                    out_row[j] = down_result[j];
                }
            }
            self.trace_path(layer, "interleaved_q4:cpu");
        }

        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, full_activation))
    }
}

#[cfg(test)]
mod tests {
    //! First tests for this file (2026-07-30 review, finding H2): the
    //! pre-fix path filtered the GPU branch on `supports_quant(Q4_K)` —
    //! true for `CpuBackend`, which nevertheless leaves
    //! `q4_matvec_pair_batch` at its trait default (`None`) — then
    //! `.unwrap()`ed the batch call: panic on the first Q4_0 layer for
    //! any non-Metal backend.
    use larql_compute::cpu::ops::q4_common::quantize_q4_0;
    use larql_compute::CpuBackend;
    use larql_models::test_fixtures::arc_mmap_from_bytes;
    use ndarray::Array2;

    use crate::ffn::FfnBackend;
    use crate::test_utils::Q4KTestFixtures;
    use crate::vindex::WalkFfn;

    /// Feature count for the synthetic Q4_0 layer. Small keeps the test
    /// fast; `hidden` (from the fixture) satisfies the Q4_0 kernel's
    /// 32-element block requirement.
    const INTERMEDIATE: usize = 8;

    /// One-layer Q4_0-only vindex: raw `[gate|up|down]` slab, no
    /// manifest (the format has none), gate matrix sized so
    /// `num_features(0) == INTERMEDIATE`.
    fn q4_0_index(hidden: usize) -> larql_vindex::VectorIndex {
        let n = INTERMEDIATE * hidden;
        let mat = |seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i + seed) % 13) as f32 * 0.01 - 0.06)
                .collect::<Vec<f32>>()
        };
        let mut payload = Vec::new();
        for m in [mat(0), mat(5), mat(9)] {
            payload.extend_from_slice(&quantize_q4_0(&m));
        }
        let gate_vectors = vec![Some(Array2::<f32>::zeros((INTERMEDIATE, hidden)))];
        let down_meta = vec![None];
        let mut index = larql_vindex::VectorIndex::new(gate_vectors, down_meta, 1, hidden);
        let mmap = arc_mmap_from_bytes(&payload);
        std::sync::Arc::make_mut(&mut index.storage).set_interleaved_q4(mmap);
        index
    }

    fn input(hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (1, hidden),
            (0..hidden).map(|i| (i as f32 + 1.0) * 0.001).collect(),
        )
        .unwrap()
    }

    #[test]
    fn cpu_backend_runs_c_kernel_instead_of_panicking() {
        let f = Q4KTestFixtures::build();
        let hidden = f.weights.hidden_size;
        let index = q4_0_index(hidden);
        let cpu = CpuBackend;

        let walk = WalkFfn::new_unlimited_with_backend(&f.weights, &index, &cpu)
            .with_dispatch_trace();
        let (out, activation) = walk
            .walk_ffn_q4_interleaved(0, &input(hidden))
            .expect("Q4_0 path must succeed with a CPU backend");

        assert_eq!(out.shape(), &[1, hidden]);
        assert_eq!(activation.shape(), &[1, INTERMEDIATE]);
        assert!(out.iter().all(|v| v.is_finite()));

        let trace = walk.take_dispatch_trace();
        assert_eq!(trace.len(), 1);
        assert_eq!(
            trace[0].path, "interleaved_q4:cpu",
            "CpuBackend must take the C-kernel branch, not the batch branch"
        );
    }

    #[test]
    fn cpu_backend_output_matches_backendless_c_kernel() {
        // A backend whose batch call declines must produce EXACTLY the
        // no-backend result — both run the same C kernel on the same
        // bytes, so this is equality, not tolerance.
        let f = Q4KTestFixtures::build();
        let hidden = f.weights.hidden_size;
        let index = q4_0_index(hidden);
        let cpu = CpuBackend;
        let x = input(hidden);

        let with_backend = WalkFfn::new_unlimited_with_backend(&f.weights, &index, &cpu)
            .walk_ffn_q4_interleaved(0, &x)
            .expect("backend run succeeds");
        let backendless = WalkFfn::new_unlimited(&f.weights, &index)
            .walk_ffn_q4_interleaved(0, &x)
            .expect("backendless run succeeds");

        assert_eq!(with_backend.0, backendless.0);
        assert_eq!(with_backend.1, backendless.1);
    }

    #[test]
    fn routing_ladder_admits_cpu_backend_to_q4_0_path() {
        // The ladder gate (walk_ffn/mod.rs step 5) now checks the format
        // the data actually is (Q4_0). CpuBackend advertises Q4_0, so a
        // full forward routes here and completes on the C kernel.
        let f = Q4KTestFixtures::build();
        let hidden = f.weights.hidden_size;
        let index = q4_0_index(hidden);
        let cpu = CpuBackend;

        let walk = WalkFfn::new_unlimited_with_backend(&f.weights, &index, &cpu)
            .with_dispatch_trace();
        let out = walk.forward(0, &input(hidden));
        assert!(out.iter().all(|v| v.is_finite()));

        let trace = walk.take_dispatch_trace();
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].path, "interleaved_q4:cpu");
    }
}
