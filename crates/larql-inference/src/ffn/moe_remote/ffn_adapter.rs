//! `RemoteMoeFfn` — an [`FfnBackend`] adapter that lets the KvEngine layer
//! drive CPU remote-MoE decode with a real KV cache.
//!
//! The engine owns attention (and its KV cache) and calls
//! [`FfnBackend::forward_moe_full_layer`] per MoE layer; this adapter
//! computes that layer's MoE FFN block via
//! [`moe_ffn_block_cpu`](crate::vindex::kquant_forward::moe_ffn_block_cpu)
//! — dense `h1` locally + experts `h2` dispatched to the remote shards
//! through [`RemoteMoeBackend`]. This is the engine-routed counterpart to
//! the standalone full-recompute `generate_kquant_cpu_remote` path that
//! closed #146; see the larql-kv "MoE-aware KV engines (C1)" roadmap item.

use larql_compute::ffn::FfnBackend;
use larql_models::ModelWeights;
use ndarray::Array2;

use super::RemoteMoeBackend;
use crate::ffn::WeightFfn;
use crate::vindex::moe_ffn_block_cpu;

/// `FfnBackend` for CPU remote-MoE decode through a `KvEngine`.
///
/// The dense `h1` contribution runs through [`WeightFfn`] (f32 dense FFN over
/// `weights.tensors` — the caller pre-dequantizes the client's Q4K FFN), and
/// the expert `h2` contribution dispatches to the remote shards via
/// `forward_moe_seq`.
///
/// (A `WalkFfn`-based Q4K-direct `h1` was tried 2026-05-29 and reverted: its
/// dense mode runs the per-position sparse-walk machinery → ~8.5× slower than
/// f32 BLAS. The genuine Q4K-direct dense kernel is
/// `kquant_ffn_forward_layer_q8k`; see the bottleneck-diagnosis follow-up.)
///
/// PLE is **not** applied on this path (`moe_ffn_block_cpu` is called with
/// `ple_input = None`), so callers must route Per-Layer-Embedding
/// architectures (Gemma 4 E-series) through the full-recompute path
/// instead. Non-PLE MoE models (Gemma 4 26B-A4B, 31B-MoE) are unaffected.
pub struct RemoteMoeFfn<'a> {
    pub weights: &'a ModelWeights,
    pub remote: &'a RemoteMoeBackend,
}

impl<'a> FfnBackend for RemoteMoeFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.general().forward(layer, x)
    }

    fn forward_observed(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, crate::ffn::FfnActivations) {
        self.general().forward_observed(layer, x)
    }

    fn name(&self) -> &str {
        REMOTE_MOE_FFN_NAME
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Option<Array2<f32>> {
        self.general().forward_moe_full_layer(layer, h_post_attn)
    }
}

impl<'a> RemoteMoeFfn<'a> {
    /// This adapter as the general one. Kept as a delegation rather than a type
    /// alias so the `{ weights, remote }` literal its callers use still
    /// type-checks — the name is load-bearing at one CLI call site and in two
    /// roadmaps.
    fn general(&self) -> MoeFfn<'a> {
        MoeFfn {
            weights: self.weights,
            moe: self.remote,
        }
    }
}

/// The name `RemoteMoeFfn` reports. Unchanged from before the generalisation:
/// it appears in engine diagnostics and is matched on in at least one test.
const REMOTE_MOE_FFN_NAME: &str = "remote-moe";

/// `FfnBackend` for CPU MoE decode through a `KvEngine`, over **any** expert
/// route.
///
/// The engine owns attention and its KV cache and calls
/// [`FfnBackend::forward_moe_full_layer`] per MoE layer; this computes that
/// layer's block — dense `h1` locally, experts `h2` through whichever
/// [`MoeExpertBackend`] is installed.
///
/// The remote-specific adapter above predates this and now delegates to it. The
/// generalisation is what lets a VINDEX3 bound route reach the decode path at
/// all: the block loop's seam became a trait in the composition rung, but the
/// *engine's* adapter still named one concrete backend, so the bound route was
/// reachable only from the full-recompute path.
///
/// PLE is **not** applied here (`moe_ffn_block_cpu` is called with
/// `ple_input = None`), so Per-Layer-Embedding architectures must go through
/// the full-recompute path instead.
pub struct MoeFfn<'a> {
    pub weights: &'a ModelWeights,
    pub moe: &'a dyn crate::ffn::MoeExpertBackend,
}

impl FfnBackend for MoeFfn<'_> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        WeightFfn {
            weights: self.weights,
        }
        .forward(layer, x)
    }

    fn forward_observed(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, crate::ffn::FfnActivations) {
        WeightFfn {
            weights: self.weights,
        }
        .forward_observed(layer, x)
    }

    fn name(&self) -> &str {
        self.moe.name()
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Option<Array2<f32>> {
        Some(moe_ffn_block_cpu(
            self.weights,
            h_post_attn,
            layer,
            &WeightFfn {
                weights: self.weights,
            },
            None,
            Some(self.moe),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_gemma4_moe_weights;
    use ndarray::Array2;

    /// `forward_moe_full_layer` runs the MoE FFN block: dense `h1` + experts
    /// via the (disconnected → zero) remote + combine. Asserts a full, finite
    /// layer output of the right shape.
    #[test]
    fn forward_moe_full_layer_returns_finite_combined_output() {
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let ffn = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        let h_post_attn = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let out = ffn
            .forward_moe_full_layer(0, &h_post_attn)
            .expect("RemoteMoeFfn always returns Some");
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `forward` / `forward_with_activation` run the dense FFN fallback;
    /// `name` is stable.
    #[test]
    fn dense_fallbacks_and_name() {
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let ffn = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        assert_eq!(ffn.name(), "remote-moe");
        let x = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let dense = ffn.forward(0, &x);
        assert_eq!(dense.shape()[0], 2);
        assert!(dense.iter().all(|v| v.is_finite()));
        let (out, obs) = ffn.forward_observed(0, &x);
        assert_eq!(out.shape()[0], 2);
        let act = obs.into_dense().expect("dense fallback observes densely");
        assert_eq!(act.shape()[0], 2);
    }

    /// The generalisation must not have changed the remote adapter. Same
    /// weights, same input, same output — the delegation relocates the call and
    /// nothing else, which is the same property the block-loop seam had to
    /// prove in the composition rung.
    #[test]
    fn the_remote_adapter_still_equals_the_general_one() {
        let weights = make_test_gemma4_moe_weights();
        let remote = RemoteMoeBackend::new_disconnected();
        let specific = RemoteMoeFfn {
            weights: &weights,
            remote: &remote,
        };
        let general = MoeFfn {
            weights: &weights,
            moe: &remote,
        };
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        assert_eq!(
            specific.forward_moe_full_layer(0, &h),
            general.forward_moe_full_layer(0, &h)
        );
        assert_eq!(specific.forward(0, &h), general.forward(0, &h));
        // The name is the one thing that deliberately differs: the remote
        // adapter keeps its historical name for diagnostics, the general one
        // reports whichever route it carries.
        assert_eq!(specific.name(), REMOTE_MOE_FFN_NAME);
        assert_eq!(general.name(), crate::ffn::MoeExpertBackend::name(&remote));
    }

    /// The point of the generalisation: a VINDEX3 bound route can now drive the
    /// engine's FFN seam, which the remote-typed adapter made impossible.
    #[test]
    fn a_bound_route_can_drive_the_engine_adapter() {
        let weights = make_test_gemma4_moe_weights();
        // The fixture stores BF16 experts, so the reference pairing is the one
        // that executes; the production Q4_K pairing would refuse them.
        let bound = crate::ffn::BoundMoeBackend::reference();
        let ffn = MoeFfn {
            weights: &weights,
            moe: &bound,
        };
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let out = ffn
            .forward_moe_full_layer(0, &h)
            .expect("the adapter always returns Some");
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|v| v.abs() > f32::EPSILON));
    }

    /// A refused route must not silently contribute zeros through the adapter.
    /// `moe_ffn_block_cpu` logs and leaves `h2` zero on error, so this pins that
    /// the dense half still flows — the failure is visible as a *different*
    /// output rather than as a plausible one.
    #[test]
    fn a_refusing_route_still_returns_the_dense_half() {
        let weights = make_test_gemma4_moe_weights();
        // Q4_K kernels against a BF16 store: refused at bind.
        let refusing = crate::ffn::BoundMoeBackend::production();
        let executing = crate::ffn::BoundMoeBackend::reference();
        let h = Array2::<f32>::from_elem((2, weights.hidden_size), 0.1);
        let refused = MoeFfn {
            weights: &weights,
            moe: &refusing,
        }
        .forward_moe_full_layer(0, &h)
        .expect("returns Some even when the route refused");
        let executed = MoeFfn {
            weights: &weights,
            moe: &executing,
        }
        .forward_moe_full_layer(0, &h)
        .expect("executes");
        assert_ne!(
            refused, executed,
            "a refused expert route must not produce the same output as one that ran"
        );
    }
}
