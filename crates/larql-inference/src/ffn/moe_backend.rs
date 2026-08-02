//! Who computes a hybrid-MoE layer's expert contribution.
//!
//! The block loop — attention, KV, the dense slab, the norms, PLE, the layer
//! scalar — is shared by every route. What varies is one step:
//!
//! ```text
//! h_post_attn
//!   ├── dense FFN slab        shared
//!   └── expert contribution   this trait
//!         ├── in-process      cpu_moe_forward over MoeLayerWeights (the default)
//!         ├── remote          RemoteMoeBackend, experts fetched from shards
//!         └── bound           a VINDEX3 BoundMoeOperation
//! ```
//!
//! # Why a trait rather than a second parameter
//!
//! `moe_ffn_block_cpu` previously took `Option<&RemoteMoeBackend>` — a
//! concrete type, so the only way to add a third route was another optional
//! parameter and a rule that at most one may be `Some`. That rule is not
//! expressible in the type, which means it is a rule someone eventually
//! breaks, and the failure mode is a layer whose expert contribution comes
//! from a route nobody chose.
//!
//! One `Option<&dyn MoeExpertBackend>` makes "exactly one route, or the
//! in-process default" the only representable state.
//!
//! # The backend derives its own operands
//!
//! The caller passes `weights` and a layer, not a prepared router. The remote
//! backend needs the router only, because it fetches experts from shards; the
//! bound backend needs the whole layer. Preparing the union at the call site
//! would make the block loop know what each route reads, which is exactly the
//! coupling this trait removes.

use larql_models::ModelWeights;
use ndarray::Array2;

use super::moe_remote::RemoteMoeError;

/// Why a backend could not produce an expert contribution.
///
/// Two variants rather than a string, so a refusal keeps the structure the
/// route gave it. In particular a bound route's [`ExecutionError`] carries
/// `refusal()`, which distinguishes an operand that lives elsewhere from a
/// binding that is wrong — a distinction that a formatted message destroys.
///
/// No `PartialEq`: [`RemoteMoeError`] does not claim it, and neither should a
/// type that wraps it.
///
/// [`ExecutionError`]: larql_vindex::runtime::ExecutionError
#[derive(Debug, thiserror::Error)]
pub enum MoeBackendError {
    #[error("remote expert dispatch failed: {0}")]
    Remote(#[from] RemoteMoeError),
    #[error("bound expert execution failed: {0}")]
    Bound(#[from] larql_vindex::runtime::ExecutionError),
}

/// A route that computes one hybrid-MoE layer's expert contribution.
pub trait MoeExpertBackend {
    /// Expert contribution for every position of `h`, shaped `[seq_len, hidden]`.
    ///
    /// Returns zeros — not an error — when the layer has no expert weights to
    /// route into. That is the in-process path's behaviour and a backend that
    /// diverged from it would change the model rather than the route.
    fn forward_moe_seq(
        &self,
        weights: &ModelWeights,
        layer: usize,
        h: &Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Array2<f32>, MoeBackendError>;

    /// Which route this is, for diagnostics. Never branched on.
    fn name(&self) -> &'static str;
}

/// The in-process route, made explicit as a backend.
///
/// Byte-identical to the block loop's default branch: the same
/// `cpu_moe_forward` call with the same arguments, position by position. It
/// exists for two reasons.
///
/// First, so a comparison can put both routes behind one interface and record
/// them the same way — otherwise the incumbent has no seam to observe and the
/// harness ends up reconstructing boundaries it cannot see.
///
/// Second, so the seam itself is falsifiable. Running with this backend must
/// equal running with no backend at all; if it does not, the trait changed the
/// model rather than merely relocating the call, and every comparison built on
/// it is measuring the wrong thing.
#[derive(Debug, Clone, Copy, Default)]
pub struct InProcessMoeBackend;

impl MoeExpertBackend for InProcessMoeBackend {
    fn forward_moe_seq(
        &self,
        weights: &ModelWeights,
        layer: usize,
        h: &Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Array2<f32>, MoeBackendError> {
        let seq_len = h.nrows();
        let hidden = h.ncols();
        let arch = &*weights.arch;
        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let Some(moe) = larql_compute::pipeline_layer::build_moe_weights(weights, arch, layer)
        else {
            return Ok(out);
        };
        // The same layer tag the default branch sets, so the within-expert
        // probe behaves identically whichever way the call arrives.
        larql_compute::cpu::ops::moe::set_current_layer(layer);
        for pos in 0..seq_len {
            let row: Vec<f32> = h.row(pos).to_vec();
            let moe_out =
                larql_compute::cpu::ops::moe::cpu_moe_forward(&row, &moe, norm_offset, eps);
            for (dst, src) in out.row_mut(pos).iter_mut().zip(moe_out.iter()) {
                *dst = *src;
            }
        }
        Ok(out)
    }

    fn name(&self) -> &'static str {
        "in-process"
    }
}
