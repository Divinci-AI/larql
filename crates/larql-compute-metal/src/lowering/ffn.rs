//! Lowering a plan's gated FFN into one encoder (VINDEX3-G6b).
//!
//! The first fragment of a `LayerOpPlan` to be lowered, chosen because it
//! is self-contained — norm in, hidden-sized vector out, no KV cache and
//! no position policy — and because it is roughly three quarters of
//! Glimmer's weight bytes.
//!
//! Deliberately literal. Every operation the plan states gets its own
//! encoder action, in plan order:
//!
//! ```text
//! h ─ pre_ffn_norm ─┬─ gate proj ─┐
//!                   └─ up proj ───┴─ SiLU-GLU ─ down proj ─ residual ─ h'
//! ```
//!
//! No fusion yet. A fused activation+down kernel exists for Q4_K/Q6_K and
//! would be faster, but it would also make the correspondence between the
//! plan and the encoded work harder to check, and this rung's job is to
//! establish that correspondence. Fusion is a later, separately-judged
//! change.
//!
//! `stages::ffn::encode_gated` is not reused: it dispatches on
//! `larql_compute::QuantFormat`, which has no NVFP4 variant, so routing a
//! plan through it would mean either lying about the format or extending
//! the serving path's enum for a format it does not serve.

use metal::{Buffer, ComputeCommandEncoderRef};

use super::MatvecOperands;
use crate::MetalBackend;

/// The weight streams one gated FFN reads, already resident.
pub struct FfnWeights<'a> {
    pub gate_packed: &'a Buffer,
    pub gate_scales: &'a Buffer,
    pub gate_tensor_scale: f32,
    pub up_packed: &'a Buffer,
    pub up_scales: &'a Buffer,
    pub up_tensor_scale: f32,
    pub down_packed: &'a Buffer,
    pub down_scales: &'a Buffer,
    pub down_tensor_scale: f32,
    /// Pre-FFN norm weight (f32).
    pub norm_weight: &'a Buffer,
}

/// Device scratch the lowered FFN needs. Caller-owned so the whole layer
/// (later, the whole token) can allocate once and reuse across layers
/// rather than churning the pool per operation.
pub struct FfnScratch<'a> {
    /// `hidden` floats — normalised input.
    pub normed: &'a Buffer,
    /// `intermediate` floats each.
    pub gate: &'a Buffer,
    pub up: &'a Buffer,
    pub act: &'a Buffer,
    /// `hidden` floats — down-projection output, before the residual.
    pub down: &'a Buffer,
}

/// Geometry and judged semantics, straight off the plan.
pub struct FfnShape {
    pub hidden: usize,
    pub intermediate: usize,
    /// From the plan's `NormOp`.
    pub norm_eps: f32,
    /// Centred-norm convention (`1 + w`); a plan fact, never assumed.
    pub norm_weight_offset: f32,
}

impl MetalBackend {
    /// Encode `h' = h + down(silu(gate(norm(h))) * up(norm(h)))`.
    ///
    /// `h_in` and `h_out` may be the same buffer only if the caller is
    /// certain no dispatch reads `h_in` after `h_out` is written; the
    /// residual reads both, so they must differ.
    pub fn encode_nvfp4_gated_ffn(
        &self,
        enc: &ComputeCommandEncoderRef,
        h_in: &Buffer,
        h_out: &Buffer,
        w: &FfnWeights<'_>,
        s: &FfnScratch<'_>,
        shape: &FfnShape,
    ) {
        // 1. pre-FFN norm.
        crate::stages::input_norm::encode_f32(
            enc,
            &self.norms.rms_norm_pipeline,
            h_in,
            0,
            w.norm_weight,
            s.normed,
            0,
            shape.hidden,
            shape.norm_eps,
            shape.norm_weight_offset,
        );

        // 2. gate and up projections. Independent of each other, and the
        //    serial encoder still orders them after the norm that feeds
        //    them.
        self.encode_nvfp4_matvec(
            enc,
            &MatvecOperands {
                packed: w.gate_packed,
                scales: w.gate_scales,
                x: s.normed,
                out: s.gate,
                out_offset: 0,
                n: shape.intermediate,
                k: shape.hidden,
            },
            w.gate_tensor_scale,
        );
        self.encode_nvfp4_matvec(
            enc,
            &MatvecOperands {
                packed: w.up_packed,
                scales: w.up_scales,
                x: s.normed,
                out: s.up,
                out_offset: 0,
                n: shape.intermediate,
                k: shape.hidden,
            },
            w.up_tensor_scale,
        );

        // 3. SiLU-GLU.
        encode_elementwise(
            enc,
            &self.ffn.geglu_pipeline,
            &[s.gate, s.up, s.act],
            shape.intermediate,
        );

        // 4. down projection.
        self.encode_nvfp4_matvec(
            enc,
            &MatvecOperands {
                packed: w.down_packed,
                scales: w.down_scales,
                x: s.act,
                out: s.down,
                out_offset: 0,
                n: shape.hidden,
                k: shape.intermediate,
            },
            w.down_tensor_scale,
        );

        // 5. residual. `b_scale` is 1.0: a residual multiplier is a
        //    judged plan fact and Glimmer's FFN residual has none, so
        //    passing anything else here would invent semantics.
        self.encode_residual_add(enc, h_in, s.down, h_out, shape.hidden, 1.0);
    }
}

/// Three-buffer elementwise kernel with a trailing `uint` length —
/// the shape `geglu_silu` and its siblings share.
fn encode_elementwise(
    enc: &ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    bufs: &[&Buffer; 3],
    len: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    for (i, b) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(*b), 0);
    }
    let n = len as u32;
    enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
    super::dispatch_linear(enc, pipeline, len);
}
