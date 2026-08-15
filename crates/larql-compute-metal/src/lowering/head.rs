//! Lowering the final norm and output head (VINDEX3-G6c-2).
//!
//! ```text
//! h_final ─ final norm ─ lm_head matvec ─ multiplier ─ softcap ─ logits
//! ```
//!
//! Three judged facts live here that nothing else in the stack carries,
//! and each is a different value from its nearest neighbour:
//!
//! - **The final norm is a third norm configuration.** Muse-Glimmer's
//!   pre-block norms use eps 1e-5 with `weight_offset` 1.0, its post-block
//!   norms 1e-8 with 1.0, and its final norm 1e-5 with **0.0**. Carrying
//!   the branch norms' offset here is a silent centred-vs-uncentred bug.
//! - **The output multiplier** (0.196… for Glimmer). `None` = the op is
//!   absent, which is not the same claim as multiplying by one.
//! - **The final logit softcap** (20.0), applied *after* the multiplier.
//!   That order is semantic, unlike the query-scale/RoPE pair: tanh is
//!   nonlinear, so `softcap(m·x)` and `m·softcap(x)` are different
//!   functions.

use metal::{Buffer, ComputeCommandEncoderRef};

use super::MatvecOperands;
use crate::MetalBackend;

/// What the head reads.
pub struct HeadWeights<'a> {
    pub projection_packed: &'a Buffer,
    pub projection_scales: &'a Buffer,
    pub projection_tensor_scale: f32,
    /// Final norm weight (f32).
    pub norm_weight: &'a Buffer,
}

/// Device scratch: `hidden` then `vocab` floats.
pub struct HeadScratch<'a> {
    pub normed: &'a Buffer,
    pub raw_logits: &'a Buffer,
}

/// Geometry and judged semantics, straight off the plan.
pub struct HeadShape {
    pub hidden: usize,
    pub vocab: usize,
    pub norm_eps: f32,
    /// Glimmer's final norm is **uncentred** (0.0) where its branch norms
    /// are centred (1.0).
    pub norm_weight_offset: f32,
    /// `None` = the op is absent.
    pub multiplier: Option<f32>,
    /// `None` = the op is absent.
    pub softcap: Option<f32>,
}

impl MetalBackend {
    /// Encode final norm → head projection → multiplier → softcap.
    pub fn encode_nvfp4_head(
        &self,
        enc: &ComputeCommandEncoderRef,
        h_final: &Buffer,
        logits_out: &Buffer,
        w: &HeadWeights<'_>,
        s: &HeadScratch<'_>,
        shape: &HeadShape,
    ) {
        crate::stages::input_norm::encode_f32(
            enc,
            &self.norms.rms_norm_pipeline,
            h_final,
            0,
            w.norm_weight,
            s.normed,
            0,
            shape.hidden,
            shape.norm_eps,
            shape.norm_weight_offset,
        );
        self.encode_nvfp4_matvec(
            enc,
            &MatvecOperands {
                packed: w.projection_packed,
                scales: w.projection_scales,
                x: s.normed,
                out: s.raw_logits,
                out_offset: 0,
                n: shape.vocab,
                k: shape.hidden,
            },
            w.projection_tensor_scale,
        );
        // Absent ops encode as 0.0, which the kernel reads as "skip" —
        // distinct from a multiplier of one or a cap of zero.
        let pipeline = &self.norms.head_scale_softcap_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(s.raw_logits), 0);
        enc.set_buffer(1, Some(logits_out), 0);
        super::set_u32(enc, 2, shape.vocab as u32);
        super::set_f32(enc, 3, shape.multiplier.unwrap_or(0.0));
        super::set_f32(enc, 4, shape.softcap.unwrap_or(0.0));
        super::dispatch_linear(enc, pipeline, shape.vocab);
    }
}
