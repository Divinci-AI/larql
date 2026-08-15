//! Lowering a plan's attention op into one encoder (VINDEX3-G6b-3).
//!
//! The delicate fragment. Unlike the FFN, attention is an **ordered**
//! program whose steps approximately commute, so a lowering can contain
//! every operation, produce plausible numbers, and still represent a
//! different model. The order below is the interpreter's
//! `condition_qk_in_place`, transcribed rather than reconstructed:
//!
//! ```text
//! h ─ pre-attn norm ─┬─ Q proj ─ param-free QK norm ─ query scale ─ RoPE ─┐
//!                    ├─ K proj ─ param-free QK norm ─────────────  RoPE ──┤ (into KV cache)
//!                    ├─ V proj ───────────────────────────────────────────┤ (into KV cache)
//!                    └─ gate proj ────────────────────────┐               │
//!                                                          │      attention
//!                                                          │          │
//!                                            sigmoid gate ─┴──────────┘
//!                                                          │
//!                                                       o_proj ─ residual ─ h'
//! ```
//!
//! **Query scale applies to Q only, after QK norm and before RoPE.** All
//! three touch Q, and swapping any pair changes the model while leaving
//! magnitudes plausible — the parity test carries an explicit ordering
//! control for exactly this.
//!
//! K and V project **directly into their KV-cache slots** rather than
//! into scratch that is later copied: the cache is `[T, num_kv,
//! head_dim]` position-major, so the current position's slot is a plain
//! byte offset, and the in-place QK norm and RoPE then operate on the
//! cache through the same offset. Removing the copy also removes the
//! chance of the cached K diverging from the K that was normed.

use metal::{Buffer, ComputeCommandEncoderRef};

use super::MatvecOperands;
use crate::MetalBackend;

/// Position encoding for this layer, from its own policy entry — never a
/// model-wide default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoweredPosition {
    /// Rotary at this base.
    Rope { theta: f64 },
    /// The layer attends position-agnostically (NoPE).
    None,
}

/// One NVFP4 projection: the three device streams plus its tensor scale.
pub struct Projection<'a> {
    pub packed: &'a Buffer,
    pub scales: &'a Buffer,
    pub tensor_scale: f32,
}

/// Everything attention reads.
pub struct AttnWeights<'a> {
    pub q: Projection<'a>,
    pub k: Projection<'a>,
    pub v: Projection<'a>,
    pub o: Projection<'a>,
    /// The judged attention output gate. `None` = no gate op — which is
    /// a different claim from a gate that happens to be near 1.
    pub gate: Option<Projection<'a>>,
    /// Pre-attention norm weight (f32).
    pub norm_weight: &'a Buffer,
}

/// Caller-owned device scratch and cache.
pub struct AttnScratch<'a> {
    /// `hidden` floats.
    pub normed: &'a Buffer,
    /// `num_q_heads * head_dim` floats.
    pub q: &'a Buffer,
    /// `[T, num_kv_heads, head_dim]` — K and V caches, written in place.
    pub k_cache: &'a Buffer,
    pub v_cache: &'a Buffer,
    /// `num_q_heads * head_dim` floats each.
    pub gate: &'a Buffer,
    pub concat: &'a Buffer,
    pub gated: &'a Buffer,
    /// `hidden` floats — o_proj output, before the residual.
    pub attn_out: &'a Buffer,
    /// `head_dim / 2` floats, host-computed to match the interpreter's
    /// `theta^(-2i/head_dim)`.
    pub inv_freq: &'a Buffer,
}

/// Geometry and judged semantics, straight off the plan.
pub struct AttnShape {
    pub hidden: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub norm_weight_offset: f32,
    pub qk_norm_eps: f32,
    pub parameter_free_q: bool,
    pub parameter_free_k: bool,
    /// `None` = the op is absent, not a multiply by one.
    pub query_scale: Option<f32>,
    /// The canonical score-time multiply, kept separate from
    /// `query_scale` because folding them is algebra-equivalent and not
    /// fp-equivalent.
    pub score_scale: f32,
    pub position: LoweredPosition,
    /// Sliding window; `None` = attends the whole prefix.
    pub window: Option<usize>,
    /// `None` = the softcap op is absent.
    pub softcap: Option<f32>,
    /// Absolute position of the token being decoded.
    pub position_index: usize,
    /// Cache length **including** this position.
    pub kv_len: usize,
}

impl AttnShape {
    fn q_rows(&self) -> usize {
        self.num_q_heads * self.head_dim
    }
    fn kv_rows(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
    /// Byte offset of this position's slot in a `[T, num_kv, head_dim]`
    /// cache.
    fn kv_slot_offset(&self) -> u64 {
        (self.position_index * self.kv_rows() * std::mem::size_of::<f32>()) as u64
    }
}

impl MetalBackend {
    /// Encode one attention op, hidden state in to hidden state out.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_nvfp4_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        h_in: &Buffer,
        h_out: &Buffer,
        w: &AttnWeights<'_>,
        s: &AttnScratch<'_>,
        shape: &AttnShape,
    ) {
        let (q_rows, kv_rows) = (shape.q_rows(), shape.kv_rows());
        let slot = shape.kv_slot_offset();

        // 1. pre-attention norm.
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

        // 2. projections. K and V land in their cache slots directly.
        for (p, out, off, n) in [
            (&w.q, s.q, 0u64, q_rows),
            (&w.k, s.k_cache, slot, kv_rows),
            (&w.v, s.v_cache, slot, kv_rows),
        ] {
            self.encode_nvfp4_matvec(
                enc,
                &MatvecOperands {
                    packed: p.packed,
                    scales: p.scales,
                    x: s.normed,
                    out,
                    out_offset: off,
                    n,
                    k: shape.hidden,
                },
                p.tensor_scale,
            );
        }
        if let Some(g) = &w.gate {
            // The gate reads the *attention input* — the same normalised
            // vector the projections read, per `GateSource::AttentionInput`.
            self.encode_nvfp4_matvec(
                enc,
                &MatvecOperands {
                    packed: g.packed,
                    scales: g.scales,
                    x: s.normed,
                    out: s.gate,
                    out_offset: 0,
                    n: q_rows,
                    k: shape.hidden,
                },
                g.tensor_scale,
            );
        }

        // 3. parameter-free QK norm — Q and K independently, per head.
        if shape.parameter_free_q {
            self.encode_parameter_free_qk_norm(
                enc,
                s.q,
                0,
                shape.num_q_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }
        if shape.parameter_free_k {
            self.encode_parameter_free_qk_norm(
                enc,
                s.k_cache,
                slot,
                shape.num_kv_heads,
                shape.head_dim,
                shape.qk_norm_eps,
            );
        }

        // 4. query scale — Q only, after the norm, before RoPE.
        if let Some(scale) = shape.query_scale {
            self.encode_scale_vector(enc, s.q, q_rows, scale);
        }

        // 5. position encoding, from this layer's policy. `None` means
        //    the op is absent and nothing is encoded — not rotation by
        //    a zero angle, which would also be a no-op here but would
        //    be the wrong reason.
        if let LoweredPosition::Rope { .. } = shape.position {
            self.encode_rope(
                enc,
                s.q,
                0,
                shape.num_q_heads,
                shape.head_dim,
                s.inv_freq,
                shape.position_index,
            );
            self.encode_rope(
                enc,
                s.k_cache,
                slot,
                shape.num_kv_heads,
                shape.head_dim,
                s.inv_freq,
                shape.position_index,
            );
        }

        // 6. attention over the cache.
        self.encode_kv_attention(enc, s, shape);

        // 7. the judged gate, then the output projection.
        let aggregated = match &w.gate {
            Some(_) => {
                self.encode_sigmoid_gate(enc, s.concat, s.gate, s.gated, q_rows);
                s.gated
            }
            None => s.concat,
        };
        self.encode_nvfp4_matvec(
            enc,
            &MatvecOperands {
                packed: w.o.packed,
                scales: w.o.scales,
                x: aggregated,
                out: s.attn_out,
                out_offset: 0,
                n: shape.hidden,
                k: q_rows,
            },
            w.o.tensor_scale,
        );

        // 8. residual.
        self.encode_residual_add(enc, h_in, s.attn_out, h_out, shape.hidden, 1.0);
    }

    fn encode_kv_attention(
        &self,
        enc: &ComputeCommandEncoderRef,
        s: &AttnScratch<'_>,
        shape: &AttnShape,
    ) {
        let pipeline = &self.attention.kv_attend_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(s.q), 0);
        enc.set_buffer(1, Some(s.k_cache), 0);
        enc.set_buffer(2, Some(s.v_cache), 0);
        enc.set_buffer(3, Some(s.concat), 0);
        super::set_u32(enc, 4, shape.kv_len as u32);
        super::set_u32(enc, 5, shape.head_dim as u32);
        super::set_u32(enc, 6, shape.num_q_heads as u32);
        super::set_u32(enc, 7, shape.num_kv_heads as u32);
        super::set_f32(enc, 8, shape.score_scale);
        super::set_u32(enc, 9, shape.window.unwrap_or(0) as u32);
        // No sink logits in this plan. The kernel reads slot 10 only when
        // `has_sinks` is non-zero, but Metal still requires the binding to
        // exist, so a one-float placeholder goes in — `inv_freq` is a
        // live buffer of the right kind and is not read by this kernel.
        enc.set_buffer(10, Some(s.inv_freq), 0);
        super::set_u32(enc, 11, 0);
        super::set_f32(enc, 12, shape.softcap.unwrap_or(0.0));
        let threads = pipeline.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(shape.num_q_heads as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
    }
}
