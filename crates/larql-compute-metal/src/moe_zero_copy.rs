//! Zero-copy MoE expert dispatch — experts bound as offsets into
//! registered mmap regions.
//!
//! The staged path in `moe_dispatch.rs` memcpys every selected expert's
//! bytes into scratch each layer each token — GPT-OSS decode: top-4 ×
//! ~22 MB × 24 layers ≈ 2.1 GB of CPU memcpy per token, the dominant
//! decode cost once attention moved to the GPU. When every selected
//! expert's byte slices resolve inside a registered region
//! (`BufferCache::register_region` over each layer-weights mmap), this
//! path binds `(region_buffer, byte_offset)` per expert instead: no
//! staging, no duplication, the GPU reads the mmap pages directly
//! through unified memory.
//!
//! Dispatch shape follows `run_experts_prestaged_metal` (per-expert
//! dispatches — separate buffers preclude the staged path's single
//! all-K matvec) with the staged path's format awareness kept: per-format
//! matvec selection (Q4_K fused gate+up / Q6_K paired matvecs), the
//! layer's typed `MoeGateRule` activation, ClampedGlu gate/up biases, and
//! the down bias + routing weight + post-experts norm on readback.

use metal::*;
use std::ffi::c_void;

use super::buffers::read_buffer_f32;
use super::moe_dispatch::MoeScratch;
use super::MetalBackend;
use larql_compute::cpu::ops::moe::moe_post_expert_output;
use larql_compute::MoeLayerWeights;

/// One selected expert resolved to zero-copy bindings.
pub(super) struct ResolvedExpert {
    pub gate_up: (Buffer, u64),
    pub down: (Buffer, u64),
    pub expert_id: usize,
    pub weight: f32,
}

impl MetalBackend {
    /// Run `resolved` experts over `expert_input` and return the weighted,
    /// post-normed MoE output. Caller guarantees:
    /// - `resolved.len() <= scratch.top_k` (output offsets are bounded by
    ///   the scratch allocation),
    /// - every `gate_up` slice held ≥ `2 × inter × row_bytes` bytes and
    ///   every `down` slice ≥ `hidden × down_row_bytes` at resolution time
    ///   (offsets stay in-bounds of the registered region's buffer),
    /// - the biased-Gated refusal already ran (shared assert at the top of
    ///   `gpu_moe_dispatch_with_scratch`).
    pub(super) fn dispatch_experts_zero_copy(
        &self,
        expert_input: &[f32],
        moe: &MoeLayerWeights<'_>,
        eps: f32,
        scratch: &MoeScratch,
        resolved: &[ResolvedExpert],
    ) -> Vec<f32> {
        let hidden = scratch.hidden;
        let inter = scratch.inter;
        let inter_padded = scratch.inter_padded;
        let valid_count = resolved.len();
        debug_assert!(valid_count <= scratch.top_k);

        let timing_enabled =
            larql_compute::options::env_flag(larql_compute::options::ENV_METAL_MOE_TIMING);
        let t_start = std::time::Instant::now();

        // ── ClampedGlu gate/up biases: slot-aligned staging (small — the
        // weights stay zero-copy; the bias rows are `inter` f32 each).
        let stage_biases = !moe.experts_gate_up_bias.is_empty();
        if stage_biases {
            for (slot, r) in resolved.iter().enumerate() {
                let mlp = moe.expert_mlp(r.expert_id);
                // SAFETY: shared-storage scratch buffers allocated at
                // `top_k × inter` f32; `slot < valid_count <= top_k`.
                unsafe {
                    let gb = (scratch.gate_bias_buf.contents() as *mut f32).add(slot * inter);
                    let ub = (scratch.up_bias_buf.contents() as *mut f32).add(slot * inter);
                    for j in 0..inter {
                        *gb.add(j) = mlp.gate_bias(j);
                        *ub.add(j) = mlp.up_bias(j);
                    }
                }
            }
        }

        // ── Router-policy input into the pre-allocated x_buf (its
        // `weight_cols` tail is permanently zero — writer row padding
        // contributes nothing to any dot product).
        // SAFETY: shared-storage buffer sized `weight_cols ≥ hidden` f32.
        unsafe {
            let x_ptr = scratch.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(expert_input.as_ptr(), x_ptr, hidden);
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // ── Gate + up per expert, at the expert's region offset.
        let gate_half_bytes = (inter * scratch.row_bytes) as u64;
        let n_rows = inter as u32;
        let k_cols = scratch.weight_cols as u32;
        match scratch.format {
            larql_compute::QuantFormat::Q6_K => {
                let kh = &self.quant.q6k_matvec_pipeline;
                let tgs = (inter as u64).div_ceil(kh.rows_per_tg);
                for (e, r) in resolved.iter().enumerate() {
                    let (buf, off) = &r.gate_up;
                    for (half, out_buf) in [(0u64, &scratch.g_out), (1, &scratch.u_out)] {
                        enc.set_compute_pipeline_state(&kh.state);
                        enc.set_buffer(0, Some(buf), off + half * gate_half_bytes);
                        enc.set_buffer(1, Some(&scratch.x_buf), 0);
                        enc.set_buffer(2, Some(out_buf), (e * inter * 4) as u64);
                        enc.set_bytes(3, 4, &n_rows as *const u32 as *const c_void);
                        enc.set_bytes(4, 4, &k_cols as *const u32 as *const c_void);
                        enc.dispatch_thread_groups(
                            MTLSize::new(tgs, 1, 1),
                            MTLSize::new(kh.threads_per_tg, 1, 1),
                        );
                    }
                }
            }
            _ => {
                // Q4_K family: fused gate+up kernel, gate at the expert
                // offset, up one gate-half further into the same buffer.
                let kh = &self.ffn.q4k_ffn_gate_up_pipeline;
                let tgs = (inter as u64).div_ceil(kh.rows_per_tg);
                for (e, r) in resolved.iter().enumerate() {
                    let (buf, off) = &r.gate_up;
                    enc.set_compute_pipeline_state(&kh.state);
                    enc.set_buffer(0, Some(buf), *off);
                    enc.set_buffer(1, Some(buf), off + gate_half_bytes);
                    enc.set_buffer(2, Some(&scratch.x_buf), 0);
                    enc.set_buffer(3, Some(&scratch.g_out), (e * inter * 4) as u64);
                    enc.set_buffer(4, Some(&scratch.u_out), (e * inter * 4) as u64);
                    enc.set_bytes(5, 4, &n_rows as *const u32 as *const c_void);
                    enc.set_bytes(6, 4, &k_cols as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(tgs * 2, 1, 1),
                        MTLSize::new(kh.threads_per_tg, 1, 1),
                    );
                }
            }
        }

        // ── Typed gate-rule activation per expert (strided to
        // inter_padded so down's `K = inter_padded` reads see zeros).
        let inter_u32 = inter as u32;
        for e in 0..valid_count {
            let g_offset = (e * inter * 4) as u64;
            let a_offset = (e * inter_padded * 4) as u64;
            match moe.gate_rule {
                larql_compute::MoeGateRule::ClampedGlu { limit, alpha } => {
                    let has_bias: u32 = u32::from(stage_biases);
                    let b_offset = (e * inter * 4) as u64;
                    enc.set_compute_pipeline_state(&self.ffn.clamped_glu_bias_pipeline);
                    enc.set_buffer(0, Some(&scratch.g_out), g_offset);
                    enc.set_buffer(1, Some(&scratch.u_out), g_offset);
                    enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
                    enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
                    enc.set_buffer(4, Some(&scratch.gate_bias_buf), b_offset);
                    enc.set_buffer(5, Some(&scratch.up_bias_buf), b_offset);
                    enc.set_bytes(6, 4, &has_bias as *const u32 as *const c_void);
                    enc.set_bytes(7, 4, &limit as *const f32 as *const c_void);
                    enc.set_bytes(8, 4, &alpha as *const f32 as *const c_void);
                }
                larql_compute::MoeGateRule::Gated(activation) => {
                    let pipeline = if activation.gate_up_is_gelu_tanh() {
                        &self.ffn.geglu_gelu_tanh_pipeline
                    } else {
                        &self.ffn.geglu_pipeline
                    };
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(&scratch.g_out), g_offset);
                    enc.set_buffer(1, Some(&scratch.u_out), g_offset);
                    enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
                    enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
                }
            }
            enc.dispatch_threads(
                MTLSize::new(inter as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(inter as u64),
                    1,
                    1,
                ),
            );
        }

        // ── Down projection per expert at its region offset.
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        let down_kh = match scratch.format {
            larql_compute::QuantFormat::Q6_K => &self.quant.q6k_matvec_pipeline,
            _ => &self.quant.q4k_matvec_pipeline,
        };
        let down_tgs = (hidden as u64).div_ceil(down_kh.rows_per_tg);
        for (e, r) in resolved.iter().enumerate() {
            let (buf, off) = &r.down;
            let act_offset = (e * inter_padded * 4) as u64;
            let out_offset = (e * hidden * 4) as u64;
            enc.set_compute_pipeline_state(&down_kh.state);
            enc.set_buffer(0, Some(buf), *off);
            enc.set_buffer(1, Some(&scratch.act_buf), act_offset);
            enc.set_buffer(2, Some(&scratch.expert_outs), out_offset);
            enc.set_bytes(3, 4, &n_out as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &k_in as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(down_tgs, 1, 1),
                MTLSize::new(down_kh.threads_per_tg, 1, 1),
            );
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let t_gpu = t_start.elapsed();

        // ── Readback: down bias joins each expert's output BEFORE the
        // routing weight (reference order), then post-experts norm.
        let all_expert_outputs = read_buffer_f32(&scratch.expert_outs, valid_count * hidden);
        let mut moe_out = vec![0.0f32; hidden];
        for (e, r) in resolved.iter().enumerate() {
            let w = r.weight;
            let mlp = moe.expert_mlp(r.expert_id);
            let out_slice = &all_expert_outputs[e * hidden..(e + 1) * hidden];
            if mlp.down_bias.is_empty() {
                for (acc, &v) in moe_out.iter_mut().zip(out_slice) {
                    *acc += v * w;
                }
            } else {
                for ((acc, &v), &b) in moe_out.iter_mut().zip(out_slice).zip(mlp.down_bias) {
                    *acc += (v + b) * w;
                }
            }
        }
        if timing_enabled {
            let t_total = t_start.elapsed();
            eprintln!(
                "[run_experts_metal/zero-copy] K={valid_count} gpu={:.2}ms \
                 readback+sum={:.2}ms total={:.2}ms",
                t_gpu.as_secs_f32() * 1000.0,
                (t_total - t_gpu).as_secs_f32() * 1000.0,
                t_total.as_secs_f32() * 1000.0,
            );
        }

        moe_post_expert_output(&moe_out, moe, 0.0, eps)
    }
}
