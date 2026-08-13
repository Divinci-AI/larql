//! Descriptor-driven MoE layer encode — rung E of the GPU-dataflow
//! routing ladder.
//!
//! The GPU twin of `moe_zero_copy::encode_experts_and_combine_zero_copy`:
//! same kernels, same dispatch geometry, same slot-aligned scratch — but
//! every ROUTE-DEPENDENT input is a GPU-resident buffer:
//!
//! - expert offsets: `moe_descriptor_gather` (was: CPU resolve + `set_bytes`)
//! - gate/up biases: `moe_bias_stage` (was: CPU memcpy loop)
//! - down biases:    `moe_down_bias_stage` (was: CPU memcpy loop)
//! - routing weights: rung B's `selected_weights` buffer bound with
//!   `set_buffer` (was: `set_bytes` of CPU-computed weights)
//! - bias presence: a LAYER fact read from the descriptor table's bank
//!   presence (was: `expert_mlp(selected_id)` — a fact accessed through
//!   an expert is not thereby an expert fact)
//!
//! E's contract: after the residual enters the GPU router, no CPU
//! operation may inspect, transform, stage, resolve, or combine anything
//! whose value depends on which experts won. Route-INDEPENDENT host work
//! (x staging, dims, layer uniforms) is permitted — removing it is F's
//! subject (scheduling), not E's (semantics). The `route_witness`
//! counters hold this path to that contract: it must not move them.
//!
//! Q6_K + ContiguousHalves only for now — rung G extends the descriptor
//! bindings to MXFP4; other formats keep the legacy path by explicit
//! caller choice (complete or refuse, never a silent partial arm).

use crate::moe_descriptor::MoeExpertDescriptorTable;
use crate::moe_dispatch::MoeScratch;
use crate::MetalBackend;
use larql_compute::MoeLayerWeights;
use metal::{Buffer, MTLSize};
use std::ffi::c_void;

impl MetalBackend {
    /// Encode the full expert block + combine with descriptor-driven
    /// bindings. `selected_ids` / `selected_weights` are rung B's
    /// GPU-resident route result; the CPU never reads them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_experts_and_combine_descriptor(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        expert_input: &[f32],
        moe: &MoeLayerWeights<'_>,
        scratch: &MoeScratch,
        table: &MoeExpertDescriptorTable,
        selected_ids: &Buffer,
        selected_weights: &Buffer,
        h_post_attn: &Buffer,
        new_h: &Buffer,
    ) {
        // Route-INDEPENDENT x staging (same bytes whichever experts win).
        unsafe {
            let x_ptr = scratch.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(expert_input.as_ptr(), x_ptr, scratch.hidden);
        }
        self.encode_experts_and_combine_descriptor_x_buf(
            enc,
            &scratch.x_buf.clone(),
            moe,
            scratch,
            table,
            selected_ids,
            selected_weights,
            h_post_attn,
            new_h,
        );
    }

    /// Core encode with the expert input already GPU-resident — the form
    /// a pre-encoded token chain needs, where layer i+1's x IS layer i's
    /// `new_h` buffer and no host staging may sit between them. When
    /// `x_buf` is not the scratch's padded staging buffer, the stored row
    /// width must equal `hidden` (no zero tail to rely on).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_experts_and_combine_descriptor_x_buf(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x_buf: &Buffer,
        moe: &MoeLayerWeights<'_>,
        scratch: &MoeScratch,
        table: &MoeExpertDescriptorTable,
        selected_ids: &Buffer,
        selected_weights: &Buffer,
        h_post_attn: &Buffer,
        new_h: &Buffer,
    ) {
        assert!(
            x_buf.gpu_address() == scratch.x_buf.gpu_address()
                || scratch.weight_cols == scratch.hidden,
            "chained x binding needs weight_cols == hidden: a padded row \
             width relies on the staging buffer's zero tail"
        );
        assert_eq!(
            scratch.format,
            larql_compute::QuantFormat::Q6_K,
            "descriptor arm serves Q6_K (rung G extends to MXFP4); other \
             formats stay on the legacy path by explicit caller choice"
        );
        assert_eq!(
            moe.fused_row_layout,
            larql_compute::MoeFusedRowLayout::ContiguousHalves,
            "gate1 = gate0 + gate_half_bytes describes ContiguousHalves only"
        );
        debug_assert!(matches!(
            moe.routing_policy.post_expert_norm,
            larql_compute::MoePostExpertNormPolicy::None
        ));
        let hidden = scratch.hidden;
        let inter = scratch.inter;
        let inter_padded = scratch.inter_padded;
        let n_slots = scratch.top_k;
        let gate_half_bytes = inter * scratch.row_bytes;
        assert_eq!(
            table.gate_up_expert_bytes,
            2 * gate_half_bytes,
            "descriptor table's expert size disagrees with the scratch dims"
        );

        // The single runtime indirection: route → stored-expert bindings.
        let bindings = self.encode_descriptor_gather(
            enc,
            table,
            selected_ids,
            n_slots,
            gate_half_bytes as u32,
        );

        // E3: bias presence is a layer fact, stated by the table.
        let stage_gate_up_bias = table.gate_up_bias_bank.is_some();
        if let Some(bank) = &table.gate_up_bias_bank {
            self.encode_bias_stage(
                enc,
                bank,
                &bindings.slot_descs,
                (&scratch.gate_bias_buf, &scratch.up_bias_buf),
                inter,
                n_slots,
            );
        }

        // Gate + up halves: the D-proven binding — same grouped kernel,
        // offsets from the gathered buffers instead of set_bytes.
        let kh = &self.quant.q6k_grouped_experts_pipeline;
        let row_tiles = (inter as u64).div_ceil(kh.rows_per_tg);
        let n_rows = inter as u32;
        let k_cols = scratch.weight_cols as u32;
        let xstride_shared: u32 = 0;
        for (offs, out_buf) in [
            (&bindings.gate0_offs, &scratch.g_out),
            (&bindings.gate1_offs, &scratch.u_out),
        ] {
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(&table.gate_up_base), 0);
            enc.set_buffer(1, Some(offs), 0);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_bytes(4, 4, &n_rows as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &k_cols as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &xstride_shared as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(row_tiles, n_slots as u64, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        }

        // Activation — slot-shaped, route-independent (identical to the
        // legacy path; only the bias-presence authority changed).
        let inter_u32 = inter as u32;
        for e in 0..n_slots {
            let g_offset = (e * inter * 4) as u64;
            let a_offset = (e * inter_padded * 4) as u64;
            match moe.gate_rule {
                larql_compute::MoeGateRule::ClampedGlu { limit, alpha } => {
                    let has_bias: u32 = u32::from(stage_gate_up_bias);
                    enc.set_compute_pipeline_state(&self.ffn.clamped_glu_bias_pipeline);
                    enc.set_buffer(0, Some(&scratch.g_out), g_offset);
                    enc.set_buffer(1, Some(&scratch.u_out), g_offset);
                    enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
                    enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
                    enc.set_buffer(4, Some(&scratch.gate_bias_buf), g_offset);
                    enc.set_buffer(5, Some(&scratch.up_bias_buf), g_offset);
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

        // Down projection: grouped, each slot reading its own activation.
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        let xstride_own: u32 = inter_padded as u32;
        let row_tiles_down = (hidden as u64).div_ceil(kh.rows_per_tg);
        enc.set_compute_pipeline_state(&kh.state);
        enc.set_buffer(0, Some(&table.down_base), 0);
        enc.set_buffer(1, Some(&bindings.down_offs), 0);
        enc.set_buffer(2, Some(&scratch.act_buf), 0);
        enc.set_buffer(3, Some(&scratch.expert_outs), 0);
        enc.set_bytes(4, 4, &n_out as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &k_in as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &xstride_own as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(row_tiles_down, n_slots as u64, 1),
            MTLSize::new(kh.threads_per_tg, 1, 1),
        );

        // Down biases: descriptor-driven staging into the same scratch the
        // combine kernel reads (E1 — the last route-dependent CPU memcpy).
        let has_down_bias = table.down_bias_bank.is_some();
        if let Some(bank) = &table.down_bias_bank {
            let hidden_u32 = hidden as u32;
            let n = n_slots as u32;
            enc.set_compute_pipeline_state(&self.moe_down_bias_stage_pipeline);
            enc.set_buffer(0, Some(bank), 0);
            enc.set_buffer(1, Some(&bindings.slot_descs), 0);
            enc.set_buffer(2, Some(&scratch.down_bias_staged), 0);
            enc.set_bytes(3, 4, &hidden_u32 as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &n as *const u32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(hidden as u64, n_slots as u64, 1),
                MTLSize::new(64.min(hidden as u64).max(1), 1, 1),
            );
        }

        // Combine — same kernel, routing weights from rung B's GPU buffer
        // (E2: the set_bytes → set_buffer flip, kernel signature unchanged).
        let hidden_u = hidden as u32;
        let k_u = n_slots as u32;
        let has_bias_u: u32 = u32::from(has_down_bias);
        enc.set_compute_pipeline_state(&self.ffn.moe_weighted_combine_pipeline);
        enc.set_buffer(0, Some(&scratch.expert_outs), 0);
        enc.set_buffer(1, Some(h_post_attn), 0);
        enc.set_buffer(2, Some(new_h), 0);
        enc.set_bytes(3, 4, &hidden_u as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &k_u as *const u32 as *const c_void);
        enc.set_buffer(5, Some(selected_weights), 0);
        enc.set_buffer(6, Some(&scratch.down_bias_staged), 0);
        enc.set_bytes(7, 4, &has_bias_u as *const u32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new(hidden as u64, 1, 1),
            MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                1,
                1,
            ),
        );
    }
}

impl MetalBackend {
    /// Test-facing CONTROL arm: today's production CPU-routed layer,
    /// end to end — CPU route → `resolve_selected_experts` → legacy
    /// encode (offset/weight `set_bytes`, CPU bias staging) → readback.
    /// Moves the `route_witness` counters; that movement is the
    /// witness's own positive control.
    pub fn moe_layer_forward_control(
        &self,
        router_in: &[f32],
        moe: &MoeLayerWeights<'_>,
        h_post_attn: &[f32],
    ) -> Option<Vec<f32>> {
        let hidden = h_post_attn.len();
        let (ids, weights) =
            larql_compute::cpu::ops::moe::moe_route_from_router_input(router_in, moe);
        let scratch = MoeScratch::new_public_with_format(
            self,
            moe.top_k,
            hidden,
            moe.intermediate_size,
            moe.expert_data_format,
            hidden,
        );
        let resolved = self.resolve_selected_experts(&scratch, moe, &ids, &weights, |e| {
            Some((moe.experts_gate_up[e], moe.experts_down[e]))
        })?;
        let h_buf = self.bufs.transient_from_f32(h_post_attn);
        let new_h = self.bufs.output((hidden * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        self.encode_experts_and_combine_zero_copy(
            enc, router_in, moe, &scratch, &resolved, &h_buf, &new_h,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        crate::buffers::try_read_buffer_f32(&new_h, hidden)
    }

    /// Test-facing CANDIDATE arm: GPU router → GPU select → descriptor
    /// arm, one command buffer, no host-visible route anywhere in the
    /// signature (type-level closure). Must NOT move the
    /// `route_witness` counters.
    ///
    /// `poison_staging_scratch` scribbles garbage into every scratch
    /// buffer the LEGACY path staged via CPU (`gate_bias_buf`,
    /// `up_bias_buf`, `down_bias_staged`) before encoding — the poison
    /// proof: the GPU kernels must fully overwrite them, so output must
    /// be identical whether or not the poison ran.
    pub fn moe_layer_forward_descriptor(
        &self,
        router_in: &[f32],
        moe: &MoeLayerWeights<'_>,
        table: &MoeExpertDescriptorTable,
        h_post_attn: &[f32],
        poison_staging_scratch: bool,
    ) -> Option<Vec<f32>> {
        use larql_compute::{MoeExpertScalePolicy, MoeTopKWeightPolicy};
        let hidden = h_post_attn.len();
        let num_experts = moe.num_experts;
        if router_in.len() != hidden
            || moe.router_proj.len() != num_experts * hidden
            || num_experts > crate::shaders::moe_router_select::MAX_EXPERTS
            || moe.top_k == 0
            || moe.top_k > crate::shaders::moe_router_select::MAX_TOP_K
        {
            return None;
        }
        let renormalize =
            moe.routing_policy.selected_weight == MoeTopKWeightPolicy::RenormalizedSoftmax;
        let pe_scale = (moe.routing_policy.expert_scale == MoeExpertScalePolicy::PerExpert
            && !moe.router_per_expert_scale.is_empty())
        .then_some(moe.router_per_expert_scale);
        if let Some(s) = pe_scale {
            if s.len() != num_experts {
                return None;
            }
        }

        let scratch = MoeScratch::new_public_with_format(
            self,
            moe.top_k,
            hidden,
            moe.intermediate_size,
            moe.expert_data_format,
            hidden,
        );
        if poison_staging_scratch {
            // Every value the LEGACY path would have host-staged becomes
            // unmistakable garbage; any read that escapes GPU staging
            // explodes the parity gate instead of coincidentally passing.
            for buf in [
                &scratch.gate_bias_buf,
                &scratch.up_bias_buf,
                &scratch.down_bias_staged,
            ] {
                let len = buf.length() as usize / 4;
                unsafe {
                    let p = buf.contents() as *mut f32;
                    for i in 0..len {
                        *p.add(i) = 1.0e30;
                    }
                }
            }
        }

        let w_buf = self.bufs.get_f32(moe.router_proj);
        let x_router = self.bufs.transient_from_f32(router_in);
        let bias_buf =
            (!moe.router_bias.is_empty()).then(|| self.bufs.transient_from_f32(moe.router_bias));
        let scale_buf = pe_scale.map(|s| self.bufs.transient_from_f32(s));
        let h_buf = self.bufs.transient_from_f32(h_post_attn);
        let new_h = self.bufs.output((hidden * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let logits = self.encode_moe_router_logits(
            enc,
            &w_buf,
            &x_router,
            bias_buf.as_ref(),
            num_experts,
            hidden,
        );
        let (ids_buf, weights_buf) = self.encode_moe_router_select(
            enc,
            &logits,
            scale_buf.as_ref(),
            num_experts,
            moe.top_k,
            renormalize,
        );
        self.encode_experts_and_combine_descriptor(
            enc,
            router_in,
            moe,
            &scratch,
            table,
            &ids_buf,
            &weights_buf,
            &h_buf,
            &new_h,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        crate::buffers::try_read_buffer_f32(&new_h, hidden)
    }
}

/// One synthetic token's scheduling measurements from
/// [`MetalBackend::moe_token_forward_descriptor`]. `out` carries the
/// final layer's output so arms can be compared numerically — a
/// submission policy must not change the numbers.
pub struct MoeTokenScheduleStats {
    pub cmd_bufs: usize,
    pub wall_ms: f64,
    /// Σ (GPUEndTime − GPUStartTime) over the token's command buffers.
    pub gpu_busy_ms: f64,
    /// Σ positive gaps between consecutive command buffers' GPU windows
    /// — the queue-starvation bubble. Zero by construction at one CB.
    pub bubble_ms: f64,
    pub out: Vec<f32>,
}

impl MetalBackend {
    /// Rung F's instrument: run `layers` chained descriptor-driven MoE
    /// layers (layer i+1's router input IS layer i's output buffer — no
    /// readback, no host staging between layers) under one of two
    /// submission policies:
    ///
    /// - `pre_encode = false` (JIT): one command buffer per layer,
    ///   commit + wait each — production decode's cadence today.
    /// - `pre_encode = true`: every layer encoded into ONE command
    ///   buffer, committed once — the shape E's semantic closure makes
    ///   legal.
    ///
    /// Identical kernels, buffers and encode order in both arms; only
    /// WHEN work is submitted differs.
    pub fn moe_token_forward_descriptor(
        &self,
        router_in: &[f32],
        moe: &MoeLayerWeights<'_>,
        table: &MoeExpertDescriptorTable,
        layers: usize,
        pre_encode: bool,
    ) -> Option<MoeTokenScheduleStats> {
        use larql_compute::{MoeExpertScalePolicy, MoeTopKWeightPolicy};
        use objc::{msg_send, sel, sel_impl};

        let hidden = router_in.len();
        let num_experts = moe.num_experts;
        if layers == 0
            || moe.router_proj.len() != num_experts * hidden
            || num_experts > crate::shaders::moe_router_select::MAX_EXPERTS
            || moe.top_k == 0
            || moe.top_k > crate::shaders::moe_router_select::MAX_TOP_K
        {
            return None;
        }
        let renormalize =
            moe.routing_policy.selected_weight == MoeTopKWeightPolicy::RenormalizedSoftmax;
        let pe_scale = (moe.routing_policy.expert_scale == MoeExpertScalePolicy::PerExpert
            && !moe.router_per_expert_scale.is_empty())
        .then_some(moe.router_per_expert_scale);

        let scratch = MoeScratch::new_public_with_format(
            self,
            moe.top_k,
            hidden,
            moe.intermediate_size,
            moe.expert_data_format,
            hidden,
        );
        let w_buf = self.bufs.get_f32(moe.router_proj);
        let bias_buf =
            (!moe.router_bias.is_empty()).then(|| self.bufs.transient_from_f32(moe.router_bias));
        let scale_buf = pe_scale.map(|s| self.bufs.transient_from_f32(s));
        let h0 = self.bufs.transient_from_f32(router_in);
        let new_hs: Vec<Buffer> = (0..layers)
            .map(|_| self.bufs.output((hidden * 4) as u64))
            .collect();

        let encode_layer =
            |enc: &metal::ComputeCommandEncoderRef, prev_h: &Buffer, out: &Buffer| {
                let logits = self.encode_moe_router_logits(
                    enc,
                    &w_buf,
                    prev_h,
                    bias_buf.as_ref(),
                    num_experts,
                    hidden,
                );
                let (ids_buf, weights_buf) = self.encode_moe_router_select(
                    enc,
                    &logits,
                    scale_buf.as_ref(),
                    num_experts,
                    moe.top_k,
                    renormalize,
                );
                self.encode_experts_and_combine_descriptor_x_buf(
                    enc,
                    prev_h,
                    moe,
                    &scratch,
                    table,
                    &ids_buf,
                    &weights_buf,
                    prev_h,
                    out,
                );
            };

        let t0 = std::time::Instant::now();
        let mut windows: Vec<(f64, f64)> = Vec::with_capacity(layers);
        let cmd_bufs;
        if pre_encode {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let mut prev = h0.clone();
            for out in &new_hs {
                encode_layer(enc, &prev, out);
                prev = out.clone();
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            windows.push(unsafe {
                let start: f64 = msg_send![cmd, GPUStartTime];
                let end: f64 = msg_send![cmd, GPUEndTime];
                (start, end)
            });
            cmd_bufs = 1;
        } else {
            let mut prev = h0.clone();
            for out in &new_hs {
                let cmd = self.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                encode_layer(enc, &prev, out);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                windows.push(unsafe {
                    let start: f64 = msg_send![cmd, GPUStartTime];
                    let end: f64 = msg_send![cmd, GPUEndTime];
                    (start, end)
                });
                prev = out.clone();
            }
            cmd_bufs = layers;
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

        let gpu_busy_ms: f64 = windows.iter().map(|(s, e)| (e - s) * 1e3).sum();
        let bubble_ms: f64 = windows
            .windows(2)
            .map(|w| ((w[1].0 - w[0].1) * 1e3).max(0.0))
            .sum();
        let out = crate::buffers::try_read_buffer_f32(&new_hs[layers - 1], hidden)?;
        Some(MoeTokenScheduleStats {
            cmd_bufs,
            wall_ms,
            gpu_busy_ms,
            bubble_ms,
            out,
        })
    }
}

/// `LARQL_GPU_ROUTE=1` switches production MoE decode to the
/// GPU-dataflow route (serve-integration rung S1). Read once —
/// a decode-path A/B switch, not a runtime toggle.
pub(crate) fn gpu_route_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        matches!(
            std::env::var("LARQL_GPU_ROUTE").ok().as_deref(),
            Some("1") | Some("true")
        )
    })
}

/// The router-input transform, resolved EXPLICITLY from the routing
/// policy — the GPU route API must not hard-wire any one model's
/// `router_input = h_post_attn` assumption (that is how rung A would
/// get reopened by the next architecture).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RouterInputTransform {
    /// Route and run experts on the raw residual.
    Identity,
    /// gpt-oss shape: one RMS norm (`pre_experts_norm`) feeds router
    /// and experts alike.
    PreExpertsRmsNorm,
}

/// Resolve the transform, or `None` for any policy combination the GPU
/// route does not implement — the caller stays on the CPU path by
/// explicit fallback, never a silently wrong transform.
pub(crate) fn router_input_transform(moe: &MoeLayerWeights<'_>) -> Option<RouterInputTransform> {
    use larql_compute::{MoeInputSource, MoeRouterNormPolicy};
    let p = &moe.routing_policy;
    // Applied by `moe_router_input` after the norm; not yet GPU-side.
    if !moe.router_scale.is_empty() || moe.router_input_scalar != 1.0 {
        return None;
    }
    // Router and experts must share one input stream: the descriptor
    // arm binds a single x for both.
    if p.router_input != p.expert_input || p.router_norm != MoeRouterNormPolicy::None {
        return None;
    }
    match p.router_input {
        MoeInputSource::Residual => Some(RouterInputTransform::Identity),
        MoeInputSource::PreExpertsNorm if !moe.pre_experts_norm.is_empty() => {
            Some(RouterInputTransform::PreExpertsRmsNorm)
        }
        MoeInputSource::PreExpertsNorm => None,
    }
}

impl MetalBackend {
    /// Everything the GPU route needs to hold, checked BEFORE any
    /// command-buffer state is touched — a `false` here means the CPU
    /// arm proceeds with nothing to roll back.
    pub(crate) fn gpu_route_supported(
        &self,
        moe: &MoeLayerWeights<'_>,
        scratch: &MoeScratch,
    ) -> bool {
        router_input_transform(moe).is_some()
            && scratch.format == larql_compute::QuantFormat::Q6_K
            && moe.fused_row_layout == larql_compute::MoeFusedRowLayout::ContiguousHalves
            && matches!(
                moe.routing_policy.post_expert_norm,
                larql_compute::MoePostExpertNormPolicy::None
            )
            && moe.num_experts <= crate::shaders::moe_router_select::MAX_EXPERTS
            && moe.top_k >= 1
            && moe.top_k <= crate::shaders::moe_router_select::MAX_TOP_K
            && moe.router_proj.len() == moe.num_experts * scratch.hidden
            // Identity binds h_post_attn directly, so a padded row width
            // (weight_cols > hidden) needs the transform to route through
            // scratch.x_buf's permanently-zero tail instead.
            && (scratch.weight_cols == scratch.hidden
                || router_input_transform(moe)
                    == Some(RouterInputTransform::PreExpertsRmsNorm))
    }

    /// Fetch (or build once) the layer's descriptor table. Model swap on
    /// a reused backend is detected by the bank's pointer identity.
    pub(crate) fn descriptor_table_for_layer(
        &self,
        layer_idx: usize,
        moe: &MoeLayerWeights<'_>,
        inter: usize,
        hidden: usize,
    ) -> Option<std::sync::Arc<MoeExpertDescriptorTable>> {
        let bank_ptr = moe.experts_gate_up.first()?.as_ptr() as usize;
        let key = (layer_idx, bank_ptr);
        let mut map = self.moe_descriptor_tables.lock().unwrap();
        if let Some(t) = map.get(&key) {
            return Some(t.clone());
        }
        let table = std::sync::Arc::new(self.build_expert_descriptor_table(moe, inter, hidden)?);
        map.insert(key, table.clone());
        Some(table)
    }

    /// S1 production encode: the full GPU-dataflow MoE layer, consuming
    /// the GPU-resident `h_post_attn` — the host slice's routing role
    /// ends here. Preconditions were checked by
    /// [`Self::gpu_route_supported`]; this function only encodes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_moe_layer_gpu_route(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        moe: &MoeLayerWeights<'_>,
        scratch: &MoeScratch,
        table: &MoeExpertDescriptorTable,
        h_post_attn: &Buffer,
        new_h: &Buffer,
        eps: f32,
    ) {
        use larql_compute::{MoeExpertScalePolicy, MoeTopKWeightPolicy};
        let hidden = scratch.hidden;
        let num_experts = moe.num_experts;

        // Router-input transform, stated explicitly.
        let x_route =
            match router_input_transform(moe).expect("gpu_route_supported checked the policy") {
                RouterInputTransform::Identity => h_post_attn.clone(),
                RouterInputTransform::PreExpertsRmsNorm => {
                    // Norm INTO the scratch staging buffer: it is weight_cols
                    // wide with a permanently-zero tail (the same invariant
                    // the CPU staging path relies on), so padded row widths
                    // read zeros beyond `hidden` exactly as they do today.
                    // The kernel writes [0..hidden]; the tail is never touched.
                    let normed = scratch.x_buf.clone();
                    let weight_buf = self.bufs.get_f32(moe.pre_experts_norm);
                    let hidden_u = hidden as u32;
                    let norm_offset: f32 = 0.0;
                    enc.set_compute_pipeline_state(&self.norms.rms_norm_pipeline);
                    enc.set_buffer(0, Some(h_post_attn), 0);
                    enc.set_buffer(1, Some(&weight_buf), 0);
                    enc.set_buffer(2, Some(&normed), 0);
                    enc.set_bytes(3, 4, &hidden_u as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
                    enc.set_bytes(5, 4, &norm_offset as *const f32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, 1, 1),
                        MTLSize::new(
                            crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                            1,
                            1,
                        ),
                    );
                    normed
                }
            };

        let renormalize =
            moe.routing_policy.selected_weight == MoeTopKWeightPolicy::RenormalizedSoftmax;
        let pe_scale = (moe.routing_policy.expert_scale == MoeExpertScalePolicy::PerExpert
            && !moe.router_per_expert_scale.is_empty())
        .then(|| self.bufs.get_f32(moe.router_per_expert_scale));
        let w_buf = self.bufs.get_f32(moe.router_proj);
        let bias_buf = (!moe.router_bias.is_empty()).then(|| self.bufs.get_f32(moe.router_bias));

        let logits = self.encode_moe_router_logits(
            enc,
            &w_buf,
            &x_route,
            bias_buf.as_ref(),
            num_experts,
            hidden,
        );
        let (ids_buf, weights_buf) = self.encode_moe_router_select(
            enc,
            &logits,
            pe_scale.as_ref(),
            num_experts,
            moe.top_k,
            renormalize,
        );
        self.encode_experts_and_combine_descriptor_x_buf(
            enc,
            &x_route,
            moe,
            scratch,
            table,
            &ids_buf,
            &weights_buf,
            h_post_attn,
            new_h,
        );
    }
}
