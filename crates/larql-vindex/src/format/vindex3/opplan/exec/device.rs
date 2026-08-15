//! The device backend: the plan's matrix work on an injected [`MatMul`]
//! device.
//!
//! **Layering.** This crate never links a GPU API. The seam it binds is
//! `larql-compute`'s [`MatMul`] trait — the same abstraction the serving
//! path dispatches through — and the *caller* injects the concrete
//! device (the CLI hands in `larql-compute-metal`'s backend for
//! `--backend metal`). vindex stays device-agnostic on every target;
//! whatever implements `MatMul` tomorrow (a second GPU API, a remote
//! device) lowers the same plan with no change here.
//!
//! **What is on the device, and what is not.** Every matrix–vector
//! product — Q/K/V/O projections, the attention gate projection, the
//! three FFN projections, and the vocabulary head — dispatches through
//! the injected device's gemv kernels. The elementwise glue between
//! them (norms, RoPE, softmax, activations, residual adds) runs on the
//! CPU, **deliberately as the production backend's own code**: sharing
//! the glue with the CPU production backend means any device-vs-
//! production divergence is attributable to device matmul arithmetic
//! alone, and the reference backend remains the fully independent leg
//! of the triangle.
//!
//! **Weight residency rides on the format.** Constructed with
//! [`WeightFormat::F16`], the backend declares f16 matrix operands; the
//! interpreter (or a decode session) loads each one once into a stable,
//! page-aligned allocation, and a device whose buffer cache keys on
//! `(pointer, length)` — the Metal backend's does — keeps every weight
//! resident across calls instead of re-uploading it per forward. With
//! `F32` the backend behaves as the first device rung did: correct, and
//! paying a full upload per fresh allocation.
//!
//! **It fails closed.** Matmuls use the *force* gemv variants, which
//! never fall back below a FLOP threshold — a threshold fallback would
//! quietly turn "device parity" into "CPU parity" for small shapes. If
//! the device has no kernel for the declared weight format, the call
//! errors naming the shape; nothing silently substitutes other
//! arithmetic.
//!
//! **Dispatches are serialised** behind a mutex. A GPU executes one
//! command buffer at a time anyway, and this sidesteps the known Metal
//! test-parallelism race class while the interpreter drives positions
//! from worker threads.

use std::sync::Mutex;

use larql_compute::backend::MatMul;
use larql_models::config::{Activation, GateActivation, GateCombine, GatePlacement, GateSource};

use super::backend::{
    AttentionCall, AttentionStepCall, AttentionStepOut, FfnCall, GateCall, NormCall, PlanBackend,
    ProjectCall, ProjectedQkv, WeightFormat, WeightSlice,
};
use super::production::{aggregate_heads, condition_qk_in_place, ProductionBackend};
use crate::error::VindexError;
use larql_compute::cpu::ops::geglu::{geglu_silu_alloc, silu};
use ndarray::ArrayView2;
use rayon::prelude::*;

use super::production::unsupported_activation;

/// Device realisation: injected-device matmuls, production-CPU glue.
pub struct DevicePlanBackend<M: MatMul + Send> {
    /// The injected device — for `--backend metal`, the same serving
    /// backend `larql run` computes with, not a lookalike wrapper.
    device: Mutex<M>,
    /// CPU glue, shared with the production backend on purpose (see
    /// module docs).
    glue: ProductionBackend,
    /// Reported through [`PlanBackend::name`] and hence the engine tag,
    /// so a dump names the concrete device and realisation that
    /// produced it.
    name: String,
    /// The matrix-operand representation this backend asks the
    /// interpreter for (see module docs on residency).
    format: WeightFormat,
}

impl<M: MatMul + Send> DevicePlanBackend<M> {
    /// `name` should carry device and realisation (e.g. `metal-r2-f16`)
    /// so a dump can never be mistaken for another lowering.
    pub fn new(device: M, name: impl Into<String>, format: WeightFormat) -> Self {
        Self {
            device: Mutex::new(device),
            glue: ProductionBackend::new(),
            name: name.into(),
            format,
        }
    }

    /// `out[out_dim] = W[out_dim, in_dim] · x` on the device, always,
    /// in whichever representation the weight arrived in.
    fn gemv(
        &self,
        weight: WeightSlice<'_>,
        out_dim: usize,
        in_dim: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, VindexError> {
        let device = self.device.lock().expect("device dispatch lock");
        match weight {
            WeightSlice::F32(w) => {
                let view = ArrayView2::from_shape((out_dim, in_dim), w).map_err(|e| {
                    VindexError::Parse(format!(
                        "device gemv: weight slice is not [{out_dim}, {in_dim}]: {e}"
                    ))
                })?;
                device.f32_gemv_force(view, x).ok_or_else(|| {
                    VindexError::Parse(format!(
                        "device f32_gemv [{out_dim} x {in_dim}] refused — no kernel or out of \
                         memory"
                    ))
                })
            }
            WeightSlice::F16(bytes) => {
                // The slice may be page-padded beyond the matrix (see
                // `weights::AlignedBytes`); geometry travels as n/k.
                if bytes.len() < out_dim * in_dim * 2 {
                    return Err(VindexError::Parse(format!(
                        "device f16 gemv: {} weight bytes cannot hold [{out_dim} x {in_dim}]",
                        bytes.len()
                    )));
                }
                device
                    .f16_gemv_force(bytes, x, out_dim, in_dim)
                    .ok_or_else(|| {
                        VindexError::Parse(format!(
                            "device f16_gemv [{out_dim} x {in_dim}] refused — no kernel or out \
                             of memory"
                        ))
                    })
            }
        }
    }

    /// One position's Q/K/V projections on the device, conditioned by
    /// the production glue — the arithmetic shared by the batch path
    /// and the decode step.
    fn project_position(
        &self,
        call: &AttentionCall<'_>,
        position: usize,
        pre: &[f32],
    ) -> Result<ProjectedQkv, VindexError> {
        let q_rows = call.num_q_heads * call.head_dim;
        let kv_rows = call.num_kv_heads * call.head_dim;
        let mut q = self.gemv(call.w_q, q_rows, call.hidden, pre)?;
        let mut k = self.gemv(call.w_k, kv_rows, call.hidden, pre)?;
        let v = self.gemv(call.w_v, kv_rows, call.hidden, pre)?;
        condition_qk_in_place(call, position, &mut q, &mut k);
        Ok((q, k, v))
    }

    /// Aggregation (production glue) plus this backend's own gate and
    /// output projections on the device.
    fn attend_position<'k>(
        &self,
        call: &AttentionCall<'_>,
        position: usize,
        query: &[f32],
        key_of: impl Fn(usize) -> &'k [f32],
        value_of: impl Fn(usize) -> &'k [f32],
        gate_input: &[f32],
    ) -> Result<Vec<f32>, VindexError> {
        let q_rows = call.num_q_heads * call.head_dim;
        let mut concat = aggregate_heads(call, position, query, key_of, value_of);

        if let Some(GateCall { spec, weight }) = &call.gate {
            // Exhaustive on the judged semantics, like both CPU
            // backends: a new variant must be implemented here before
            // it can execute on the device.
            let GateSource::AttentionInput = spec.source;
            let GateActivation::Sigmoid = spec.activation;
            let GateCombine::ElementwiseMultiply = spec.combine;
            let GatePlacement::AfterAggregationBeforeOutputProjection = spec.placement;
            let gate_values = self.gemv(*weight, q_rows, call.hidden, gate_input)?;
            for (c, g) in concat.iter_mut().zip(&gate_values) {
                *c *= 1.0 / (1.0 + (-g).exp());
            }
        }

        self.gemv(call.w_o, call.hidden, q_rows, &concat)
    }
}

impl<M: MatMul + Send> PlanBackend for DevicePlanBackend<M> {
    fn name(&self) -> &str {
        &self.name
    }

    fn weight_format(&self) -> WeightFormat {
        self.format
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        self.glue.embed(table, hidden, token, scale)
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        self.glue.norm(call)
    }

    fn project(&self, call: ProjectCall<'_>) -> Result<Vec<f32>, VindexError> {
        self.gemv(call.weight, call.out_dim, call.in_dim, call.x)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        // Same structure as the production backend's attention; the only
        // substitution is which arithmetic performs each projection.
        // Projections stay serial over positions here — the GPU queue is
        // one lane, so parallel callers would only contend on the lock.
        let mut queries = Vec::with_capacity(call.inputs.len());
        let mut keys = Vec::with_capacity(call.inputs.len());
        let mut values = Vec::with_capacity(call.inputs.len());
        for (position, pre) in call.inputs.iter().enumerate() {
            let (q, k, v) = self.project_position(&call, position, pre)?;
            queries.push(q);
            keys.push(k);
            values.push(v);
        }

        // Score/softmax/weighted-V on the CPU (parallel over query
        // positions, arithmetic per position untouched); the gate and
        // output projections return to the device, serially — one GPU
        // lane again.
        let aggregated: Vec<Vec<f32>> = queries
            .par_iter()
            .enumerate()
            .map(|(position, query)| {
                aggregate_heads(
                    &call,
                    position,
                    query,
                    |p| keys[p].as_slice(),
                    |p| values[p].as_slice(),
                )
            })
            .collect();

        let mut out = Vec::with_capacity(aggregated.len());
        for (position, mut concat) in aggregated.into_iter().enumerate() {
            if let Some(GateCall { spec, weight }) = &call.gate {
                // Exhaustive on the judged semantics (see attend_position).
                let GateSource::AttentionInput = spec.source;
                let GateActivation::Sigmoid = spec.activation;
                let GateCombine::ElementwiseMultiply = spec.combine;
                let GatePlacement::AfterAggregationBeforeOutputProjection = spec.placement;
                let q_rows = call.num_q_heads * call.head_dim;
                let gate_values =
                    self.gemv(*weight, q_rows, call.hidden, &call.inputs[position])?;
                for (c, g) in concat.iter_mut().zip(&gate_values) {
                    *c *= 1.0 / (1.0 + (-g).exp());
                }
            }
            out.push(self.gemv(
                call.w_o,
                call.hidden,
                call.num_q_heads * call.head_dim,
                &concat,
            )?);
        }
        Ok(out)
    }

    fn attention_step(&self, step: AttentionStepCall<'_>) -> Result<AttentionStepOut, VindexError> {
        let call = &step.op;
        let pre = &call.inputs[0];
        let (q, k, v) = self.project_position(call, step.position, pre)?;
        let output = self.attend_position(
            call,
            step.position,
            &q,
            |p| {
                if p == step.position {
                    k.as_slice()
                } else {
                    step.keys[p].as_slice()
                }
            },
            |p| {
                if p == step.position {
                    v.as_slice()
                } else {
                    step.values[p].as_slice()
                }
            },
            pre,
        )?;
        Ok(AttentionStepOut {
            key: k,
            value: v,
            output,
        })
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        let up = self.gemv(call.up, call.intermediate, call.hidden, call.x)?;
        let inner = match call.gate {
            Some(gate_weight) => {
                let gate = self.gemv(gate_weight, call.intermediate, call.hidden, call.x)?;
                match call.activation {
                    Activation::Silu => geglu_silu_alloc(&gate, &up),
                    other => return Err(unsupported_activation("gated", other)),
                }
            }
            None => match call.activation {
                Activation::Silu => up.iter().map(|u| silu(*u)).collect(),
                other => return Err(unsupported_activation("ungated", other)),
            },
        };
        self.gemv(call.down, call.hidden, call.intermediate, &inner)
    }

    fn output_head(
        &self,
        projection: WeightSlice<'_>,
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Result<Vec<f32>, VindexError> {
        let mut logits = self.gemv(projection, vocab, hidden, x)?;
        for logit in &mut logits {
            if let Some(multiplier) = multiplier {
                *logit *= multiplier as f32;
            }
            if let Some(cap) = softcapping {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        Ok(logits)
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        self.glue.residual_add(acc, delta);
    }
}
