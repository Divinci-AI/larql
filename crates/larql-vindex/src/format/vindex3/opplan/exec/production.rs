//! The production backend: the same plan, realised by `larql-compute`.
//!
//! Deliberately boring. Every method maps to the most direct existing
//! production kernel that preserves the resolved operation — no fusion,
//! no special-casing, no optimisation. The only claim being made at this
//! rung is that one semantic IR drives two numerical implementations;
//! speed comes later, under two fixed correctness anchors.
//!
//! **It binds the real kernels, not lookalikes.** `matmul_vec` is public
//! precisely so a VINDEX3 backend can call *that* function rather than
//! reimplement a similar loop — binding the real one is the difference
//! between proving kernel binding works and proving two similar loops
//! agree.
//!
//! **It fails closed.** Where `larql-compute` has no kernel for a judged
//! variant, this returns an error naming what is missing. Falling back to
//! the reference's arithmetic would make the two backends agree by
//! sharing code, which is exactly the agreement that proves nothing.

use larql_models::config::{
    Activation, GateActivation, GateCombine, GatePlacement, GateSource, NormType, PositionPolicy,
};
use ndarray::Array2;

use larql_compute::attention::softmax::softmax_in_place_f32;
use larql_compute::cpu::ops::geglu::{geglu_silu_alloc, silu};
use larql_compute::cpu::ops::moe::math::matmul_vec;
use larql_compute::residual::{
    layer_norm_eps, rms_norm_eps, rms_norm_heads_no_weight_eps, rms_norm_qk_eps,
};

use super::super::super::graph::policy::AttentionSpan;
use super::backend::{
    AttentionCall, FfnCall, GateCall, NormCall, PlanBackend, ProjectCall, QkNormCall,
};
use super::kernels::rope_rotate;
use crate::error::VindexError;

/// Name reported by [`PlanBackend::name`].
const NAME: &str = "production-larql-compute";

/// `larql-compute` realisation of every plan operation.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProductionBackend;

impl ProductionBackend {
    pub fn new() -> Self {
        Self
    }
}

/// Refuse an activation `larql-compute` has no kernel for.
///
/// Naming what is missing, rather than silently reusing the reference's
/// scalar loop: two backends that share arithmetic agree by construction,
/// and that agreement is exactly what this rung must not manufacture.
fn unsupported_activation(shape: &str, activation: Activation) -> VindexError {
    VindexError::Parse(format!(
        "no production {shape}-FFN kernel for activation {activation:?} — refusing rather \
         than borrowing the reference backend's arithmetic"
    ))
}

/// Wrap one vector as a `[1, n]` matrix for the row-wise norm kernels.
fn as_row(x: &[f32]) -> Array2<f32> {
    Array2::from_shape_vec((1, x.len()), x.to_vec()).expect("row shape matches length")
}

/// Take the single row back out.
fn from_row(m: Array2<f32>) -> Vec<f32> {
    m.into_raw_vec_and_offset().0
}

/// Apply Q/K normalisation to one projection in place.
///
/// Head geometry is passed to the kernel rather than sliced here, so the
/// production path exercises the production reduction over `head_dim`.
fn qk_norm_in_place(
    values: &mut [f32],
    weight: Option<(&[f32], f32)>,
    parameter_free: bool,
    num_heads: usize,
    head_dim: usize,
    scope: larql_models::config::QkNormScope,
    eps: f64,
) {
    if let Some((w, offset)) = weight {
        let normed = rms_norm_qk_eps(&as_row(values), w, num_heads, head_dim, offset, scope, eps);
        values.copy_from_slice(&from_row(normed));
    }
    if parameter_free {
        let normed = rms_norm_heads_no_weight_eps(&as_row(values), num_heads, head_dim, eps);
        values.copy_from_slice(&from_row(normed));
    }
}

impl PlanBackend for ProductionBackend {
    fn name(&self) -> &str {
        NAME
    }

    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32> {
        let row = &table[token as usize * hidden..(token as usize + 1) * hidden];
        match scale {
            Some(scale) => row.iter().map(|v| v * scale).collect(),
            None => row.to_vec(),
        }
    }

    fn norm(&self, call: NormCall<'_>) -> Vec<f32> {
        let weight = (!call.weight.is_empty()).then(|| call.weight.to_vec());
        let normed = match call.kind {
            NormType::RmsNorm => rms_norm_eps(
                &as_row(call.x),
                weight.as_ref(),
                call.weight_offset,
                call.eps,
            ),
            // The production layer-norm kernel takes a bias; the plan
            // carries none, so it is absent rather than zeroed.
            NormType::LayerNorm => layer_norm_eps(&as_row(call.x), weight.as_ref(), None, call.eps),
        };
        from_row(normed)
    }

    fn project(&self, call: ProjectCall<'_>) -> Vec<f32> {
        matmul_vec(call.x, call.weight, call.out_dim, call.in_dim)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        let head_dim = call.head_dim;
        let q_rows = call.num_q_heads * head_dim;
        let kv_rows = call.num_kv_heads * head_dim;
        let group = call.num_q_heads / call.num_kv_heads;
        let hidden = call.hidden;
        let qk_weight = call.qk_norm.as_ref().map(
            |QkNormCall {
                 weight_offset,
                 q_weight,
                 k_weight,
                 scope,
             }| (*scope, *weight_offset, *q_weight, *k_weight),
        );

        let mut queries = Vec::with_capacity(call.inputs.len());
        let mut keys = Vec::with_capacity(call.inputs.len());
        let mut values = Vec::with_capacity(call.inputs.len());
        for (position, pre) in call.inputs.iter().enumerate() {
            let mut q = matmul_vec(pre, call.w_q, q_rows, hidden);
            let mut k = matmul_vec(pre, call.w_k, kv_rows, hidden);
            let v = matmul_vec(pre, call.w_v, kv_rows, hidden);

            let (scope, offset, q_w, k_w) = match qk_weight {
                Some((scope, offset, q_w, k_w)) => (scope, offset, Some(q_w), Some(k_w)),
                None => (larql_models::config::QkNormScope::PerHead, 0.0, None, None),
            };
            qk_norm_in_place(
                &mut q,
                q_w.map(|w| (w, offset)),
                call.parameter_free_qk_norm.q,
                call.num_q_heads,
                head_dim,
                scope,
                call.qk_norm_eps,
            );
            qk_norm_in_place(
                &mut k,
                k_w.map(|w| (w, offset)),
                call.parameter_free_qk_norm.k,
                call.num_kv_heads,
                head_dim,
                scope,
                call.qk_norm_eps,
            );

            if let Some(query_scale) = call.query_scale {
                for value in &mut q {
                    *value *= query_scale as f32;
                }
            }
            if let PositionPolicy::Rope { theta } = call.position {
                for head in q.chunks_exact_mut(head_dim) {
                    rope_rotate(head, position, theta);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    rope_rotate(head, position, theta);
                }
            }
            queries.push(q);
            keys.push(k);
            values.push(v);
        }

        let mut out = Vec::with_capacity(call.inputs.len());
        for (position, query) in queries.iter().enumerate() {
            let start = match (call.span, call.window) {
                (AttentionSpan::Sliding, Some(window)) => (position + 1).saturating_sub(window),
                _ => 0,
            };
            let mut concat = vec![0.0f32; q_rows];
            for q_head in 0..call.num_q_heads {
                let kv_head = q_head / group;
                let q_slice = &query[q_head * head_dim..(q_head + 1) * head_dim];
                let mut scores: Vec<f32> = (start..=position)
                    .map(|key_position| {
                        let k_slice =
                            &keys[key_position][kv_head * head_dim..(kv_head + 1) * head_dim];
                        let dot: f32 = q_slice.iter().zip(k_slice).map(|(a, b)| a * b).sum();
                        let scaled = dot * call.score_scale as f32;
                        match call.logit_softcapping {
                            Some(cap) => cap * (scaled / cap).tanh(),
                            None => scaled,
                        }
                    })
                    .collect();
                softmax_in_place_f32(&mut scores);
                let head_out = &mut concat[q_head * head_dim..(q_head + 1) * head_dim];
                for (offset, key_position) in (start..=position).enumerate() {
                    let v_slice =
                        &values[key_position][kv_head * head_dim..(kv_head + 1) * head_dim];
                    let weight = scores[offset];
                    for (acc, v) in head_out.iter_mut().zip(v_slice) {
                        *acc += weight * v;
                    }
                }
            }

            if let Some(GateCall { spec, weight }) = &call.gate {
                // Exhaustive on the judged semantics, same as the
                // reference: a new variant must be implemented before it
                // can execute on this backend either.
                let GateSource::AttentionInput = spec.source;
                let GateActivation::Sigmoid = spec.activation;
                let GateCombine::ElementwiseMultiply = spec.combine;
                let GatePlacement::AfterAggregationBeforeOutputProjection = spec.placement;
                let gate_values = matmul_vec(&call.inputs[position], weight, q_rows, hidden);
                for (c, g) in concat.iter_mut().zip(&gate_values) {
                    *c *= 1.0 / (1.0 + (-g).exp());
                }
            }

            out.push(matmul_vec(&concat, call.w_o, hidden, q_rows));
        }
        Ok(out)
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        let up = matmul_vec(call.x, call.up, call.intermediate, call.hidden);
        let inner: Vec<f32> = match call.gate {
            Some(gate_weight) => {
                let gate = matmul_vec(call.x, gate_weight, call.intermediate, call.hidden);
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
        Ok(matmul_vec(
            &inner,
            call.down,
            call.hidden,
            call.intermediate,
        ))
    }

    fn output_head(
        &self,
        projection: &[f32],
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Vec<f32> {
        let mut logits = matmul_vec(x, projection, vocab, hidden);
        for logit in &mut logits {
            if let Some(multiplier) = multiplier {
                *logit *= multiplier as f32;
            }
            if let Some(cap) = softcapping {
                *logit = cap * (*logit / cap).tanh();
            }
        }
        logits
    }

    fn residual_add(&self, acc: &mut [f32], delta: &[f32]) {
        for (a, b) in acc.iter_mut().zip(delta) {
            *a += b;
        }
    }
}
