//! The reference backend: naive f32, the semantic anchor.
//!
//! Shares **nothing** with `larql-compute`'s kernels. That is the whole
//! point of it — a reference that called the production kernels would
//! agree with them by construction, and the agreement would prove
//! nothing. Plain loops, row-major `[out, in]` weights, no BLAS, no
//! SIMD, no fusion.
//!
//! When the production backend disagrees with this one, this one is
//! right about *meaning* and may well be wrong about speed. Divergence
//! is a bug in the production backend or a hole in the seam, never a
//! licence to change what the plan means.

use larql_models::config::{GateActivation, GateCombine, GatePlacement, GateSource, QkNormScope};

use super::super::super::graph::policy::AttentionSpan;
use super::backend::{
    AttentionCall, FfnCall, GateCall, NormCall, PlanBackend, ProjectCall, QkNormCall,
};
use super::kernels::{activate, matvec, norm, rope_rotate, sigmoid, softcap, softmax};
use crate::error::VindexError;
use larql_models::config::NormType;
use larql_models::config::PositionPolicy;

/// Name reported by [`PlanBackend::name`].
const NAME: &str = "reference-f32";

/// Naive f32 realisation of every plan operation.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceBackend;

impl ReferenceBackend {
    pub fn new() -> Self {
        Self
    }

    /// Q/K normalisation: weighted per-head when the plan binds weights,
    /// parameter-free when the surface judged it. Both may apply.
    fn apply_qk_norm(
        call: &AttentionCall<'_>,
        q: &mut [f32],
        k: &mut [f32],
    ) -> Result<(), VindexError> {
        let head_dim = call.head_dim;
        let eps = call.qk_norm_eps;
        if let Some(QkNormCall {
            scope,
            weight_offset,
            q_weight,
            k_weight,
        }) = &call.qk_norm
        {
            match scope {
                QkNormScope::PerHead => {
                    for head in q.chunks_exact_mut(head_dim) {
                        let normed = norm(NormType::RmsNorm, head, q_weight, *weight_offset, eps);
                        head.copy_from_slice(&normed);
                    }
                    for head in k.chunks_exact_mut(head_dim) {
                        let normed = norm(NormType::RmsNorm, head, k_weight, *weight_offset, eps);
                        head.copy_from_slice(&normed);
                    }
                }
                QkNormScope::FullProjection => {
                    return Err(VindexError::Parse(
                        "full-projection QK norm has no judged reference execution yet".to_string(),
                    ));
                }
            }
        }
        if call.parameter_free_qk_norm.q {
            for head in q.chunks_exact_mut(head_dim) {
                let normed = norm(NormType::RmsNorm, head, &[], 0.0, eps);
                head.copy_from_slice(&normed);
            }
        }
        if call.parameter_free_qk_norm.k {
            for head in k.chunks_exact_mut(head_dim) {
                let normed = norm(NormType::RmsNorm, head, &[], 0.0, eps);
                head.copy_from_slice(&normed);
            }
        }
        Ok(())
    }
}

impl PlanBackend for ReferenceBackend {
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
        norm(call.kind, call.x, call.weight, call.weight_offset, call.eps)
    }

    fn project(&self, call: ProjectCall<'_>) -> Vec<f32> {
        matvec(call.weight, call.out_dim, call.in_dim, call.x)
    }

    fn attention(&self, call: AttentionCall<'_>) -> Result<Vec<Vec<f32>>, VindexError> {
        let head_dim = call.head_dim;
        let q_rows = call.num_q_heads * head_dim;
        let kv_rows = call.num_kv_heads * head_dim;
        let group = call.num_q_heads / call.num_kv_heads;
        let hidden = call.hidden;

        // Projections per position, with QK normalisation, query scale
        // and position encoding applied in the judged order.
        let mut queries = Vec::with_capacity(call.inputs.len());
        let mut keys = Vec::with_capacity(call.inputs.len());
        let mut values = Vec::with_capacity(call.inputs.len());
        for (position, pre) in call.inputs.iter().enumerate() {
            let mut q = matvec(call.w_q, q_rows, hidden, pre);
            let mut k = matvec(call.w_k, kv_rows, hidden, pre);
            let v = matvec(call.w_v, kv_rows, hidden, pre);

            Self::apply_qk_norm(&call, &mut q, &mut k)?;
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
            // Span: which key positions this query may attend to.
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
                        let mut dot = 0.0f32;
                        for (a, b) in q_slice.iter().zip(k_slice) {
                            dot += a * b;
                        }
                        let mut score = dot * call.score_scale as f32;
                        if let Some(cap) = call.logit_softcapping {
                            score = softcap(score, cap);
                        }
                        score
                    })
                    .collect();
                softmax(&mut scores);
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
                // Exhaustive on the judged semantics: a new variant must
                // be implemented here before it can execute.
                let GateSource::AttentionInput = spec.source;
                let GateActivation::Sigmoid = spec.activation;
                let GateCombine::ElementwiseMultiply = spec.combine;
                let GatePlacement::AfterAggregationBeforeOutputProjection = spec.placement;
                let gate_values = matvec(weight, q_rows, hidden, &call.inputs[position]);
                for (c, g) in concat.iter_mut().zip(&gate_values) {
                    *c *= sigmoid(*g);
                }
            }

            out.push(matvec(call.w_o, hidden, q_rows, &concat));
        }
        Ok(out)
    }

    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError> {
        let up = matvec(call.up, call.intermediate, call.hidden, call.x);
        let inner: Vec<f32> = match call.gate {
            Some(gate_weight) => {
                let gate = matvec(gate_weight, call.intermediate, call.hidden, call.x);
                gate.iter()
                    .zip(&up)
                    .map(|(g, u)| activate(call.activation, *g) * u)
                    .collect()
            }
            None => up.iter().map(|u| activate(call.activation, *u)).collect(),
        };
        Ok(matvec(call.down, call.hidden, call.intermediate, &inner))
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
        let mut logits = matvec(projection, vocab, hidden, x);
        for logit in &mut logits {
            if let Some(multiplier) = multiplier {
                *logit *= multiplier as f32;
            }
            if let Some(cap) = softcapping {
                *logit = softcap(*logit, cap);
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
