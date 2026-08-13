//! The reference plan executor (V3-G5b-2 Stage A).
//!
//! Executes a [`ComponentOpPlan`] — and **nothing else**. Every argument
//! comes from the plan (which came from the container); every operand
//! loads through the closure-verified `object → representation → segment`
//! path; every judged enum is matched exhaustively so an unjudged variant
//! is a compile error, not a guess. There is no family name, no layer
//! arithmetic, no HF tensor name, and no default anywhere in this module.
//!
//! Deliberately naive f32 (see [`kernels`]): the operation plan is not
//! documentation of execution — it *is* execution — and this module is
//! the smallest thing that makes that sentence true. Performance comes
//! later by lowering the same plan onto real kernels, not by changing
//! what it means.
//!
//! The trace mirrors the production forward's hook points
//! (`post_attention` = after the attention residual add, `post_layer` =
//! after the FFN residual add) so Stage A parity can compare layer by
//! layer against the checkpoint-driven oracle.

pub mod kernels;
pub mod operands;

#[cfg(test)]
mod tests;

use larql_models::config::{
    GateActivation, GateCombine, GatePlacement, GateSource, PositionPolicy, QkNormScope,
};

use super::super::graph::policy::AttentionSpan;
use super::{AttentionOp, ComponentOpPlan, FfnOp, LayerPlan, NormOp};
use crate::error::VindexError;
use kernels::{activate, matvec, norm, rope_rotate, sigmoid, softcap, softmax};
use operands::OperandStore;

/// Per-layer hidden-state taps, mirroring the production hook points.
#[derive(Debug)]
pub struct LayerTrace {
    /// Hidden state after the attention residual add, per position.
    pub post_attention: Vec<Vec<f32>>,
    /// Hidden state after the FFN residual add, per position.
    pub post_layer: Vec<Vec<f32>>,
}

/// The full execution record of one component over one token sequence.
#[derive(Debug)]
pub struct ExecutionTrace {
    pub layers: Vec<LayerTrace>,
    /// Final-normed hidden state of the last position.
    pub final_hidden: Vec<f32>,
    /// Logits of the last position, when the plan carries an output op.
    pub logits: Option<Vec<f32>>,
}

/// Execute a text-component plan over `tokens`, tracing every layer.
pub fn execute_text(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    tokens: &[u32],
) -> Result<ExecutionTrace, VindexError> {
    let embedding = plan.embedding.as_ref().ok_or_else(|| {
        VindexError::Parse(format!(
            "component `{}` has no embedding op — external hidden-state input is a later rung",
            plan.component
        ))
    })?;
    let table = store.load(&embedding.table)?;
    let hidden = embedding.table.shape[1];
    let mut h: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| {
            table[t as usize * hidden..(t as usize + 1) * hidden]
                .iter()
                .map(|v| v * embedding.scale)
                .collect()
        })
        .collect();

    let mut layers = Vec::with_capacity(plan.layers.len());
    for layer in &plan.layers {
        let trace = execute_layer(layer, store, &mut h, hidden)?;
        layers.push(trace);
    }

    let last = h.last().ok_or_else(|| {
        VindexError::Parse("cannot execute over an empty token sequence".to_string())
    })?;
    let final_hidden = match &plan.final_norm {
        Some(op) => apply_norm_op(op, store, last)?,
        None => last.clone(),
    };
    let logits = match &plan.output {
        Some(output) => {
            let weight = store.load(&output.projection)?;
            let vocab = output.projection.shape[0];
            let mut logits = matvec(&weight, vocab, hidden, &final_hidden);
            for logit in &mut logits {
                *logit *= output.multiplier as f32;
                if let Some(cap) = output.softcapping {
                    *logit = softcap(*logit, cap);
                }
            }
            Some(logits)
        }
        None => None,
    };
    Ok(ExecutionTrace {
        layers,
        final_hidden,
        logits,
    })
}

/// One decoder layer: norms and residuals exactly where the plan puts
/// them — placement is data, not code structure.
fn execute_layer(
    layer: &LayerPlan,
    store: &OperandStore,
    h: &mut [Vec<f32>],
    hidden: usize,
) -> Result<LayerTrace, VindexError> {
    let attn_out = attention(
        &layer.attention,
        &layer.pre_attention_norm,
        store,
        h,
        hidden,
    )?;
    for (row, out) in h.iter_mut().zip(&attn_out) {
        let out = match &layer.post_attention_norm {
            Some(op) => apply_norm_op(op, store, out)?,
            None => out.clone(),
        };
        for (a, b) in row.iter_mut().zip(&out) {
            *a += b;
        }
    }
    let post_attention = h.to_vec();

    for row in h.iter_mut() {
        let normed = apply_norm_op(&layer.pre_ffn_norm, store, row)?;
        let ffn_out = ffn(&layer.ffn, store, &normed, hidden)?;
        let ffn_out = match &layer.post_ffn_norm {
            Some(op) => apply_norm_op(op, store, &ffn_out)?,
            None => ffn_out,
        };
        for (a, b) in row.iter_mut().zip(&ffn_out) {
            *a += b;
        }
    }
    Ok(LayerTrace {
        post_attention,
        post_layer: h.to_vec(),
    })
}

/// The attention op: geometry, scales, span, position, QK normalisation
/// and the optional output gate — all plan arguments.
fn attention(
    op: &AttentionOp,
    pre_norm: &NormOp,
    store: &OperandStore,
    h: &[Vec<f32>],
    hidden: usize,
) -> Result<Vec<Vec<f32>>, VindexError> {
    let head_dim = op.head_dim;
    let q_rows = op.num_q_heads * head_dim;
    let kv_rows = op.num_kv_heads * head_dim;
    let group = op.num_q_heads / op.num_kv_heads;

    let w_q = store.load(&op.q)?;
    let w_k = store.load(&op.k)?;
    let w_v = store.load(&op.v)?;
    let w_o = store.load(&op.o)?;
    let qk_weights = match &op.qk_norm {
        Some(qk) => Some((store.load(&qk.q)?, store.load(&qk.k)?)),
        None => None,
    };
    let w_gate = match &op.output_gate {
        Some(gate) => Some(store.load(&gate.projection)?),
        None => None,
    };

    // Projections per position, with QK normalisation, query scale and
    // position encoding applied in the judged order.
    let mut queries = Vec::with_capacity(h.len());
    let mut keys = Vec::with_capacity(h.len());
    let mut values = Vec::with_capacity(h.len());
    let mut pre_rows = Vec::with_capacity(h.len());
    for (position, row) in h.iter().enumerate() {
        let pre = apply_norm_op(pre_norm, store, row)?;
        let mut q = matvec(&w_q, q_rows, hidden, &pre);
        let mut k = matvec(&w_k, kv_rows, hidden, &pre);
        let v = matvec(&w_v, kv_rows, hidden, &pre);

        apply_qk_norm(op, &qk_weights, pre_norm.eps, &mut q, &mut k, head_dim)?;
        for value in &mut q {
            *value *= op.query_scale as f32;
        }
        if let PositionPolicy::Rope { theta } = op.position {
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
        pre_rows.push(pre);
    }

    let mut out = Vec::with_capacity(h.len());
    for position in 0..h.len() {
        // Span: which key positions this query may attend to.
        let start = match (op.span, op.window) {
            (AttentionSpan::Sliding, Some(window)) => (position + 1).saturating_sub(window),
            _ => 0,
        };
        let mut concat = vec![0.0f32; q_rows];
        for q_head in 0..op.num_q_heads {
            let kv_head = q_head / group;
            let q_slice = &queries[position][q_head * head_dim..(q_head + 1) * head_dim];
            let mut scores: Vec<f32> = (start..=position)
                .map(|key_position| {
                    let k_slice = &keys[key_position][kv_head * head_dim..(kv_head + 1) * head_dim];
                    let mut dot = 0.0f32;
                    for (a, b) in q_slice.iter().zip(k_slice) {
                        dot += a * b;
                    }
                    let mut score = dot * op.score_scale as f32;
                    if let Some(cap) = op.logit_softcapping {
                        score = softcap(score, cap);
                    }
                    score
                })
                .collect();
            softmax(&mut scores);
            let head_out = &mut concat[q_head * head_dim..(q_head + 1) * head_dim];
            for (offset, key_position) in (start..=position).enumerate() {
                let v_slice = &values[key_position][kv_head * head_dim..(kv_head + 1) * head_dim];
                let weight = scores[offset];
                for (acc, v) in head_out.iter_mut().zip(v_slice) {
                    *acc += weight * v;
                }
            }
        }

        if let (Some(gate), Some(w_gate)) = (&op.output_gate, &w_gate) {
            // Exhaustive on the judged semantics: a new variant must be
            // implemented here before it can execute.
            let GateSource::AttentionInput = gate.spec.source;
            let GateActivation::Sigmoid = gate.spec.activation;
            let GateCombine::ElementwiseMultiply = gate.spec.combine;
            let GatePlacement::AfterAggregationBeforeOutputProjection = gate.spec.placement;
            let gate_values = matvec(w_gate, q_rows, hidden, &pre_rows[position]);
            for (c, g) in concat.iter_mut().zip(&gate_values) {
                *c *= sigmoid(*g);
            }
        }

        out.push(matvec(&w_o, hidden, q_rows, &concat));
    }
    Ok(out)
}

/// QK normalisation: weighted per-head when the plan binds norm weights,
/// parameter-free when the surface judged it. Epsilon rides with the
/// layer's norm surface (`pre_norm.eps`) — neither QK form declares its
/// own upstream.
fn apply_qk_norm(
    op: &AttentionOp,
    qk_weights: &Option<(Vec<f32>, Vec<f32>)>,
    eps: f64,
    q: &mut [f32],
    k: &mut [f32],
    head_dim: usize,
) -> Result<(), VindexError> {
    if let (Some(qk), Some((q_weight, k_weight))) = (&op.qk_norm, qk_weights) {
        match qk.scope {
            QkNormScope::PerHead => {
                for head in q.chunks_exact_mut(head_dim) {
                    let normed = norm(
                        larql_models::config::NormType::RmsNorm,
                        head,
                        q_weight,
                        qk.weight_offset,
                        eps,
                    );
                    head.copy_from_slice(&normed);
                }
                for head in k.chunks_exact_mut(head_dim) {
                    let normed = norm(
                        larql_models::config::NormType::RmsNorm,
                        head,
                        k_weight,
                        qk.weight_offset,
                        eps,
                    );
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
    if op.parameter_free_qk_norm.q {
        for head in q.chunks_exact_mut(head_dim) {
            let normed = norm(larql_models::config::NormType::RmsNorm, head, &[], 0.0, eps);
            head.copy_from_slice(&normed);
        }
    }
    if op.parameter_free_qk_norm.k {
        for head in k.chunks_exact_mut(head_dim) {
            let normed = norm(larql_models::config::NormType::RmsNorm, head, &[], 0.0, eps);
            head.copy_from_slice(&normed);
        }
    }
    Ok(())
}

/// The FFN op: gated or standard per the plan.
fn ffn(
    op: &FfnOp,
    store: &OperandStore,
    x: &[f32],
    hidden: usize,
) -> Result<Vec<f32>, VindexError> {
    let up = matvec(&store.load(&op.up)?, op.intermediate_size, hidden, x);
    let inner: Vec<f32> = match &op.gate {
        Some(gate_ref) => {
            let gate = matvec(&store.load(gate_ref)?, op.intermediate_size, hidden, x);
            gate.iter()
                .zip(&up)
                .map(|(g, u)| activate(op.activation, *g) * u)
                .collect()
        }
        None => up.iter().map(|u| activate(op.activation, *u)).collect(),
    };
    Ok(matvec(
        &store.load(&op.down)?,
        hidden,
        op.intermediate_size,
        &inner,
    ))
}

/// Apply one norm op to one vector.
fn apply_norm_op(op: &NormOp, store: &OperandStore, x: &[f32]) -> Result<Vec<f32>, VindexError> {
    let weight = store.load(&op.weight)?;
    Ok(norm(op.kind, x, &weight, op.weight_offset, op.eps))
}
