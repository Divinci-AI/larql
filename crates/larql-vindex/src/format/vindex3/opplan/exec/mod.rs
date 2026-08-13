//! The plan interpreter (V3-G5b-2 Stage A, V3-G5b-3b seam).
//!
//! Executes a [`ComponentOpPlan`] — and **nothing else**. Every argument
//! comes from the plan (which came from the container); every operand
//! loads through the closure-verified `object → representation → segment`
//! path; every judged enum is matched exhaustively so an unjudged variant
//! is a compile error, not a guess. There is no family name, no layer
//! arithmetic, no HF tensor name, and no default anywhere in this module.
//!
//! This file owns *meaning*: operation ordering, residual ordering, layer
//! traversal, whether an optional operation exists, and how position and
//! span policy dispatch. A [`PlanBackend`] owns only arithmetic. One
//! interpreter drives every backend, so a second implementation cannot
//! quietly become a second reading of the model — see [`backend`].
//!
//! The trace mirrors the production forward's hook points
//! (`post_attention` = after the attention residual add, `post_layer` =
//! after the FFN residual add) so parity can compare layer by layer
//! against a checkpoint-driven oracle.

pub mod backend;
pub mod kernels;
pub mod operands;
pub mod production;
pub mod reference;

#[cfg(test)]
mod tests;

use super::{AttentionOp, ComponentOpPlan, LayerPlan, NormOp};
use crate::error::VindexError;
use backend::{AttentionCall, FfnCall, GateCall, NormCall, PlanBackend, ProjectCall, QkNormCall};
use operands::OperandStore;
use reference::ReferenceBackend;

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
    /// The residual *entering* layer 0, per position — everything the
    /// embedding op produced and nothing else.
    ///
    /// Captured because a layer-by-layer comparison needs somewhere to
    /// stand before layer 0: if the two sides already disagree here, no
    /// per-layer margin below means what it appears to mean. It is the
    /// same tap `scripts/dump_layers_hf.py` takes with a pre-hook on
    /// layer 0.
    pub embedded: Vec<Vec<f32>>,
    pub layers: Vec<LayerTrace>,
    /// Final-normed hidden state of the last position.
    pub final_hidden: Vec<f32>,
    /// Logits of the last position, when the plan carries an output op.
    pub logits: Option<Vec<f32>>,
}

/// Execute a text-component plan on the reference backend.
///
/// The semantic anchor: naive f32, sharing no arithmetic with
/// `larql-compute`.
pub fn execute_text(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    tokens: &[u32],
) -> Result<ExecutionTrace, VindexError> {
    execute_plan(plan, store, tokens, &ReferenceBackend::new())
}

/// Execute a text-component plan over `tokens` on `backend`, tracing
/// every layer.
///
/// The backend is a parameter, not a branch: nothing below reads its
/// identity, and swapping it must not change which operations run.
pub fn execute_plan<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    tokens: &[u32],
    backend: &B,
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
        .map(|&t| backend.embed(&table, hidden, t, embedding.scale))
        .collect();
    // The judged embedding normalisation, when the plan carries one. It
    // is weightless — no operand, hence the empty weight slice — and it
    // runs *after* any embedding scale, matching the upstream order in
    // which the scale belongs to the table and the norm to the lookup.
    if let Some(norm) = embedding.norm {
        for row in h.iter_mut() {
            *row = backend.norm(NormCall {
                kind: norm.kind,
                x: row,
                weight: &[],
                weight_offset: 0.0,
                eps: norm.eps,
            });
        }
    }
    let embedded = h.clone();

    let mut layers = Vec::with_capacity(plan.layers.len());
    for layer in &plan.layers {
        layers.push(execute_layer(layer, store, &mut h, hidden, backend)?);
    }

    let last = h.last().ok_or_else(|| {
        VindexError::Parse("cannot execute over an empty token sequence".to_string())
    })?;
    let final_hidden = match &plan.final_norm {
        Some(op) => apply_norm_op(op, store, last, backend)?,
        None => last.clone(),
    };
    let logits = match &plan.output {
        Some(output) => {
            let weight = store.load(&output.projection)?;
            let vocab = output.projection.shape[0];
            Some(backend.output_head(
                &weight,
                vocab,
                hidden,
                &final_hidden,
                output.multiplier,
                output.softcapping,
            ))
        }
        None => None,
    };
    Ok(ExecutionTrace {
        embedded,
        layers,
        final_hidden,
        logits,
    })
}

/// One decoder layer: norms and residuals exactly where the plan puts
/// them — placement is data, not code structure.
fn execute_layer<B: PlanBackend + ?Sized>(
    layer: &LayerPlan,
    store: &OperandStore,
    h: &mut [Vec<f32>],
    hidden: usize,
    backend: &B,
) -> Result<LayerTrace, VindexError> {
    // The attention input is normalised here, once, and handed to the
    // backend — the judged gate reads the same vector, so producing it
    // in one place is what keeps the two from drifting apart.
    let mut inputs = Vec::with_capacity(h.len());
    for row in h.iter() {
        inputs.push(apply_norm_op(
            &layer.pre_attention_norm,
            store,
            row,
            backend,
        )?);
    }
    let attn_out = attention(
        &layer.attention,
        &inputs,
        layer.pre_attention_norm.eps,
        store,
        hidden,
        backend,
    )?;
    for (row, out) in h.iter_mut().zip(&attn_out) {
        let out = match &layer.post_attention_norm {
            Some(op) => apply_norm_op(op, store, out, backend)?,
            None => out.clone(),
        };
        backend.residual_add(row, &out);
    }
    let post_attention = h.to_vec();

    // FFN operands load once per layer, not once per position. They are
    // the bulk of a decoder layer's weight, and `OperandStore::load`
    // allocates a fresh f32 copy per call — re-reading them for every
    // token would dominate the run on a real model without changing a
    // single number.
    let up = store.load(&layer.ffn.up)?;
    let down = store.load(&layer.ffn.down)?;
    let gate = match &layer.ffn.gate {
        Some(gate_ref) => Some(store.load(gate_ref)?),
        None => None,
    };
    for row in h.iter_mut() {
        let normed = apply_norm_op(&layer.pre_ffn_norm, store, row, backend)?;
        let ffn_out = backend.ffn(FfnCall {
            x: &normed,
            hidden,
            intermediate: layer.ffn.intermediate_size,
            gate: gate.as_deref(),
            up: &up,
            down: &down,
            activation: layer.ffn.activation,
        })?;
        let ffn_out = match &layer.post_ffn_norm {
            Some(op) => apply_norm_op(op, store, &ffn_out, backend)?,
            None => ffn_out,
        };
        backend.residual_add(row, &ffn_out);
    }
    Ok(LayerTrace {
        post_attention,
        post_layer: h.to_vec(),
    })
}

/// Load the attention operands and hand the backend a fully resolved
/// call. Every judged fact travels as an argument; none is re-derived.
fn attention<B: PlanBackend + ?Sized>(
    op: &AttentionOp,
    inputs: &[Vec<f32>],
    qk_norm_eps: f64,
    store: &OperandStore,
    hidden: usize,
    backend: &B,
) -> Result<Vec<Vec<f32>>, VindexError> {
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
    let qk_norm = match (&op.qk_norm, &qk_weights) {
        (Some(qk), Some((q_weight, k_weight))) => Some(QkNormCall {
            scope: qk.scope,
            weight_offset: qk.weight_offset,
            q_weight,
            k_weight,
        }),
        _ => None,
    };
    let gate = match (&op.output_gate, &w_gate) {
        (Some(gate), Some(weight)) => Some(GateCall {
            spec: gate.spec,
            weight,
        }),
        _ => None,
    };
    backend.attention(AttentionCall {
        inputs,
        hidden,
        num_q_heads: op.num_q_heads,
        num_kv_heads: op.num_kv_heads,
        head_dim: op.head_dim,
        w_q: &w_q,
        w_k: &w_k,
        w_v: &w_v,
        w_o: &w_o,
        qk_norm,
        parameter_free_qk_norm: op.parameter_free_qk_norm,
        qk_norm_eps,
        query_scale: op.query_scale,
        score_scale: op.score_scale,
        logit_softcapping: op.logit_softcapping,
        position: op.position,
        span: op.span,
        window: op.window,
        gate,
    })
}

/// Apply one norm op to one vector.
fn apply_norm_op<B: PlanBackend + ?Sized>(
    op: &NormOp,
    store: &OperandStore,
    x: &[f32],
    backend: &B,
) -> Result<Vec<f32>, VindexError> {
    let weight = store.load(&op.weight)?;
    Ok(backend.norm(NormCall {
        kind: op.kind,
        x,
        weight: &weight,
        weight_offset: op.weight_offset,
        eps: op.eps,
    }))
}

/// Project one vector through an `[out, in]` weight.
///
/// Kept as a named helper so the interpreter never open-codes a matvec:
/// every projection in a plan goes through the backend.
#[allow(dead_code)]
fn project<B: PlanBackend + ?Sized>(
    backend: &B,
    weight: &[f32],
    out_dim: usize,
    in_dim: usize,
    x: &[f32],
) -> Vec<f32> {
    backend.project(ProjectCall {
        weight,
        out_dim,
        in_dim,
        x,
    })
}
