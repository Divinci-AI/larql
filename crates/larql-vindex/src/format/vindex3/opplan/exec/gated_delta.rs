//! Reference Gated DeltaNet: the recurrence, written to be read.
//!
//! Deliberately slow and literal. No fused kernel, no vectorisation, no
//! reassociation — this exists so someone can put it beside
//! `torch_recurrent_gated_delta_rule` in `transformers` and compare it line
//! by line. Speed is QW-4's problem.
//!
//! The state is owned here, on purpose. A DeltaNet layer's continuation
//! state is one dense `Dk × Dv` matrix per value head and does not grow
//! with the sequence, so it is not a KV cache and must not be forced
//! through one. Whether the engine should have a single abstraction
//! covering both is a real question, and it is QW-3's — answering it before
//! the arithmetic is proven would risk baking a wrong recurrence into a
//! nice-looking generic interface.

// Explicit index loops on purpose. This module's stated job is to sit
// beside `torch_recurrent_gated_delta_rule` and be checkable line by line,
// and the reference indexes `[..., i]` over named axes. Iterator chains
// would read better in isolation and worse against the thing they must be
// verified against — and verification is the entire point of the file.
#![allow(clippy::needless_range_loop)]

use super::super::GatedDeltaOp;

/// L2-normalisation epsilon, from the reference kernel's own default.
const L2NORM_EPS: f32 = 1e-6;

/// One layer's recurrent state: `[value_heads][key_head_dim][value_head_dim]`.
///
/// Held in f32 regardless of the weights' storage dtype. That is the
/// checkpoint's own instruction — Qwen3.8 declares `mamba_ssm_dtype:
/// float32` against a bf16 model — and it is not decoration: the state
/// feeds itself forward, so rounding compounds across the whole sequence
/// in a way a one-shot weight rounding does not.
#[derive(Debug, Clone, PartialEq)]
pub struct GatedDeltaState {
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    /// Row-major `[head][k][v]`.
    cells: Vec<f32>,
}

impl GatedDeltaState {
    /// The zero state a sequence starts from.
    pub fn zeros(op: &GatedDeltaOp) -> Self {
        Self {
            value_heads: op.num_value_heads,
            key_head_dim: op.key_head_dim,
            value_head_dim: op.value_head_dim,
            cells: vec![0.0; op.state_elements()],
        }
    }

    /// Adopt an existing state (a captured one, or a session's).
    pub fn from_cells(op: &GatedDeltaOp, cells: Vec<f32>) -> Result<Self, String> {
        if cells.len() != op.state_elements() {
            return Err(format!(
                "state has {} elements, this layer's geometry needs {}",
                cells.len(),
                op.state_elements()
            ));
        }
        Ok(Self {
            value_heads: op.num_value_heads,
            key_head_dim: op.key_head_dim,
            value_head_dim: op.value_head_dim,
            cells,
        })
    }

    pub fn cells(&self) -> &[f32] {
        &self.cells
    }

    fn at(&self, head: usize, k: usize, v: usize) -> usize {
        (head * self.key_head_dim + k) * self.value_head_dim + v
    }
}

/// One position's inputs to the recurrence, per value head, already split
/// and head-expanded by the caller.
///
/// Taking them pre-derived keeps this function the *recurrence* and nothing
/// else: the projections, convolution and head expansion are separate
/// stages with their own comparison planes.
pub struct RecurrenceStep<'a> {
    /// `[value_heads * key_head_dim]`, NOT yet L2-normalised or scaled.
    pub query: &'a [f32],
    /// `[value_heads * key_head_dim]`, NOT yet L2-normalised.
    pub key: &'a [f32],
    /// `[value_heads * value_head_dim]`.
    pub value: &'a [f32],
    /// `[value_heads]` — already `-exp(A_log) * softplus(a + dt_bias)`, so
    /// it is negative and `exp(g)` is a decay in `(0, 1]`.
    pub g: &'a [f32],
    /// `[value_heads]` — already through the sigmoid.
    pub beta: &'a [f32],
}

fn l2_normalise(row: &mut [f32]) {
    let sum_sq: f32 = row.iter().map(|x| x * x).sum();
    let inv = 1.0 / (sum_sq + L2NORM_EPS).sqrt();
    for x in row.iter_mut() {
        *x *= inv;
    }
}

/// Advance the state by one position and return that position's output.
///
/// Returns `[value_heads * value_head_dim]` — the recurrence's own output,
/// before the gated norm and the output projection.
///
/// Transcribed from `torch_recurrent_gated_delta_rule`. The order is the
/// specification, not an implementation detail:
///
/// ```text
/// S  = S * exp(g)                  decay first
/// kv = k · S                       read with the key
/// d  = (v - kv) * beta             the delta rule
/// S  = S + outer(k, d)             rank-1 write
/// o  = q · S                       read with the query, AFTER the write
/// ```
///
/// That last ordering is why a single-position test cannot validate this:
/// the current position reads a state it has just written, so an
/// implementation that reads before writing produces a plausible first
/// output and a wrong second one.
pub fn recurrence_step(
    op: &GatedDeltaOp,
    step: &RecurrenceStep<'_>,
    state: &mut GatedDeltaState,
) -> Vec<f32> {
    step_inner(op, step, state, Mutation::None)
}

/// Deliberate defects, for the negative controls.
///
/// Test-only, but they perturb the REAL function rather than a copy of it:
/// a control that mutates a duplicate proves only that the duplicate is
/// detectable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutation {
    None,
    /// Apply the query scale BEFORE the L2-norm, so normalisation undoes it.
    ScaleBeforeNorm,
    /// Read the state with q BEFORE the rank-1 write, so the current
    /// position cannot see its own contribution.
    ReadBeforeWrite,
    /// Skip the decay entirely.
    NoDecay,
    /// Drop beta from the delta rule.
    NoBeta,
    /// Use g directly instead of exp(g).
    RawGate,
}

fn step_inner(
    op: &GatedDeltaOp,
    step: &RecurrenceStep<'_>,
    state: &mut GatedDeltaState,
    mutation: Mutation,
) -> Vec<f32> {
    let (hv, dk, dv) = (op.num_value_heads, op.key_head_dim, op.value_head_dim);
    let mut out = vec![0.0f32; hv * dv];
    // The reference L2-normalises inside the kernel and applies the query
    // scale AFTERWARDS. Scaling first would rescale the normalisation and
    // is a different function.
    let scale = 1.0 / (dk as f32).sqrt();

    for h in 0..hv {
        let mut q: Vec<f32> = step.query[h * dk..(h + 1) * dk].to_vec();
        let mut k: Vec<f32> = step.key[h * dk..(h + 1) * dk].to_vec();
        if mutation == Mutation::ScaleBeforeNorm {
            for x in q.iter_mut() {
                *x *= scale;
            }
        }
        l2_normalise(&mut q);
        l2_normalise(&mut k);
        if mutation != Mutation::ScaleBeforeNorm {
            for x in q.iter_mut() {
                *x *= scale;
            }
        }
        let v = &step.value[h * dv..(h + 1) * dv];
        let decay = match mutation {
            Mutation::NoDecay => 1.0,
            Mutation::RawGate => step.g[h],
            _ => step.g[h].exp(),
        };
        let beta = if mutation == Mutation::NoBeta {
            1.0
        } else {
            step.beta[h]
        };

        for kk in 0..dk {
            for vv in 0..dv {
                let idx = state.at(h, kk, vv);
                state.cells[idx] *= decay;
            }
        }
        // kv = sum over the KEY axis, weighted by k.
        let mut kv = vec![0.0f32; dv];
        for kk in 0..dk {
            let kw = k[kk];
            for vv in 0..dv {
                kv[vv] += state.cells[state.at(h, kk, vv)] * kw;
            }
        }
        let delta: Vec<f32> = (0..dv).map(|vv| (v[vv] - kv[vv]) * beta).collect();
        let mut read = |state: &GatedDeltaState| {
            for kk in 0..dk {
                let qw = q[kk];
                for vv in 0..dv {
                    out[h * dv + vv] += state.cells[state.at(h, kk, vv)] * qw;
                }
            }
        };
        if mutation == Mutation::ReadBeforeWrite {
            read(state);
        }
        for kk in 0..dk {
            let kw = k[kk];
            for vv in 0..dv {
                let idx = state.at(h, kk, vv);
                state.cells[idx] += kw * delta[vv];
            }
        }
        if mutation != Mutation::ReadBeforeWrite {
            read(state);
        }
    }
    out
}

/// Run the recurrence with a deliberate defect. Negative controls only.
pub fn recurrence_step_mutated(
    op: &GatedDeltaOp,
    step: &RecurrenceStep<'_>,
    state: &mut GatedDeltaState,
    mutation: Mutation,
) -> Vec<f32> {
    step_inner(op, step, state, mutation)
}
