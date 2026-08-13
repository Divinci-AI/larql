//! The backend seam (V3-G5b-3b): what *executes* a plan, versus what the
//! plan *means*.
//!
//! One [`ComponentOpPlan`](super::super::ComponentOpPlan), one interpreter,
//! many backends. The interpreter in [`super`] owns every decision that is
//! semantics — operation ordering, residual ordering, layer traversal,
//! whether an optional operation exists at all, and how position and span
//! policy dispatch. A [`PlanBackend`] owns only the numerical realisation
//! of work it is handed.
//!
//! **Nothing in this file mentions a model family, and nothing in it takes
//! a plan type.** Backends receive primitives, judged enums, and *already
//! loaded* weight slices — never a `LayerPlan`, an `OperandRef`, or the
//! `OperandStore`. That is deliberate and load-bearing: a backend that
//! could resolve its own operands by name, or read the layer structure,
//! could quietly grow into a second implementation of the model and
//! disagree with the IR while still passing. It cannot reach the bytes,
//! so it cannot reinterpret them.
//!
//! The corollary for anyone adding a method: if a backend needs to ask
//! *whether* to do something, the seam is in the wrong place. It should
//! only ever be told what to compute.

use larql_models::config::{
    Activation, AttentionGateSpec, NormType, ParameterFreeQkNorm, PositionPolicy, QkNormScope,
};

use super::super::super::graph::policy::AttentionSpan;
use crate::error::VindexError;

/// One normalisation, fully resolved.
///
/// `weight` empty means a parameter-free application (statistic only) —
/// the interpreter decides that from the plan, never the backend.
pub struct NormCall<'a> {
    pub kind: NormType,
    pub x: &'a [f32],
    pub weight: &'a [f32],
    pub weight_offset: f32,
    pub eps: f64,
}

/// One `[out, in]` row-major projection applied to one vector.
pub struct ProjectCall<'a> {
    pub weight: &'a [f32],
    pub out_dim: usize,
    pub in_dim: usize,
    pub x: &'a [f32],
}

/// QK normalisation weights and scope, when the plan binds them.
pub struct QkNormCall<'a> {
    pub scope: QkNormScope,
    pub weight_offset: f32,
    pub q_weight: &'a [f32],
    pub k_weight: &'a [f32],
}

/// The attention output gate, when the surface judged one.
pub struct GateCall<'a> {
    pub spec: AttentionGateSpec,
    pub weight: &'a [f32],
}

/// One attention operation over a whole sequence, fully resolved.
///
/// `inputs` are the attention *inputs* — already normalised by the
/// interpreter — because the judged gate reads that same vector, and
/// handing the backend one operand for both uses removes any chance of
/// the two drifting apart.
pub struct AttentionCall<'a> {
    pub inputs: &'a [Vec<f32>],
    pub hidden: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub w_q: &'a [f32],
    pub w_k: &'a [f32],
    pub w_v: &'a [f32],
    pub w_o: &'a [f32],
    pub qk_norm: Option<QkNormCall<'a>>,
    pub parameter_free_qk_norm: ParameterFreeQkNorm,
    /// Epsilon for both QK-norm forms; rides with the layer's norm
    /// surface because neither form declares its own.
    pub qk_norm_eps: f64,
    /// `None` = no query-scale operation, never an invented 1.0.
    pub query_scale: Option<f64>,
    pub score_scale: f64,
    pub logit_softcapping: Option<f32>,
    pub position: PositionPolicy,
    pub span: AttentionSpan,
    pub window: Option<usize>,
    pub gate: Option<GateCall<'a>>,
}

/// One feed-forward operation over one vector, fully resolved.
///
/// `gate` present means gated; absent means standard. Again the
/// interpreter reads that from the plan.
pub struct FfnCall<'a> {
    pub x: &'a [f32],
    pub hidden: usize,
    pub intermediate: usize,
    pub gate: Option<&'a [f32]>,
    pub up: &'a [f32],
    pub down: &'a [f32],
    pub activation: Activation,
}

/// The numerical realisation of a plan's operations.
///
/// Every method is total over its arguments: the caller has already
/// decided the operation happens. A backend may fail on work it cannot
/// perform (an unimplemented QK-norm scope, a device error), but it may
/// not decline work on semantic grounds — that judgment was made before
/// the call.
pub trait PlanBackend {
    /// A name for diagnostics and parity reports. Not dispatched on.
    fn name(&self) -> &str;

    /// Look up one embedding row, applying the scale operation when the
    /// plan carries one. `scale` `None` = no such operation, so the row
    /// is returned unscaled rather than multiplied by an identity.
    fn embed(&self, table: &[f32], hidden: usize, token: u32, scale: Option<f32>) -> Vec<f32>;

    fn norm(&self, call: NormCall<'_>) -> Vec<f32>;

    fn project(&self, call: ProjectCall<'_>) -> Vec<f32>;

    /// Attention over the whole sequence, returning one output vector per
    /// position (post output-projection).
    fn attention(&self, call: AttentionCall<'_>) -> Result<Vec<Vec<f32>>, VindexError>;

    /// Fallible for the same reason as [`Self::attention`]: a backend
    /// with no kernel for a judged variant must say so, not borrow
    /// another backend's arithmetic to fill the gap.
    fn ffn(&self, call: FfnCall<'_>) -> Result<Vec<f32>, VindexError>;

    /// Vocabulary projection plus the head's optional multiplier and
    /// softcap, in that order.
    fn output_head(
        &self,
        projection: &[f32],
        vocab: usize,
        hidden: usize,
        x: &[f32],
        multiplier: Option<f64>,
        softcapping: Option<f32>,
    ) -> Vec<f32>;

    /// Add `delta` into `acc` elementwise — the residual write.
    ///
    /// A method rather than a loop in the interpreter because residual
    /// accumulation order is exactly the kind of thing a fused production
    /// kernel wants to own, and because a backend that reassociates it
    /// should have to say so.
    fn residual_add(&self, acc: &mut [f32], delta: &[f32]);
}
