//! Operands lowered into the backend's execution form, once.
//!
//! A [`ComponentOpPlan`] names its operands; it does not hold them.
//! Turning those names into arithmetic-ready weights — widening,
//! re-quantising to the backend's declared format, and handing the
//! backend a chance to place them on a device — is the expensive step,
//! and it is *model-shaped*, not request-shaped.
//!
//! Before this module both traversals loaded operands as they went:
//! [`DecodeSession`](super::decode::DecodeSession) built its own set at
//! construction, and the batch traversal called `store.load(...)` per
//! layer (per *position*, for norms). A server that batch-prefills and
//! then decodes therefore materialised the whole model twice per
//! request — measured at 3.8 s + 3.3 s against 0.13 s of actual decode
//! on a 3 B container.
//!
//! [`PreparedOperands`] is that state made explicit. It is deliberately
//! *not* a cache inside the operand loader: residency is a fact about a
//! served model, and hiding it behind a memoised loader would leave
//! device placement, accounting, and slicing with nowhere to live.
//!
//! # Composition with the operand seam
//!
//! Preparation resolves through an [`OperandSource`], not the bare
//! store, so a prepared image is "the **effective** operands for this
//! source" — base representation plus whatever overlay it carries.
//! That keeps the two seams orthogonal and in the right order:
//!
//! ```text
//! base representation + overlay → OperandSource → PreparedOperands → executor
//! ```
//!
//! An image is therefore immutable *for the source it was prepared
//! from*: a session composing new edits prepares its own view rather
//! than mutating the shared one, so one image can serve every
//! concurrent request that shares its overlay.
//!
//! # Slicing
//!
//! Preparation takes an [`ExecutionSlice`] because a VINDEX3 component
//! is not only ever executed whole. A shard that owns layers 10–19, an
//! attention-only node, or an expert server all want *part* of the same
//! plan prepared, and none of them should pay for operands they will
//! never execute. `Full` is the common case; the variants below are the
//! seam the decoupled surfaces grow from, and preparation refuses a
//! slice the plan cannot satisfy rather than silently preparing less.

use super::backend::{MatrixClass, NormCall, PlanBackend, WeightSlice};
use super::experts::FfnOperands;
use super::operands::{OperandSource, SourceStamp};
use super::weights::{load_weight, LoadedWeight};
use super::AttentionOperands;
use crate::error::VindexError;

use super::super::{ComponentOpPlan, GatedDeltaOp, LayerAttention, NormOp, OperandRef, OutputOp};

/// Which part of a component's program to prepare.
///
/// The plan is the authority for what exists; a slice says which of it
/// this process is responsible for executing. Preparing a slice loads
/// only that slice's operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionSlice {
    /// Embedding, every layer, final norm and head — a whole model.
    Full,
    /// Layers `[start, end)` of the stack and nothing else: no
    /// embedding, no final norm, no head. Hidden states in, hidden
    /// states out — the shape a layer-range shard executes.
    LayerRange { start: usize, end: usize },
}

impl ExecutionSlice {
    /// The layer indices this slice covers, as a half-open range.
    pub fn layers(&self, plan: &ComponentOpPlan) -> std::ops::Range<usize> {
        match self {
            Self::Full => 0..plan.layers.len(),
            Self::LayerRange { start, end } => *start..*end,
        }
    }

    /// Whether the slice carries the stack's ends — embedding on the
    /// way in, final norm and output head on the way out.
    pub fn is_whole_stack(&self) -> bool {
        matches!(self, Self::Full)
    }

    /// Refuse a slice the plan cannot satisfy. A shard asked for layers
    /// the model does not have is a deployment error, and preparing
    /// "as much as exists" would serve a silently wrong submodel — the
    /// same failure the V3 load options used to have.
    fn validate(&self, plan: &ComponentOpPlan) -> Result<(), VindexError> {
        let Self::LayerRange { start, end } = self else {
            return Ok(());
        };
        if start >= end {
            return Err(VindexError::Parse(format!(
                "execution slice {start}..{end} is empty — a slice must cover at least one layer"
            )));
        }
        if *end > plan.layers.len() {
            return Err(VindexError::Parse(format!(
                "execution slice {start}..{end} is outside component `{}`, which has {} layers",
                plan.component,
                plan.layers.len()
            )));
        }
        Ok(())
    }
}

/// One norm site's weight, held resident beside the op that names it.
pub(super) struct PreparedNorm {
    op: NormOp,
    weight: Vec<f32>,
}

impl PreparedNorm {
    fn load(op: &NormOp, store: OperandSource<'_>) -> Result<Self, VindexError> {
        Ok(Self {
            op: op.clone(),
            weight: store.load(&op.weight)?,
        })
    }

    pub(super) fn apply<B: PlanBackend + ?Sized>(&self, backend: &B, x: &[f32]) -> Vec<f32> {
        backend.norm(NormCall {
            kind: self.op.kind,
            x,
            weight: &self.weight,
            weight_offset: self.op.weight_offset,
            eps: self.op.eps,
        })
    }
}

/// One layer's operands, lowered into the backend's execution form.
pub(super) struct PreparedLayer {
    pub(super) pre_attention: PreparedNorm,
    pub(super) attention: PreparedAttention,
    pub(super) post_attention: Option<PreparedNorm>,
    pub(super) pre_ffn: PreparedNorm,
    pub(super) ffn: FfnOperands,
    pub(super) post_ffn: Option<PreparedNorm>,
    /// The layer's output scalar, when the plan carries one.
    pub(super) layer_scale: Option<f32>,
}

/// Which attention-class operator a prepared layer holds operands for.
///
/// An enum, not `Option<AttentionOperands>` and not "softmax unless
/// proven otherwise": a layer runs exactly one operator, and the
/// alternative spellings both make "I could not tell" indistinguishable
/// from "it is softmax". Qwen3.8 is 48 layers where that difference is
/// the whole model.
///
/// Chosen from the op plan's `LayerAttention`, which the op builder
/// derived from operand EVIDENCE — so the operands loaded here and the
/// operator dispatched later cannot disagree.
pub(super) enum PreparedAttention {
    Softmax(Box<AttentionOperands>),
    GatedDelta(Box<GatedDeltaOperands>),
}

impl PreparedAttention {
    /// Matrix operands for device placement.
    ///
    /// A recurrence contributes none: its nine operands are elementwise
    /// glue and a depthwise convolution, not the matrix traffic a device
    /// backend holds resident — and no device backend runs this operator
    /// yet, so placing them would reserve memory nothing reads.
    fn weight_slices(&self) -> Vec<WeightSlice<'_>> {
        match self {
            Self::Softmax(ops) => ops.weight_slices(),
            Self::GatedDelta(_) => Vec::new(),
        }
    }
}

/// The nine operands a Gated DeltaNet layer reads, loaded once.
pub(super) struct GatedDeltaOperands {
    pub(super) op: GatedDeltaOp,
    pub(super) in_proj_qkv: Vec<f32>,
    pub(super) in_proj_a: Vec<f32>,
    pub(super) in_proj_b: Vec<f32>,
    pub(super) in_proj_z: Vec<f32>,
    pub(super) conv1d: Vec<f32>,
    pub(super) a_log: Vec<f32>,
    pub(super) dt_bias: Vec<f32>,
    pub(super) norm: Vec<f32>,
    pub(super) out_proj: Vec<f32>,
    pub(super) norm_eps: f32,
}

impl GatedDeltaOperands {
    fn load(
        op: &GatedDeltaOp,
        store: OperandSource<'_>,
        norm_eps: f32,
    ) -> Result<Self, VindexError> {
        // f32 throughout: the recurrence is elementwise glue and a
        // convolution, not the matrix traffic a backend picks a format
        // for. The reference path is the only consumer today.
        let load = |r: &OperandRef| store.load(r);
        Ok(Self {
            op: op.clone(),
            in_proj_qkv: load(&op.in_proj_qkv)?,
            in_proj_a: load(&op.in_proj_a)?,
            in_proj_b: load(&op.in_proj_b)?,
            in_proj_z: load(&op.in_proj_z)?,
            conv1d: load(&op.conv1d)?,
            a_log: load(&op.a_log)?,
            dt_bias: load(&op.dt_bias)?,
            norm: load(&op.norm)?,
            out_proj: load(&op.out_proj)?,
            norm_eps,
        })
    }

    pub(super) fn weights(&self) -> super::gated_delta::GatedDeltaWeights<'_> {
        super::gated_delta::GatedDeltaWeights {
            in_proj_qkv: &self.in_proj_qkv,
            in_proj_a: &self.in_proj_a,
            in_proj_b: &self.in_proj_b,
            in_proj_z: &self.in_proj_z,
            conv1d: &self.conv1d,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            norm: &self.norm,
            out_proj: &self.out_proj,
            norm_eps: self.norm_eps,
        }
    }
}

/// A component's operands, lowered once for a given slice and backend.
///
/// Immutable for its lifetime: this is the canonical base model. A
/// session that carries an overlay composes *over* these operands
/// rather than mutating them, so one prepared image can serve every
/// concurrent request on the model.
pub struct PreparedOperands {
    /// Which effective source this image was compiled from.
    stamp: SourceStamp,
    slice: ExecutionSlice,
    hidden: usize,
    /// Present only for a slice that carries the stack's input end.
    embed_table: Option<Vec<f32>>,
    /// Plan index of `layers[0]`, so a sliced image can still address
    /// the plan's per-layer ops and the KV state's layer rows.
    first_layer: usize,
    layers: Vec<PreparedLayer>,
    final_norm: Option<PreparedNorm>,
    output: Option<(OutputOp, LoadedWeight)>,
}

impl PreparedOperands {
    /// Lower `slice` of `plan`'s operands into `backend`'s execution
    /// form, and give the backend its chance to place them (device
    /// residency). Every operand this slice needs is loaded here, and
    /// none of it is loaded again.
    pub fn load<'s, B: PlanBackend + ?Sized>(
        plan: &ComponentOpPlan,
        store: impl Into<OperandSource<'s>>,
        backend: &B,
        slice: ExecutionSlice,
    ) -> Result<Self, VindexError> {
        let store = store.into();
        slice.validate(plan)?;
        let stamp = store.stamp();
        let whole = slice.is_whole_stack();
        let embedding = plan.embedding.as_ref().ok_or_else(|| {
            VindexError::Parse(format!(
                "component `{}` has no embedding op — external hidden-state input is a later rung",
                plan.component
            ))
        })?;
        let hidden = embedding.table.shape[1];
        let embed_table = if whole {
            Some(store.load(&embedding.table)?)
        } else {
            None
        };

        let range = slice.layers(plan);
        let first_layer = range.start;
        let mut layers = Vec::with_capacity(range.len());
        for layer in &plan.layers[range] {
            layers.push(PreparedLayer {
                pre_attention: PreparedNorm::load(&layer.pre_attention_norm, store)?,
                // The operator is decided here, from the plan, and the
                // operands follow it. No layer is prepared as softmax by
                // default.
                attention: match &layer.attention {
                    LayerAttention::Softmax(op) => {
                        PreparedAttention::Softmax(Box::new(AttentionOperands::load(
                            op,
                            store,
                            backend.weight_format(MatrixClass::AttentionProjection),
                        )?))
                    }
                    LayerAttention::GatedDelta(op) => PreparedAttention::GatedDelta(Box::new(
                        GatedDeltaOperands::load(op, store, layer.pre_attention_norm.eps as f32)?,
                    )),
                },
                post_attention: layer
                    .post_attention_norm
                    .as_ref()
                    .map(|op| PreparedNorm::load(op, store))
                    .transpose()?,
                pre_ffn: PreparedNorm::load(&layer.pre_ffn_norm, store)?,
                ffn: FfnOperands::load(
                    &layer.ffn,
                    store,
                    backend.weight_format(MatrixClass::FfnProjection),
                )?,
                post_ffn: layer
                    .post_ffn_norm
                    .as_ref()
                    .map(|op| PreparedNorm::load(op, store))
                    .transpose()?,
                layer_scale: layer
                    .layer_scale
                    .as_ref()
                    .map(|op| store.load(op).and_then(|v| super::layer_scalar_of(&v)))
                    .transpose()?,
            });
        }

        let final_norm = if whole {
            plan.final_norm
                .as_ref()
                .map(|op| PreparedNorm::load(op, store))
                .transpose()?
        } else {
            None
        };
        let output = if whole {
            plan.output
                .as_ref()
                .map(|op| {
                    Ok::<_, VindexError>((
                        op.clone(),
                        load_weight(
                            store,
                            &op.projection,
                            backend.weight_format(MatrixClass::OutputHead),
                        )?,
                    ))
                })
                .transpose()?
        } else {
            None
        };

        let prepared = Self {
            stamp,
            slice,
            hidden,
            embed_table,
            first_layer,
            layers,
            final_norm,
            output,
        };
        prepared.place(backend);
        Ok(prepared)
    }

    /// Hand every matrix operand to the backend once, so a device
    /// backend can hold the model resident for this image's lifetime.
    fn place<B: PlanBackend + ?Sized>(&self, backend: &B) {
        let mut weights: Vec<WeightSlice<'_>> = Vec::new();
        for layer in &self.layers {
            weights.extend(layer.attention.weight_slices());
            weights.extend(layer.ffn.weight_slices());
        }
        if let Some((_, projection)) = &self.output {
            weights.push(projection.slice());
        }
        backend.prepare(&weights);
    }

    /// The slice this image was prepared for.
    pub fn slice(&self) -> &ExecutionSlice {
        &self.slice
    }

    /// The effective source this image was compiled from.
    pub fn source_stamp(&self) -> SourceStamp {
        self.stamp
    }

    /// Whether this image still describes `source`.
    ///
    /// False after any overlay mutation, and for a different store or a
    /// different override set. A caller that has the source in hand
    /// should ask before reusing a cached image; one that does not
    /// (the serve path, which holds only its own image) is safe by
    /// ownership — it has nothing else to confuse it with.
    pub fn is_current_for(&self, source: &OperandSource<'_>) -> bool {
        self.stamp == source.stamp()
    }

    /// [`Self::is_current_for`] as a refusal, for callers that would
    /// otherwise execute a stale image.
    pub fn ensure_current_for(&self, source: &OperandSource<'_>) -> Result<(), VindexError> {
        if self.is_current_for(source) {
            return Ok(());
        }
        Err(VindexError::Parse(
            "this prepared image was compiled from a different effective operand source — \
             the overlay changed, or it belongs to another container. Re-prepare rather than \
             executing a stale compilation of the model."
                .to_string(),
        ))
    }

    /// Hidden width, read from the plan's embedding op.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// How many layers this image can execute.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Whether this image carries an output head (only a whole-stack
    /// slice does).
    pub fn has_output(&self) -> bool {
        self.output.is_some()
    }

    pub(super) fn embed_table(&self) -> Option<&[f32]> {
        self.embed_table.as_deref()
    }

    pub(super) fn first_layer(&self) -> usize {
        self.first_layer
    }

    pub(super) fn layers(&self) -> &[PreparedLayer] {
        &self.layers
    }

    pub(super) fn final_norm(&self) -> Option<&PreparedNorm> {
        self.final_norm.as_ref()
    }

    pub(super) fn output(&self) -> Option<&(OutputOp, LoadedWeight)> {
        self.output.as_ref()
    }
}
