//! What has to survive between decode steps.
//!
//! LARQL no longer has a KV-cache abstraction. It has a **continuation
//! state** abstraction, of which KV is one form. That is not a rename: a
//! hybrid checkpoint carries two genuinely different mechanisms at once,
//! and Qwen3.8-27B carries 48 of one and 16 of the other.
//!
//! ```text
//! softmax layer     grows a sequence-indexed K/V store
//! recurrent layer   carries a fixed-size matrix, constant in sequence length
//! ```
//!
//! # The division of labour
//!
//! Continuation geometry says **what must persist**. The operator says
//! **how it changes**. Nothing here knows what a Gated DeltaNet is, and
//! that is the point: the next consumer of [`RecurrentGeometry`] is Kimi's
//! KDA, a different update rule over the same kind of durable storage. If
//! this type had been shaped around DeltaNet's semantics, KDA would need a
//! second variant and the abstraction would have failed at its first real
//! test.
//!
//! # Why an enum rather than `Option<LayerKvGeometry>`
//!
//! `Option` would collapse four distinct facts into one `None`: no state at
//! all, recurrent state instead of KV, unknown, and unsupported. A planner
//! reading `None` could not tell "this layer needs nothing" from "this
//! layer needs something I cannot describe", and would size memory for both
//! identically.

use larql_models::inventory::report::RecurrentStateDtype;

use super::super::{ComponentOpPlan, LayerAttention};
use super::kv::LayerKvGeometry;

/// How a recurrent state begins a sequence.
///
/// Recorded rather than assumed. A zero start is what the judged operators
/// declare, but "the reference starts from zeros" and "any recurrence must
/// start from zeros" are different claims, and only the first is evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateInitialization {
    Zeros,
}

/// A durable state whose size does not depend on how many positions have
/// been seen.
///
/// Deliberately says nothing about the operator that owns it — no head
/// semantics, no update rule, no family name. `shape` is whatever that
/// operator's state is; for a Gated DeltaNet layer it happens to be
/// `[value_heads, key_head_dim, value_head_dim]`, but this type does not
/// know that and must not learn it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentGeometry {
    pub shape: Vec<usize>,
    pub dtype: RecurrentStateDtype,
    pub initialization: StateInitialization,
}

impl RecurrentGeometry {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn bytes(&self) -> usize {
        self.elements()
            * match self.dtype {
                RecurrentStateDtype::Float32 => 4,
            }
    }
}

/// One layer's continuation requirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerContinuationGeometry {
    /// Grows with the sequence.
    Kv(LayerKvGeometry),
    /// Constant in the sequence.
    Recurrent(RecurrentGeometry),
    /// Nothing survives this layer.
    ///
    /// No judged operator produces this yet — it is here because "this
    /// layer needs nothing" is a distinct fact from "this layer needs
    /// something I cannot describe", and collapsing them is exactly what
    /// `Option<LayerKvGeometry>` did.
    Stateless,
}

impl LayerContinuationGeometry {
    /// Elements this layer must retain after `positions` steps.
    ///
    /// The asymmetry is the whole architectural claim, expressed as
    /// arithmetic rather than as a label: a KV layer's answer scales with
    /// `positions` and a recurrent layer's does not.
    pub fn elements_at(&self, positions: usize) -> usize {
        match self {
            // K and V, one row of `kv_dim` each, per position.
            Self::Kv(kv) => kv.kv_dim * 2 * positions,
            Self::Recurrent(r) => r.elements(),
            Self::Stateless => 0,
        }
    }

    pub fn kv(&self) -> Option<&LayerKvGeometry> {
        match self {
            Self::Kv(kv) => Some(kv),
            _ => None,
        }
    }

    pub fn recurrent(&self) -> Option<&RecurrentGeometry> {
        match self {
            Self::Recurrent(r) => Some(r),
            _ => None,
        }
    }
}

/// Every layer's continuation requirement, in layer order.
///
/// The authoritative seam. [`plan_kv_geometry`](super::kv::plan_kv_geometry)
/// remains only as a compatibility adapter and refuses any model this
/// cannot be flattened for.
pub fn plan_continuation_geometry(
    plan: &ComponentOpPlan,
) -> Result<Vec<LayerContinuationGeometry>, String> {
    plan.layers
        .iter()
        .map(|layer| match &layer.attention {
            LayerAttention::Softmax(op) => Ok(LayerContinuationGeometry::Kv(LayerKvGeometry {
                kv_dim: op.num_kv_heads * op.head_dim,
                window: op.window,
            })),
            LayerAttention::GatedDelta(op) => {
                // A recurrence must be held at SOME precision, and the
                // planner does not get to pick one. An undeclared state
                // dtype means the checkpoint never said, or said it in a
                // spelling this build does not represent — either way,
                // sizing it at the model's bulk dtype would run the
                // recurrence at a precision its author did not choose, and
                // the whole reason `mamba_ssm_dtype` exists is that the two
                // differ. Refuse instead.
                let dtype = op.state_dtype.ok_or_else(|| {
                    format!(
                        "layer {} carries a recurrence whose state precision the \
                         checkpoint never declared; continuation state cannot be \
                         sized without it",
                        layer.layer
                    )
                })?;
                Ok(LayerContinuationGeometry::Recurrent(RecurrentGeometry {
                    shape: vec![op.num_value_heads, op.key_head_dim, op.value_head_dim],
                    dtype,
                    initialization: StateInitialization::Zeros,
                }))
            }
        })
        .collect()
}

/// One layer's durable state, mirroring [`LayerContinuationGeometry`].
///
/// Storage only. There is deliberately no update method, no callback and no
/// operator handle on any variant: the state owns what survives between
/// steps, and the operator owns how it changes. A state that knew how a
/// Gated DeltaNet updates itself would have to learn KDA's rule too, and
/// then every future recurrence's — which is how a storage type becomes a
/// dispatch table.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerContinuationState {
    /// Sequence-indexed rows, appended per position.
    Kv(LayerKvRows),
    /// A fixed-size buffer the operator reads and rewrites in place.
    Recurrent(RecurrentState),
    Stateless,
}

/// One softmax layer's retained rows.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerKvRows {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl LayerKvRows {
    pub fn append(&mut self, key: Vec<f32>, value: Vec<f32>) {
        self.keys.push(key);
        self.values.push(value);
    }

    pub fn keys(&self) -> &[Vec<f32>] {
        &self.keys
    }

    pub fn values(&self) -> &[Vec<f32>] {
        &self.values
    }
}

/// A recurrence's durable buffer.
///
/// Carries its shape so a consumer can index it, and nothing else. It does
/// not know which operator owns it: the same type serves Gated DeltaNet
/// today and is meant to serve KDA unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentState {
    shape: Vec<usize>,
    cells: Vec<f32>,
}

impl RecurrentState {
    /// The zero start a sequence begins from.
    pub fn zeros(geometry: &RecurrentGeometry) -> Self {
        match geometry.initialization {
            StateInitialization::Zeros => Self {
                shape: geometry.shape.clone(),
                cells: vec![0.0; geometry.elements()],
            },
        }
    }

    /// Adopt existing cells — a captured state, or one being resumed.
    pub fn from_cells(geometry: &RecurrentGeometry, cells: Vec<f32>) -> Result<Self, String> {
        if cells.len() != geometry.elements() {
            return Err(format!(
                "state has {} cells, this geometry needs {}",
                cells.len(),
                geometry.elements()
            ));
        }
        Ok(Self {
            shape: geometry.shape.clone(),
            cells,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn cells(&self) -> &[f32] {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut [f32] {
        &mut self.cells
    }
}

/// A component's whole continuation state, one entry per layer.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuationState {
    layers: Vec<LayerContinuationState>,
    position: usize,
}

impl ContinuationState {
    /// Allocate exactly what each layer's geometry asks for — and nothing
    /// for the layers that ask for nothing.
    pub fn prepare(geometry: &[LayerContinuationGeometry]) -> Self {
        Self {
            layers: geometry
                .iter()
                .map(|g| match g {
                    LayerContinuationGeometry::Kv(_) => {
                        LayerContinuationState::Kv(LayerKvRows::default())
                    }
                    LayerContinuationGeometry::Recurrent(r) => {
                        LayerContinuationState::Recurrent(RecurrentState::zeros(r))
                    }
                    LayerContinuationGeometry::Stateless => LayerContinuationState::Stateless,
                })
                .collect(),
            position: 0,
        }
    }

    pub fn layer(&self, index: usize) -> &LayerContinuationState {
        &self.layers[index]
    }

    pub fn layer_mut(&mut self, index: usize) -> &mut LayerContinuationState {
        &mut self.layers[index]
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// The logical position this state continues from. Owned explicitly,
    /// never derived from a row count — the same contract
    /// [`KvState`](super::kv::KvState) states, and for the same reason: a
    /// recurrent layer retains no rows at all, so a count could not answer
    /// it even in principle.
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    /// Total elements retained after `positions` steps.
    pub fn elements_at(&self, positions: usize) -> usize {
        self.layers
            .iter()
            .map(|l| match l {
                LayerContinuationState::Kv(rows) => {
                    rows.keys.first().map_or(0, |r| r.len()) * 2 * positions
                }
                LayerContinuationState::Recurrent(r) => r.cells.len(),
                LayerContinuationState::Stateless => 0,
            })
            .sum()
    }
}

impl LayerContinuationState {
    pub fn kv_mut(&mut self) -> Option<&mut LayerKvRows> {
        match self {
            Self::Kv(rows) => Some(rows),
            _ => None,
        }
    }

    pub fn recurrent_mut(&mut self) -> Option<&mut RecurrentState> {
        match self {
            Self::Recurrent(state) => Some(state),
            _ => None,
        }
    }

    pub fn recurrent(&self) -> Option<&RecurrentState> {
        match self {
            Self::Recurrent(state) => Some(state),
            _ => None,
        }
    }
}
