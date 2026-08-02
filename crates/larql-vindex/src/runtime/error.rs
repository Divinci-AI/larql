//! Execution failures.
//!
//! Binding is where operands are chosen, checked and refused; by the time
//! `execute` runs, every question of *which* bytes and *whether they fit* has
//! been answered. So these variants are not the normal diagnostic surface —
//! they are the assertions that keep a binding bug from being interpreted as
//! numerics.
//!
//! Each one therefore names the operand and both sides of the disagreement. An
//! execution error that says only "dimension mismatch" sends the reader back
//! through the whole load path; one that says which tensor, which axis, and
//! what the two values were points at the binding decision that produced it.

use crate::format::lyrw2::region_format::RegionFormat;

use super::axis::Axis;

/// Why an execution could not run to completion.
///
/// `PartialEq` without `Eq`: `NonFiniteRouterScore` carries the offending
/// value, and reflexivity genuinely fails for the NaN it most often holds.
/// Claiming `Eq` here would be claiming a property this type does not have.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ExecutionError {
    /// The reference decoder has no implementation for this encoding.
    ///
    /// Not a defect in the index: the reference path deliberately implements
    /// the directly-readable encodings only, and a quantised region is a
    /// missing kernel rather than bad bytes.
    #[error("the reference decoder does not implement {format} (needed for {operand})")]
    UnsupportedFormat { format: String, operand: String },

    /// The reference decoder cannot serve this access pattern.
    #[error("the reference decoder does not implement a {view} view (needed for {operand})")]
    UnsupportedView { view: String, operand: String },

    /// An operand's shape disagrees with what the operation requires.
    #[error("{operand}: expected {axis} of {expected}, found {found}")]
    DimensionMismatch {
        operand: String,
        axis: Axis,
        expected: usize,
        found: usize,
    },

    /// A region is shorter than its declared shape needs.
    #[error("{operand}: shape needs {needed} bytes, region holds {found}")]
    ShortRegion {
        operand: String,
        needed: usize,
        found: usize,
    },

    /// A row index fell outside the tensor.
    #[error("{operand}: row {row} is out of range for {rows} rows")]
    RowOutOfRange {
        operand: String,
        row: usize,
        rows: usize,
    },

    /// A matrix operand turned out not to be a matrix.
    #[error("{operand}: expected a matrix, found {found}")]
    NotAMatrix { operand: String, found: String },

    /// A router score is not a finite number.
    ///
    /// Refused rather than ordered. Under a total order a NaN still lands
    /// *somewhere*, so it would be selected or rejected by sort mechanics —
    /// a routing decision nobody made, reached silently, and propagated into
    /// the residual stream as a plausible token.
    #[error("router score for expert {expert} is {value}, which is not a finite number")]
    NonFiniteRouterScore { expert: usize, value: f32 },

    /// An expert id outside the router's address space.
    ///
    /// A catalogue fault: nothing could ever select this expert, so the router
    /// and the bank disagree about which population they are describing.
    #[error("expert {expert} is outside the router's addressable population of {population}")]
    ExpertOutOfRange { expert: u32, population: usize },

    /// Routing selected an expert this bank does not hold.
    ///
    /// **Not** a catalogue fault. The routing decision is correct and the
    /// expert exists in the model; it is simply not resident here, which is
    /// the normal condition for a shard. Distinct from
    /// [`Self::ExpertOutOfRange`] because the repairs differ completely: that
    /// one means the index is wrong, this one means the operand must be
    /// fetched from wherever it lives.
    ///
    /// It must never degrade into skipping the expert, renormalising the
    /// surviving weights, or substituting a neighbour. Each of those produces
    /// a token missing part of its FFN contribution and looks entirely
    /// reasonable. Remote placement will later satisfy this request without
    /// changing router semantics — which is only possible if the request
    /// reaches the caller intact.
    #[error(
        "routing selected expert {expert}, which is not resident in {bank} \
         (this bank holds {resident} of {population} experts)"
    )]
    SelectedExpertNotResident {
        expert: u32,
        bank: String,
        resident: usize,
        population: usize,
    },
}

impl ExecutionError {
    /// Construct an unsupported-format error from a region encoding.
    pub fn unsupported_format(format: RegionFormat, operand: impl Into<String>) -> Self {
        Self::UnsupportedFormat {
            format: format.name(),
            operand: operand.into(),
        }
    }
}
