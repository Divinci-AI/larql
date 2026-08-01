//! Capability derivation over a resolved index (spec §10, §11).
//!
//! One traversal, many consumers. Authority derivation, operation admission
//! and kernel binding are all *readers* of the report this module produces —
//! not three systems independently inspecting the index. Three inspectors is
//! how permissive logic gets in: each one is individually reasonable, and the
//! union of their leniencies is what actually ships.
//!
//! # Pipeline position
//!
//! ```text
//! raw profile
//!     ↓  inheritance / schema resolution
//! resolved requested selections
//!     ↓  physical variant and segment resolution
//! concrete region selection            ← everything above is INPUT here
//!     ↓  programme traversal
//! capability report                    ← this module
//!     ↓
//! authority derivation · operation admission · kernel binding
//! ```
//!
//! The split above the line matters. "Does the profile inherit from `exact`?"
//! and "is this variant physically present?" are answered *before* traversal;
//! "can this resolved selection decode, browse, or claim source-exact?" is
//! answered *from* it. Letting traversal answer the first pair would make the
//! derivation circular — a profile whose validity depended on capabilities
//! derived from that same profile.

pub mod authority;
pub mod compatibility;
pub mod coordinate;
pub mod kernel;
pub mod operation;
#[cfg(test)]
mod operation_tests;
pub mod plan;
#[cfg(test)]
mod plan_tests;
pub mod reconstruction;
#[cfg(test)]
mod reconstruction_tests;
pub mod role;
pub mod scope;
pub mod selection;
pub mod traversal;
#[cfg(test)]
mod traversal_tests;
pub mod walk_request;

pub use authority::{
    derive_authority, AuthorityInputs, DerivedAuthority, Fidelity, StructuralChange,
};
pub use compatibility::{IncompatibilityShape, RoleSegmentCoverage, SegmentCompatibility};
pub use coordinate::{AbsenceKind, RegionCoordinate};
pub use kernel::{KernelId, KernelMaturity};
pub use operation::{Degradation, OperationAdmission, OperationCapability, OperationFailure};
pub use plan::{
    OperationPlan, PlanChoice, PlannedRegion, QualifiedAlternative, QualifiedOperationRoute,
};
pub use reconstruction::{
    CanonicalSelection, CatalogueVariant, ReconstructionFailure, RegionSetCatalogue,
    TensorReconstruction,
};
pub use role::{OperandAvailability, ReferenceSupport, RoleCapability};
pub use scope::{CapabilityReport, DocumentCapabilities, ProfileCapabilities};
pub use selection::{BankSelection, SelectedRegion};
pub use traversal::{
    traverse_bank, AlternativeReport, BankCapabilityReport, ReferenceCodecs, ReferenceExecution,
};
pub use walk_request::{
    validate_query_vector, PartialPolicy, QueryVectorFault, WalkInput, WalkRequest, WalkTarget,
};
