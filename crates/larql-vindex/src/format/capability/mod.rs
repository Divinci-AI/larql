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
pub mod coordinate;
pub mod operand;
pub mod selection;

pub use authority::{
    derive_authority, AuthorityInputs, DerivedAuthority, Fidelity, StructuralChange,
};
pub use coordinate::{AbsenceKind, RegionCoordinate};
pub use operand::{KernelId, KernelMaturity, OperandCapability};
pub use selection::{BankSelection, SelectedRegion};
