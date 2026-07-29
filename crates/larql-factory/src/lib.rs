//! Vindex Factory driver — recipe schema, `build_id` canonicaliser, and
//! structural validator. See `docs/vindex-factory.md` §3.1: this crate
//! is the single implementation both the GitHub Action and the rig
//! worker call (as `larql recipe validate` / `larql recipe build-id`),
//! so there's nothing to keep in sync between them.
//!
//! Scope as of this crate's first cut: §4's recipe schema, §5's
//! `build_id`, and the structural half of §6.1's PR-check gate. The
//! build-stage driver (§7), card generator, and verify-from-hub harness
//! are not yet implemented here.

#![deny(missing_docs)]

mod build_id;
mod capabilities;
mod card;
mod constants;
mod estimate;
mod hex;
mod recipe;
mod validate;

pub use build_id::build_id;
pub use capabilities::{
    manifest as capabilities_manifest, ArchitectureCapability, CapabilityManifest,
};
pub use card::{
    render as render_card, revision_tag, CardInputs, LogitMatchResult, ReconstructionResult,
    SliceSummary, VerificationReport,
};
pub use estimate::{
    estimate as estimate_size, ExecutorClass, HttpError as EstimateError, ModelDims,
    OutputEstimate, SizeEstimate,
};
pub use recipe::{
    Budget, BudgetRequires, Extractor, HubPublish, LogitMatch, Metadata, MirrorPublish, OutputSpec,
    Publish, Recipe, Reconstruction, Source, Spec, Verify, API_VERSION, KIND,
};
pub use validate::{validate, RecipeError};
