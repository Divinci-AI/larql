//! Turn a successful `larql publish` into a deterministic candidate
//! registry record (R3C).
//!
//! # Scope, deliberately narrow
//!
//! This is `publish` (artifact exists) emitting a *proposal* for what
//! `registry/models/<name>.json` could say — it never writes into
//! `registry/models/`, never touches `registry/index.json`, and never
//! commits anything. That split (publish vs. promote) is the design's
//! own R3D boundary (`docs/vindex3-registry-design.md` §6): promotion —
//! re-fetching the pinned artifact, re-validating it, running whatever
//! gates exist, and actually writing the registry entry — is separate,
//! later work.
//!
//! # The one invariant that matters here
//!
//! **Candidate generation must never invent information.** Every field
//! is either mechanically known from the publish that just happened
//! (`artifact.repo`/`artifact.revision`, straight from
//! [`super::manifest::RegistryArtifactRef`]'s source of truth — the real
//! `PublishResult` a caller already has, not re-derived here), an
//! explicit policy default this binary can state about itself (the ABI
//! it implements), or an explicit value the caller supplied
//! (`source.repo`, `source.revision`, who's attesting to them). Nothing
//! here guesses a source checkpoint from a filename, an HF cache path,
//! or the published repo's own name — a caller with no source
//! information gets a refusal, not a plausible-looking fabrication.
//!
//! # Why this reuses `RegistryManifest::validate()`, not a lighter check
//!
//! [`build_candidate`] wraps its one model into a one-entry
//! [`RegistryManifest`] and calls `.validate()` on it before returning
//! anything — the exact code `production_registry()` trusts and
//! `registry check` (R3B) calls out to. A candidate that would fail
//! real registry validation (a floating revision, an empty attestation)
//! is refused here, at generation time, not discovered later by
//! whatever eventually runs `registry check` on it.

use std::collections::BTreeMap;

use super::abi::Vindex3Abi;
use super::error::RegistryError;
use super::manifest::{
    Attestation, Provenance, RegistryArtifactRef, RegistryManifest, RegistryModel, RegistryVariant,
    REGISTRY_MANIFEST_SCHEMA_VERSION,
};

/// Every fact a candidate needs, named by where it came from — see the
/// module docs' invariant. No field here has a silent default except
/// [`CandidateInputs::abi`], which explicitly documents what its
/// default means.
pub struct CandidateInputs {
    /// The registry model name this candidate is *for* — not embedded
    /// in the returned [`RegistryModel`] itself (that type has no name
    /// field, by the same single-source-of-truth choice `check.rs`
    /// already makes: a model's name lives in `registry/index.json` and
    /// its filename, never duplicated inside the model's own JSON), but
    /// callers need it to know what to name the file/print in a
    /// summary.
    pub name: String,
    /// The variant name this candidate selects as its (only, so far)
    /// default variant.
    pub variant: String,
    /// The published VINDEX3 container's own HF repo — mechanically
    /// known from the publish that just happened.
    pub artifact_repo: String,
    /// The published container's pinned HF revision — mechanically
    /// known (`PublishResult::revision`, `fetch_repo_head_sha`).
    pub artifact_revision: String,
    /// The VINDEX3 runtime ABI this candidate declares. Defaults to
    /// [`super::abi::CURRENT_VINDEX3_ABI`] when `None` — not a guess:
    /// it's the one ABI this binary actually implements, the only
    /// value that could be correct without inventing a compatibility
    /// range no second ABI value exists to justify yet (see
    /// `abi.rs`'s own module docs).
    pub abi: Option<Vindex3Abi>,
    /// Upstream checkpoint repo the published artifact was built from.
    /// Explicit input only — R3C never derives this from the published
    /// repo's name, a filename, or an HF cache path.
    pub source_repo: String,
    /// Upstream checkpoint's pinned revision. Explicit input only, same
    /// reasoning as `source_repo`.
    pub source_revision: String,
    /// Who is vouching for `source_repo`/`source_revision`. This rung's
    /// only attestation path is [`Attestation::HandAttested`] — no
    /// mechanical capture exists yet (`encode` takes no source
    /// parameter at all, the same gap `docs/vindex3-registry-design.md`
    /// §4 and the publishing design's Q2/Q3 already named).
    pub attested_by: String,
}

/// Build and validate a one-model, one-variant registry candidate.
///
/// Returns the [`RegistryModel`] body exactly as it would appear at
/// `registry/models/<name>.json` — a caller serialises it directly,
/// with `inputs.name` (not embedded in the return value) driving the
/// filename or a printed summary.
pub fn build_candidate(inputs: CandidateInputs) -> Result<RegistryModel, RegistryError> {
    let variant = RegistryVariant {
        artifact: RegistryArtifactRef {
            repo: inputs.artifact_repo,
            revision: inputs.artifact_revision,
        },
        abi: inputs.abi.unwrap_or(super::abi::CURRENT_VINDEX3_ABI),
        source: Provenance {
            repo: inputs.source_repo,
            revision: inputs.source_revision,
            attestation: Attestation::HandAttested {
                by: inputs.attested_by,
            },
        },
    };

    let mut variants = BTreeMap::new();
    variants.insert(inputs.variant.clone(), variant);
    let model = RegistryModel {
        default_variant: inputs.variant,
        variants,
    };

    // Validate through the real schema, wrapped as the one-entry
    // manifest it would join — never a lighter, candidate-only check
    // that could accept something `registry check` would later refuse.
    let mut models = BTreeMap::new();
    models.insert(inputs.name, model.clone());
    let manifest = RegistryManifest {
        schema_version: REGISTRY_MANIFEST_SCHEMA_VERSION,
        models,
    };
    manifest.validate()?;

    Ok(model)
}
