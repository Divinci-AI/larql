//! The production VINDEX3 registry, and the shared claimed/unclaimed
//! dispatch every convergence caller uses (`docs/vindex3-registry-design.md`
//! §10).
//!
//! # Why this lives here, not per-caller
//!
//! Rung 2A first built this dispatch inside `larql-cli`'s `serve`
//! trampoline alone. Grounding rung 2B in `larql-server`'s actual code
//! found the same question asked from three more places — the server
//! binary's own CLI arg, its `--dir` bulk loader, and the
//! `/v1/runtime/model` HTTP lifecycle endpoint — none of which go
//! through the CLI trampoline at all. Two independent copies of
//! "is this name claimed" is exactly the kind of divergence the
//! initiative exists to remove (`qwen3.8` must mean the same VINDEX3
//! identity everywhere), so the dispatch moved here, into the one crate
//! every caller already depends on.
//!
//! # The production registry is empty
//!
//! No official VINDEX3 model has been published yet, and where real
//! registry data will actually come from (embedded? a file? fetched?)
//! remains a separate, undecided question — out of scope for the
//! resolver-convergence rung. Returning an empty manifest here means
//! zero behaviour change for any caller today; the claimed/unclaimed
//! split (see [`resolve_claimed`]) activates itself correctly the
//! moment a real entry is added, with no second migration required.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::error::RegistryError;
use super::manifest::{RegistryManifest, REGISTRY_MANIFEST_SCHEMA_VERSION};
use super::reference::ModelReference;
use super::resolver::lookup_claimed_variant;
use crate::VindexError;

/// The production VINDEX3 registry. Empty until an official model is
/// published — see the module docs.
pub fn production_registry() -> RegistryManifest {
    RegistryManifest {
        schema_version: REGISTRY_MANIFEST_SCHEMA_VERSION,
        models: BTreeMap::new(),
    }
}

/// The claimed/unclaimed boundary every convergence caller dispatches on:
/// `Ok(None)` — `raw` is not a name `registry` has claimed (not a bare
/// registry-shaped reference at all, or a bare name the registry has
/// never heard of) — the caller should fall through to its own existing
/// resolution. `Ok(Some(path))` — `raw` names a claimed model/variant,
/// resolved and materialised (its pinned Hugging Face artifact
/// downloaded via [`crate::format::huggingface::resolve_hf_vindex`] if
/// not already cached). `Err` — `raw` names a **claimed** model, but
/// resolution failed (unknown variant, incompatible ABI, a malformed
/// manifest): a real refusal. **The caller must never turn this into a
/// fallback to its own legacy resolution** — that would silently
/// downgrade a real registry failure into a guess, exactly the pattern
/// the convergence rung forbids (design doc §10.1).
///
/// Checked as registry membership (`registry.models.contains_key`), not
/// by pattern-matching [`RegistryError::UnknownModel`] out of the
/// resolution result — the two look identical today, but only
/// membership stays correct once a claimed name can fail for reasons
/// other than being absent.
pub fn resolve_claimed(
    raw: &str,
    registry: &RegistryManifest,
) -> Result<Option<PathBuf>, RegistryError> {
    // A bare function reference, not a closure literal wrapping it: the
    // latter would be its own never-covered MIR region (the fetch never
    // actually runs in a unit test — that would mean touching HF for
    // real), which a plain fn-item reference has no separate body to
    // measure at all.
    resolve_claimed_with(raw, registry, crate::format::huggingface::resolve_hf_vindex)
}

/// Testable core of [`resolve_claimed`]. `fetch_hf` is injected so
/// callers (including this module's own tests) can prove the
/// claimed/unclaimed contract without ever touching the network.
pub fn resolve_claimed_with(
    raw: &str,
    registry: &RegistryManifest,
    fetch_hf: impl FnOnce(&str) -> Result<PathBuf, VindexError>,
) -> Result<Option<PathBuf>, RegistryError> {
    let Ok(ModelReference::Registry { name, variant }) = ModelReference::parse(raw) else {
        return Ok(None);
    };
    if !registry.models.contains_key(name.as_str()) {
        return Ok(None);
    }
    // `RegistryArtifactRef` is a plain `{repo, revision}` struct, not the
    // `ArtifactRef` enum `resolve_registry` wraps it into for the public
    // API — reusing the shared lookup directly means no enum variant this
    // caller can't reach needs to be matched (and defended against) here.
    let (entry, _variant_name) = lookup_claimed_variant(&name, variant.as_ref(), registry)?;
    let path = fetch_hf(&format!(
        "hf://{}@{}",
        entry.artifact.repo, entry.artifact.revision
    ))?;
    Ok(Some(path))
}
