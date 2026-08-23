//! `larql serve`'s model-reference resolution — the first "resolver
//! convergence" rung of the vindex3-registry initiative
//! (`docs/vindex3-registry-design.md` §8/§9).
//!
//! # The claimed/unclaimed boundary
//!
//! A bare name (`qwen3.8`, optionally `:variant`) the production
//! registry has claimed is resolved by the VINDEX3 registry
//! **exclusively** — any failure (unknown variant, incompatible ABI, a
//! malformed manifest) is a real refusal, never rescued by falling
//! through to [`cache::resolve_model`]'s legacy cache-shorthand lookup.
//! A name the registry has never claimed is not a failed VINDEX3
//! resolution being silently downgraded to a heuristic — it was never
//! the registry's to resolve in the first place, so today's
//! cache-shorthand behaviour (VINDEX2 and VINDEX3 mixed) keeps working
//! unchanged for it. This is deliberately checked as membership
//! (`registry.models.contains_key`), not by pattern-matching
//! `resolve()`'s `UnknownModel` error — the two would look almost
//! identical today, but only membership stays correct once a claimed
//! name can fail for other reasons (bad variant, incompatible ABI) that
//! must never fall through.
//!
//! An explicit `hf://`/local-path reference is untouched by this rung:
//! both already dispatch correctly on whichever generation they find
//! (`larql-server`'s `load_artifact` calls `detect_generation` itself),
//! so routing them through the new resolver's stricter explicit arms
//! here — which refuse a VINDEX2 local directory outright — would
//! regress existing VINDEX2 `serve` usage, not fix anything. Widening
//! this rung's scope to those forms is a later decision, not a side
//! effect of this one.
//!
//! # Why the production registry is empty
//!
//! No official VINDEX3 model has been published yet, and rung 1
//! explicitly left "where does registry data actually come from"
//! (embedded? a file? fetched?) undecided. Shipping this wiring now
//! against an empty manifest means zero behaviour change for any
//! current caller today, and the claimed/unclaimed split activates
//! itself correctly the moment a real entry is added — no second
//! migration required.
//!
//! # The other fix this rung makes
//!
//! `run_serve` used to do
//! `cache::resolve_model(path).unwrap_or_else(|_| path.clone())` —
//! silently substituting the raw, unresolved string on *any* resolution
//! failure (including an ambiguous shorthand with a perfectly good error
//! message) and handing it across the process boundary to
//! `larql-server`, which has no shorthand knowledge at all and fails
//! with a confusing IO error three layers down. This module propagates
//! the real error instead; nothing legitimate depended on the fallback,
//! since [`cache::resolve_model`]'s own "already a local directory"
//! branch already accepts a raw valid path.

use std::collections::BTreeMap;
use std::path::PathBuf;

use larql_vindex::registry::{
    resolve as resolve_vindex3, ArtifactRef, ModelReference, RegistryManifest, Vindex3Resolution,
    REGISTRY_MANIFEST_SCHEMA_VERSION,
};

use super::cache;

/// The production VINDEX3 registry. Empty until an official model is
/// published — see the module docs.
fn production_registry() -> RegistryManifest {
    RegistryManifest {
        schema_version: REGISTRY_MANIFEST_SCHEMA_VERSION,
        models: BTreeMap::new(),
    }
}

/// Resolve a `larql serve <path>` argument to a literal, already-fetched
/// local path — the string `run_serve` hands to the `larql-server`
/// subprocess.
pub fn resolve_serve_target(path: &str) -> Result<String, Box<dyn std::error::Error>> {
    resolve_serve_target_with(path, &production_registry(), cache::resolve_model, |hf| {
        Ok(larql_vindex::resolve_hf_vindex(hf)?)
    })
}

/// Testable core of [`resolve_serve_target`]. `legacy` and `fetch_hf` are
/// injected so the claimed/unclaimed dispatch can be proven without
/// touching `~/.cache` or the network.
fn resolve_serve_target_with(
    path: &str,
    registry: &RegistryManifest,
    legacy: impl FnOnce(&str) -> Result<PathBuf, Box<dyn std::error::Error>>,
    fetch_hf: impl FnOnce(&str) -> Result<PathBuf, Box<dyn std::error::Error>>,
) -> Result<String, Box<dyn std::error::Error>> {
    if let Ok(ModelReference::Registry { name, .. }) = ModelReference::parse(path) {
        if registry.models.contains_key(name.as_str()) {
            return resolve_claimed(path, registry, fetch_hf);
        }
    }
    Ok(legacy(path)?.display().to_string())
}

/// A name the registry has claimed: resolve it and materialise its
/// pinned artifact. Never falls back to `legacy` — a claimed name's
/// failures (unknown variant, incompatible ABI) are refusals, not
/// prompts to guess.
fn resolve_claimed(
    path: &str,
    registry: &RegistryManifest,
    fetch_hf: impl FnOnce(&str) -> Result<PathBuf, Box<dyn std::error::Error>>,
) -> Result<String, Box<dyn std::error::Error>> {
    let resolution = resolve_vindex3(path, registry)?;
    let Vindex3Resolution::Registry(resolved) = resolution else {
        unreachable!(
            "a name found in registry.models always resolves to Vindex3Resolution::Registry"
        )
    };
    // Registry artifacts are HuggingFace-only by schema — `RegistryArtifactRef`
    // has no local form (design doc §8: registry entries name a pinned HF
    // repo, never a local path) — so `ArtifactRef::Local` cannot arise here.
    let ArtifactRef::HuggingFace { repo, revision } = resolved.artifact else {
        unreachable!("a registry-resolved artifact is always ArtifactRef::HuggingFace")
    };
    let downloaded = fetch_hf(&format!("hf://{repo}@{revision}"))?;
    Ok(downloaded.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::registry::{
        Provenance, RegistryArtifactRef, RegistryModel, RegistryVariant, Vindex3Abi,
        CURRENT_VINDEX3_ABI,
    };

    fn unreachable_legacy(_: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        panic!("legacy resolver must not run for a claimed registry name")
    }

    fn unreachable_fetch(_: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        panic!("fetch_hf must not run when resolution fails before it")
    }

    fn registry_claiming_qwen38(abi: Vindex3Abi) -> RegistryManifest {
        let mut variants = BTreeMap::new();
        variants.insert(
            "27b-nvfp4".to_string(),
            RegistryVariant {
                artifact: RegistryArtifactRef {
                    repo: "larql/qwen3.8-27b-nvfp4".to_string(),
                    revision: "abc123f0".to_string(),
                },
                abi,
                source: Provenance {
                    repo: "Qwen/Qwen3.8-27B".to_string(),
                    revision: "8c4fdeadbeef".to_string(),
                },
            },
        );
        let mut models = BTreeMap::new();
        models.insert(
            "qwen3.8".to_string(),
            RegistryModel {
                default_variant: "27b-nvfp4".to_string(),
                variants,
            },
        );
        RegistryManifest {
            schema_version: REGISTRY_MANIFEST_SCHEMA_VERSION,
            models,
        }
    }

    // ── Unclaimed names: existing legacy behaviour, unchanged ────────────

    #[test]
    fn unclaimed_name_falls_through_to_legacy_and_returns_its_result() {
        let registry = production_registry();
        let out = resolve_serve_target_with(
            "some-local-alias",
            &registry,
            |_| Ok(PathBuf::from("/fake/legacy/path")),
            unreachable_fetch,
        )
        .unwrap();
        assert_eq!(out, "/fake/legacy/path");
    }

    #[test]
    fn unclaimed_name_propagates_legacy_error_instead_of_silently_falling_back() {
        // The bug this rung fixes: a legacy resolution failure must
        // surface, never be swallowed into the raw unresolved string.
        let registry = production_registry();
        let err = resolve_serve_target_with(
            "ambiguous-name",
            &registry,
            |_| Err("shorthand `ambiguous-name` is ambiguous".into()),
            unreachable_fetch,
        )
        .unwrap_err();
        assert!(err.to_string().contains("ambiguous"), "{err}");
    }

    // ── Claimed names: registry-exclusive, no fallback on any failure ───

    #[test]
    fn claimed_name_wins_even_when_legacy_would_have_succeeded() {
        let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
        let out = resolve_serve_target_with("qwen3.8", &registry, unreachable_legacy, |hf| {
            assert_eq!(hf, "hf://larql/qwen3.8-27b-nvfp4@abc123f0");
            Ok(PathBuf::from("/resolved/hf/path"))
        })
        .unwrap();
        assert_eq!(out, "/resolved/hf/path");
    }

    #[test]
    fn claimed_name_with_unknown_variant_hard_errors_without_touching_legacy() {
        let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
        let err = resolve_serve_target_with(
            "qwen3.8:does-not-exist",
            &registry,
            unreachable_legacy,
            unreachable_fetch,
        )
        .unwrap_err();
        assert!(err.to_string().contains("does-not-exist"), "{err}");
    }

    #[test]
    fn claimed_name_with_incompatible_abi_hard_errors_without_touching_legacy() {
        let registry = registry_claiming_qwen38(Vindex3Abi(CURRENT_VINDEX3_ABI.get() + 1));
        let err =
            resolve_serve_target_with("qwen3.8", &registry, unreachable_legacy, unreachable_fetch)
                .unwrap_err();
        assert!(err.to_string().contains("ABI"), "{err}");
    }

    // ── Explicit forms and malformed input bypass the claim check ───────

    #[test]
    fn explicit_hf_reference_bypasses_the_claim_check() {
        let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
        let out = resolve_serve_target_with(
            "hf://owner/repo",
            &registry,
            |_| Ok(PathBuf::from("/legacy/hf/resolved")),
            unreachable_fetch,
        )
        .unwrap();
        assert_eq!(out, "/legacy/hf/resolved");
    }

    #[test]
    fn explicit_local_reference_bypasses_the_claim_check() {
        let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
        let out = resolve_serve_target_with(
            "/some/local/path",
            &registry,
            |_| Ok(PathBuf::from("/some/local/path")),
            unreachable_fetch,
        )
        .unwrap();
        assert_eq!(out, "/some/local/path");
    }

    #[test]
    fn malformed_reference_falls_through_to_legacy() {
        let registry = registry_claiming_qwen38(CURRENT_VINDEX3_ABI);
        let out = resolve_serve_target_with(
            "qwen3.8:",
            &registry,
            |_| Ok(PathBuf::from("/legacy/fallback")),
            unreachable_fetch,
        )
        .unwrap();
        assert_eq!(out, "/legacy/fallback");
    }

    #[test]
    fn production_registry_is_empty_today() {
        assert!(production_registry().models.is_empty());
    }

    // ── The real, non-injected wrapper ────────────────────────────────

    #[test]
    fn resolve_serve_target_resolves_an_explicit_local_directory() {
        // Exercises the real `resolve_serve_target` (production registry +
        // real `cache::resolve_model`) end to end, hermetically: an
        // existing directory is `cache::resolve_model`'s own "already a
        // local directory" branch, so this never touches `~/.cache` or
        // the network.
        let dir = tempfile::tempdir().unwrap();
        let out = resolve_serve_target(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(out, dir.path().display().to_string());
    }
}
