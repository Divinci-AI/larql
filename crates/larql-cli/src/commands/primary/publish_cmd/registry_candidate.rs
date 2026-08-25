//! Registry candidate emission (R3C) — the `--emit-registry-candidate`
//! flag family on `larql publish`.
//!
//! Plain `larql publish` (the flag omitted) is untouched by this
//! module: [`parse_candidate_request`] returns `Ok(None)` immediately,
//! and [`super::run`] never calls [`emit`] at all in that case.

use std::path::PathBuf;

use larql_vindex::registry::{build_candidate, CandidateInputs, Vindex3Abi};

use super::upload::StepOutcome;
use super::PublishArgs;

/// Every argument `--emit-registry-candidate` needs, checked up front
/// — before any upload runs — so a caller with a missing companion
/// flag finds out immediately, not after paying for a real publish.
#[derive(Debug)]
pub(super) struct CandidateRequest {
    name: String,
    variant: String,
    source_repo: String,
    source_revision: String,
    attested_by: String,
    abi: Option<Vindex3Abi>,
    out: Option<PathBuf>,
}

/// Parse and validate `args`' registry-candidate flags.
///
/// `Ok(None)` — `--emit-registry-candidate` wasn't passed, nothing to
/// do. `Ok(Some(_))` — every required companion flag is present.
/// `Err` — the flag combination is invalid (a required flag missing,
/// or paired with `--no-full`), named clearly enough to fix without
/// re-reading `--help`.
pub(super) fn parse_candidate_request(
    args: &PublishArgs,
) -> Result<Option<CandidateRequest>, Box<dyn std::error::Error>> {
    if !args.emit_registry_candidate {
        return Ok(None);
    }
    if args.no_full {
        return Err(
            "--emit-registry-candidate requires publishing the full artifact \
             (--no-full names no full artifact to build a candidate for)"
                .into(),
        );
    }
    Ok(Some(CandidateRequest {
        name: require(&args.registry_name, "--registry-name")?,
        variant: require(&args.registry_variant, "--registry-variant")?,
        source_repo: require(&args.source_repo, "--source-repo")?,
        source_revision: require(&args.source_revision, "--source-revision")?,
        attested_by: require(&args.attested_by, "--attested-by")?,
        abi: args.registry_abi.map(Vindex3Abi),
        out: args.registry_candidate_out.clone(),
    }))
}

fn require(value: &Option<String>, flag: &str) -> Result<String, Box<dyn std::error::Error>> {
    value
        .clone()
        .ok_or_else(|| format!("--emit-registry-candidate requires {flag}").into())
}

/// Build the candidate from the just-completed publish's own `results`
/// — the `full` step's real, pinned `{repo, revision}`, never
/// re-derived or guessed here — validate it through the real registry
/// schema, then print or write it. A candidate that fails to build is
/// a real error: a successful artifact publish with a broken candidate
/// request is reported, never silently dropped.
pub(super) fn emit(
    request: CandidateRequest,
    results: &[StepOutcome],
) -> Result<(), Box<dyn std::error::Error>> {
    let full = results.iter().find(|r| r.label == "full").ok_or(
        "--emit-registry-candidate found no `full` publish step to build a candidate from",
    )?;

    let model = build_candidate(CandidateInputs {
        name: request.name.clone(),
        variant: request.variant,
        artifact_repo: full.repo.clone(),
        artifact_revision: full.revision.clone(),
        abi: request.abi,
        source_repo: request.source_repo,
        source_revision: request.source_revision,
        attested_by: request.attested_by,
    })?;

    let json = serde_json::to_string_pretty(&model)?;
    match request.out {
        Some(path) => {
            std::fs::write(&path, format!("{json}\n"))?;
            println!("\nRegistry candidate:\n  {}", path.display());
        }
        None => {
            println!("\nRegistry candidate ({}):", request.name);
            println!("{json}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> PublishArgs {
        PublishArgs {
            source: "src".to_string(),
            repo: "larql/example".to_string(),
            full: true,
            no_full: false,
            slices: Vec::new(),
            slice_repo_template: "{repo}-{preset}".to_string(),
            tmp_dir: None,
            dry_run: false,
            collections: Vec::new(),
            model_title: None,
            family: None,
            library_title: "LARQL Vindex Library".to_string(),
            force_upload: false,
            no_prune: false,
            repo_type: "model".to_string(),
            private: false,
            emit_registry_candidate: false,
            registry_name: None,
            registry_variant: None,
            source_repo: None,
            source_revision: None,
            attested_by: None,
            registry_abi: None,
            registry_candidate_out: None,
        }
    }

    #[test]
    fn the_flag_omitted_returns_none_and_requires_nothing() {
        let args = base_args();
        assert!(parse_candidate_request(&args).unwrap().is_none());
    }

    #[test]
    fn the_flag_alone_with_no_companions_errors_naming_the_first_missing_one() {
        let mut args = base_args();
        args.emit_registry_candidate = true;
        let err = parse_candidate_request(&args).unwrap_err();
        assert!(err.to_string().contains("--registry-name"));
    }

    #[test]
    fn every_companion_flag_present_parses() {
        let mut args = base_args();
        args.emit_registry_candidate = true;
        args.registry_name = Some("granite-4.1-3b".to_string());
        args.registry_variant = Some("bf16".to_string());
        args.source_repo = Some("ibm-granite/granite-4.1-3b".to_string());
        args.source_revision = Some("c0650403".to_string());
        args.attested_by = Some("chrishayuk".to_string());

        let request = parse_candidate_request(&args).unwrap().unwrap();
        assert_eq!(request.name, "granite-4.1-3b");
        assert_eq!(request.variant, "bf16");
    }

    #[test]
    fn combined_with_no_full_refuses() {
        let mut args = base_args();
        args.emit_registry_candidate = true;
        args.no_full = true;
        let err = parse_candidate_request(&args).unwrap_err();
        assert!(err.to_string().contains("--no-full"));
    }

    fn valid_request() -> CandidateRequest {
        CandidateRequest {
            name: "granite-4.1-3b".to_string(),
            variant: "bf16".to_string(),
            source_repo: "ibm-granite/granite-4.1-3b".to_string(),
            source_revision: "c0650403e44e78ec0262dab1c90914c65b196c4e".to_string(),
            attested_by: "chrishayuk".to_string(),
            abi: None,
            out: None,
        }
    }

    fn full_step_outcome() -> StepOutcome {
        StepOutcome {
            label: "full".to_string(),
            repo: "larql/granite-4.1-3b".to_string(),
            url: "https://huggingface.co/larql/granite-4.1-3b".to_string(),
            revision: "1048a8eb2fec5812a698e76d7e603527d0475c17".to_string(),
        }
    }

    #[test]
    fn emit_with_no_full_step_in_results_errors() {
        let err = emit(valid_request(), &[]).unwrap_err();
        assert!(err.to_string().contains("full"));
    }

    #[test]
    fn emit_writes_the_pinned_artifact_revision_from_the_real_publish_result() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("candidate.json");
        let mut request = valid_request();
        request.out = Some(out.clone());

        emit(request, std::slice::from_ref(&full_step_outcome())).unwrap();

        let text = std::fs::read_to_string(&out).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            json["variants"]["bf16"]["artifact"]["revision"],
            "1048a8eb2fec5812a698e76d7e603527d0475c17"
        );
        assert_eq!(
            json["variants"]["bf16"]["artifact"]["repo"],
            "larql/granite-4.1-3b"
        );
        assert!(json.get("name").is_none());
    }

    #[test]
    fn emit_propagates_a_candidate_that_fails_real_schema_validation() {
        let mut request = valid_request();
        request.source_revision = "main".to_string();
        let err = emit(request, &[full_step_outcome()]).unwrap_err();
        assert!(err.to_string().contains("immutable pin"));
    }
}
