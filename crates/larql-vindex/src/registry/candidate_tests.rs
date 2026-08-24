//! Colocated tests for [`super::candidate`] — R3C's core invariant
//! ("candidate generation must never invent information") and its
//! reuse of the real registry schema for validation.

use super::abi::{Vindex3Abi, CURRENT_VINDEX3_ABI};
use super::candidate::{build_candidate, CandidateInputs};
use super::error::RegistryError;
use super::manifest::Attestation;

fn valid_inputs() -> CandidateInputs {
    CandidateInputs {
        name: "granite-4.1-3b".to_string(),
        variant: "bf16".to_string(),
        artifact_repo: "larql/granite-4.1-3b".to_string(),
        artifact_revision: "1048a8eb2fec5812a698e76d7e603527d0475c17".to_string(),
        abi: None,
        source_repo: "ibm-granite/granite-4.1-3b".to_string(),
        source_revision: "c0650403e44e78ec0262dab1c90914c65b196c4e".to_string(),
        attested_by: "chrishayuk".to_string(),
    }
}

#[test]
fn a_well_formed_candidate_builds_and_validates() {
    let model = build_candidate(valid_inputs()).unwrap();
    assert_eq!(model.default_variant, "bf16");
    let variant = model.variants.get("bf16").unwrap();
    assert_eq!(variant.artifact.repo, "larql/granite-4.1-3b");
    assert_eq!(
        variant.artifact.revision,
        "1048a8eb2fec5812a698e76d7e603527d0475c17"
    );
    assert_eq!(variant.source.repo, "ibm-granite/granite-4.1-3b");
    assert_eq!(
        variant.source.revision,
        "c0650403e44e78ec0262dab1c90914c65b196c4e"
    );
    assert_eq!(
        variant.source.attestation,
        Attestation::HandAttested {
            by: "chrishayuk".to_string()
        }
    );
}

#[test]
fn abi_defaults_to_the_current_abi_when_not_overridden() {
    let model = build_candidate(valid_inputs()).unwrap();
    let variant = model.variants.get("bf16").unwrap();
    assert_eq!(variant.abi, CURRENT_VINDEX3_ABI);
}

#[test]
fn an_explicit_abi_override_is_used_verbatim() {
    let mut inputs = valid_inputs();
    inputs.abi = Some(Vindex3Abi(7));
    let model = build_candidate(inputs).unwrap();
    assert_eq!(model.variants.get("bf16").unwrap().abi, Vindex3Abi(7));
}

#[test]
fn a_floating_artifact_revision_refuses_through_the_real_schema() {
    // Proves build_candidate actually validates, not just assembles —
    // a floating artifact.revision must be refused the same way a
    // hand-written registry/models/*.json would be.
    let mut inputs = valid_inputs();
    inputs.artifact_revision = "main".to_string();
    let err = build_candidate(inputs).unwrap_err();
    assert!(matches!(err, RegistryError::UnpinnedRevision { .. }));
}

#[test]
fn a_floating_source_revision_refuses() {
    let mut inputs = valid_inputs();
    inputs.source_revision = "latest".to_string();
    let err = build_candidate(inputs).unwrap_err();
    assert!(matches!(err, RegistryError::UnpinnedRevision { .. }));
}

#[test]
fn an_empty_attested_by_refuses() {
    // The only attestation path R3C offers is HandAttested — an empty
    // `by` must refuse exactly the way a hand-written registry entry
    // would, not silently accept a hand-attestation naming no one.
    let mut inputs = valid_inputs();
    inputs.attested_by = String::new();
    let err = build_candidate(inputs).unwrap_err();
    assert!(matches!(err, RegistryError::EmptyAttestationBy { .. }));
}

#[test]
fn the_candidate_never_carries_its_own_name_field() {
    // The model body is exactly what would live at
    // registry/models/<name>.json — the name lives in the index and
    // the filename, never duplicated inside the model's own JSON
    // (same choice check.rs's own docs make).
    let model = build_candidate(valid_inputs()).unwrap();
    let json = serde_json::to_value(&model).unwrap();
    assert!(
        json.get("name").is_none(),
        "RegistryModel must not serialise a name field: {json}"
    );
}
