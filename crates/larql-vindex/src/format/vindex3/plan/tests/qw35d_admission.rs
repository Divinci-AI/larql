//! QW-3.5D: capability relevance as a judgement separate from semantic
//! classification.
//!
//! The rule this file exists to pin:
//!
//! > Unknown semantics fail closed **unless** there is an explicit,
//! > evidence-backed capability-relevance judgement proving they lie
//! > outside that capability's execution closure.
//!
//! `output_gate_type` keeps its class and its `Unrepresented` carriage —
//! nothing here reclassifies a fact to make a number go green. What
//! changes is only whether `TextGeneration`'s closure reaches it, and that
//! is decided by a predicate over the BUILT GRAPH, so the exclusion is
//! conditional on the evidence being present in the container in front of
//! us.
//!
//! Two falsifiers keep this from collapsing into "unread keys stopped
//! mattering":
//!
//! * [`an_undisposed_unknown_text_key_still_blocks_text`] — a sibling key
//!   with no disposition still blocks;
//! * [`text_admission_comes_from_the_operator_not_from_excluding_the_key`]
//!   — break the gate GEOMETRY while leaving `output_gate_type` untouched,
//!   and text must refuse.

use super::support::glimmer_shaped_target_with;
use crate::format::vindex3::plan::capability::Capability;
use crate::format::vindex3::plan::{plan_system, SystemPlan};

/// The Qwen3.8-shaped text config: a fused output gate plus the two
/// unowned metadata keys.
fn plan_with(edit: impl FnOnce(&mut serde_json::Value)) -> SystemPlan {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        config["text_config"]["attn_output_gate"] = serde_json::json!(true);
        config["text_config"]["output_gate_type"] = serde_json::json!("swish");
        config["language_model_only"] = serde_json::json!(false);
        edit(config);
    });
    plan_system(&[("target-artifact".to_string(), inventory)])
}

fn text_blocking(plan: &SystemPlan) -> usize {
    plan.capabilities
        .iter()
        .find(|c| c.capability == Capability::TextGeneration)
        .expect("a text-generation verdict")
        .blocking
}

/// **D3.** With the gate represented, `output_gate_type` leaves the text
/// closure — while remaining a blocking, unresolved whole-model finding.
///
/// Both halves are asserted. Dropping either would let this pass on a
/// build that had simply stopped grading the key.
#[test]
fn an_unowned_gate_type_leaves_the_text_closure_but_not_the_census() {
    let plan = plan_with(|_| {});
    let finding = plan
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .find(|f| f.subject.ends_with("output_gate_type"))
        .expect("the key is still censused");

    // Unchanged as a fact: still unresolved, still blocking the whole
    // model. This is the part that keeps the system honest.
    assert!(finding.blocks(), "{finding:?}");
    assert!(
        !plan.admissible,
        "whole-model admissibility must stay false"
    );

    // Changed only as a dependency: text does not reach it.
    assert_eq!(
        text_blocking(&plan),
        0,
        "text generation should not be gated by a key its operator cannot consume"
    );
}

/// **Falsifier 1.** A sibling unknown key with no disposition blocks text.
///
/// Deliberately named to look like it belongs to the gate. Resemblance
/// earns nothing; only an explicit, evidence-backed entry does.
#[test]
fn an_undisposed_unknown_text_key_still_blocks_text() {
    for key in ["output_gate_scale", "gate_type", "some_unjudged_knob"] {
        let plan = plan_with(|config| {
            config["text_config"][key] = serde_json::json!("whatever");
        });
        assert!(
            text_blocking(&plan) > 0,
            "`{key}` has no disposition and must block text — nothing generalises from \
             the two subjects that do"
        );
    }
}

/// **Falsifier 2, and the important one.** Text admission comes from the
/// independently understood operator, not from excluding the config key.
///
/// `output_gate_type` is left exactly as it is; only the structural
/// evidence is removed. Without a represented gate the disposition's
/// predicate is false, so the key blocks text again — which is the whole
/// claim: the exclusion is conditional on this container proving the
/// operator is determined without it.
#[test]
fn text_admission_comes_from_the_operator_not_from_excluding_the_key() {
    let gated = plan_with(|_| {});
    assert_eq!(text_blocking(&gated), 0);

    let ungated = plan_with(|config| {
        // The declaration that produces the judged gate, withdrawn. The
        // metadata key is untouched.
        config["text_config"]["attn_output_gate"] = serde_json::json!(false);
    });
    assert!(
        text_blocking(&ungated) > 0,
        "with no represented gate there is no evidence, so the key must block text \
         again — if this passes, admission was coming from the exclusion, not the operator"
    );
}

/// **D1 end to end.** The seven MTP findings leave `TextGeneration` and
/// stay visible, blocking, on `Drafting`.
#[test]
fn the_draft_head_leaves_text_and_remains_blocking_for_drafting() {
    let plan = plan_with(|config| {
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        config["text_config"]["mtp_use_dedicated_embeddings"] = serde_json::json!(false);
    });
    assert_eq!(text_blocking(&plan), 0);

    let drafting = plan
        .capabilities
        .iter()
        .find(|c| c.capability == Capability::Drafting)
        .expect("a drafting verdict");
    assert!(
        drafting.blocking >= 2,
        "the MTP declarations must still block the capability that would run them: {drafting:?}"
    );
    assert!(
        !drafting.supported,
        "no draft-head executor exists; claiming support would be the lie"
    );
}
