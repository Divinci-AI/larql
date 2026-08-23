//! Hybrid linear-attention interleaves (Qwen3.5/Kimi-Linear-style):
//! `layer_types` declaring a span kind outside VINDEX3's executable
//! vocabulary must block honestly — not fabricate a collapsed "all full"
//! resolution — and must not take unrelated per-layer facts (rope theta)
//! down with it.

use super::support::{glimmer_shaped_target_with, FIXTURE_LAYERS};
use crate::format::vindex3::plan::{plan_system, Finding, FindingCategory, SemanticClass};

/// The Glimmer-shaped fixture with its `layer_types` swapped for a
/// Qwen3.5-style hybrid interleave — three `linear_attention` layers to
/// one `full_attention` layer — plus the declared-but-unexecuted hybrid
/// linear-attention / MTP / mRoPE fields a real Qwen3.5 `config.json`
/// carries alongside it.
fn hybrid_findings() -> Vec<Finding> {
    let dir = tempfile::tempdir().unwrap();
    let inventory = glimmer_shaped_target_with(dir.path(), |config| {
        let layer_types: Vec<&str> = (0..FIXTURE_LAYERS)
            .map(|i| {
                if i % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        config["text_config"]["layer_types"] = serde_json::json!(layer_types);
        config["text_config"]["full_attention_interval"] = serde_json::json!(4);
        config["text_config"]["linear_conv_kernel_dim"] = serde_json::json!(4);
        config["text_config"]["linear_key_head_dim"] = serde_json::json!(16);
        config["text_config"]["linear_value_head_dim"] = serde_json::json!(16);
        config["text_config"]["linear_num_key_heads"] = serde_json::json!(2);
        config["text_config"]["linear_num_value_heads"] = serde_json::json!(4);
        config["text_config"]["mamba_ssm_dtype"] = serde_json::json!("float32");
        config["text_config"]["attn_output_gate"] = serde_json::json!(true);
        config["text_config"]["output_gate_type"] = serde_json::json!("swish");
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        config["text_config"]["mtp_use_dedicated_embeddings"] = serde_json::json!(false);
        config["text_config"]["rope_parameters"]["mrope_interleaved"] = serde_json::json!(true);
        config["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([2, 2, 1]);
    });
    let named = vec![("target-artifact".to_string(), inventory)];
    plan_system(&named)
        .artifacts
        .into_iter()
        .flat_map(|a| a.findings)
        .collect()
}

fn finding_for<'a>(findings: &'a [Finding], suffix: &str) -> &'a Finding {
    findings
        .iter()
        .find(|f| f.subject.ends_with(suffix))
        .unwrap_or_else(|| panic!("no finding for `{suffix}`"))
}

/// The core fix: a declared span outside the executable vocabulary
/// (`linear_attention`) must not resolve to a fabricated "all full"
/// value. It stays `Unrepresented` — an honest statement that the schema
/// has no home for it yet — rather than a `Representable`/`Mismatched`
/// verdict built on a resolved value nothing actually computed.
///
/// Two independent findings carry `text_config.layer_types`: the
/// declared-vs-resolved comparator (`compare::layer_types_finding`,
/// `carriage: None`) and the carriage gate's probe
/// (`carriage::probe_layer_types`, `carriage: Some(_)`). Both must
/// refuse to call the interleave carried — neither may echo the
/// collapsed "all full" default as if it were a real resolution.
#[test]
fn hybrid_layer_types_blocks_honestly_on_both_findings() {
    let findings = hybrid_findings();
    let layer_types_findings: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.subject == "text_config.layer_types")
        .collect();
    assert_eq!(
        layer_types_findings.len(),
        2,
        "expected the comparator finding and the carriage finding"
    );

    let fabricated_all_full = serde_json::json!(vec!["full_attention"; FIXTURE_LAYERS]);
    for finding in &layer_types_findings {
        assert_ne!(
            finding.category,
            FindingCategory::Representable,
            "{finding:?}"
        );
        assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
        assert!(finding.blocks(), "{finding:?}");
        assert_ne!(
            finding.resolved,
            Some(fabricated_all_full.clone()),
            "must not fabricate a collapsed all-full resolution: {finding:?}"
        );
    }

    // The carriage-gate finding specifically: parsed, but nothing built
    // could vouch for the declared interleave.
    let carriage_finding = layer_types_findings
        .iter()
        .find(|f| f.carriage.is_some())
        .expect("a carriage-sourced finding");
    assert_eq!(carriage_finding.category, FindingCategory::Unrepresented);

    // The comparator finding: an honest disagreement count, not a
    // fabricated exact match.
    let comparator_finding = layer_types_findings
        .iter()
        .find(|f| f.carriage.is_none())
        .expect("a comparator-sourced finding");
    assert_eq!(comparator_finding.category, FindingCategory::Mismatched);

    // The full declared interleave is still visible on at least one
    // finding — "the container records what's actually declared" is
    // satisfied regardless of what the graph could resolve.
    let declared_array: Vec<&str> = (0..FIXTURE_LAYERS)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    assert!(
        layer_types_findings
            .iter()
            .any(|f| f.declared == Some(serde_json::json!(declared_array))),
        "the full declared interleave must survive to the report"
    );
}

/// **Regression guard.** The span fix must not take unrelated per-layer
/// facts down with it: rope theta is carried independently of span, and
/// must stay representable even while `layer_types` blocks.
#[test]
fn unrelated_per_layer_facts_still_carry_with_a_hybrid_interleave() {
    let findings = hybrid_findings();
    for subject in [
        "text_config.rope_parameters.rope_theta",
        "text_config.layer_rope_theta",
    ] {
        let finding = finding_for(&findings, subject);
        assert_eq!(
            finding.category,
            FindingCategory::Representable,
            "{subject}: {}",
            finding.detail
        );
        assert!(!finding.blocks(), "{subject}");
    }
}

/// The hybrid fields that still have no destination stay honestly
/// `unrepresented`.
///
/// QW-1 gave the five linear GEOMETRY fields a real destination and QW-2
/// gave `mamba_ssm_dtype` one — see
/// [`the_linear_geometry_is_carried_into_the_operator`] and
/// [`the_state_precision_moves_with_its_executor`]. What remains here has
/// none: MTP has no head object, mRoPE has no multi-axis position policy,
/// and the attention output gate has no judged semantics. Each stays
/// honestly `unrepresented` rather than claiming a home that does not
/// exist.
#[test]
fn declared_hybrid_fields_without_a_destination_stay_unrepresented() {
    let findings = hybrid_findings();
    for subject in [
        "text_config.attn_output_gate",
        "text_config.output_gate_type",
        "text_config.mtp_num_hidden_layers",
        "text_config.mtp_use_dedicated_embeddings",
        "text_config.rope_parameters.mrope_interleaved",
        "text_config.rope_parameters.mrope_section",
    ] {
        let finding = finding_for(&findings, subject);
        assert_eq!(finding.class, SemanticClass::ExecutionSemantic, "{subject}");
        assert_eq!(
            finding.category,
            FindingCategory::Unrepresented,
            "{subject}: {}",
            finding.detail
        );
        assert!(
            finding.detail.contains("no schema field"),
            "{subject}: {} — must not be silently dropped, must name the missing judgement",
            finding.detail
        );
        assert!(finding.blocks(), "{subject}");
    }
}

/// `mamba_ssm_dtype` moved only when something could honour it.
///
/// QW-1 deliberately left it blocking while the field was parsed and
/// nearby: with no executor able to keep a recurrence at a declared
/// precision, claiming carriage would have asserted a runtime surface that
/// could not use the value. QW-2's reference operator allocates and
/// accumulates `GatedDeltaState` at exactly this precision, so the claim is
/// now true — and it is the ONLY blocker that rung moved.
#[test]
fn the_state_precision_moves_with_its_executor() {
    let findings = hybrid_findings();
    let finding = finding_for(&findings, "text_config.mamba_ssm_dtype");
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert!(!finding.blocks());
    assert!(
        finding.detail.contains("state_dtype") || finding.detail.contains("linear_attention"),
        "must name where it lands: {}",
        finding.detail
    );
}

/// The five linear geometry fields now terminate in a real operator.
///
/// Each lands on `ExecutionSurface.linear_attention` and is consumed by
/// `GatedDeltaOp`, and together they derive the `qkv_channels` and
/// `value_width` that the nine `LinearAttn*` operand contracts close
/// against stored tensors. That whole path is why these grade `Lowered`
/// while `mamba_ssm_dtype` beside them does not.
#[test]
fn the_linear_geometry_is_carried_into_the_operator() {
    let findings = hybrid_findings();
    for subject in [
        "text_config.linear_conv_kernel_dim",
        "text_config.linear_key_head_dim",
        "text_config.linear_value_head_dim",
        "text_config.linear_num_key_heads",
        "text_config.linear_num_value_heads",
    ] {
        let finding = finding_for(&findings, subject);
        assert_eq!(
            finding.category,
            FindingCategory::Representable,
            "{subject}: {}",
            finding.detail
        );
        assert!(!finding.blocks(), "{subject}");
        assert!(
            finding.detail.contains("linear_attention"),
            "{subject} must name where it lands: {}",
            finding.detail
        );
    }
}

/// `full_attention_interval` is a redundant spelling of the same
/// interleave `layer_types` states explicitly, and the parser reads the
/// array — so this leaf grades `Alias` and does not block, even while
/// the interleave it aliases is itself unrepresented.
#[test]
fn full_attention_interval_is_a_non_blocking_alias() {
    let findings = hybrid_findings();
    let finding = finding_for(&findings, "text_config.full_attention_interval");
    assert_eq!(finding.class, SemanticClass::Alias);
    assert!(!finding.blocks());
}

/// The informational per-component attention-policy summary discloses
/// the gap rather than reporting a silently fabricated full count.
#[test]
fn attention_policy_summary_discloses_the_unexecutable_span() {
    let findings = hybrid_findings();
    let finding = findings
        .iter()
        .find(|f| f.subject == "attention_policy")
        .expect("attention policy finding");
    assert_eq!(finding.category, FindingCategory::Representable);
    assert!(
        finding
            .detail
            .contains("this schema has no execution vocabulary for"),
        "{}",
        finding.detail
    );
}
