//! Stage C: causal authority of the persisted IR (V3-G5b-2c).
//!
//! Three controls, three distinct semantic pathways, one rule: each
//! mutation touches **only the container's persisted graph** — never
//! oracle code, never executor code. Together they prove the IR's facts
//! are load-bearing:
//!
//! ```text
//! C1  query_scale 3.87 → 3.5      scalar semantics are authoritative
//! C2  layer 1 None → Rope         per-layer topology is authoritative
//! C3  remove the gate judgment    operation semantics are authoritative
//!                                 (fail-closed: refusal, not drift)
//! ```
//!
//! A hidden default is precisely a fact whose mutation changes nothing;
//! these tests are the search for hidden defaults.

use super::golden::{executor_trace_from, golden_forward, max_abs, miniature_glimmer, G_LAYERS};
use crate::format::vindex3::encode::{encode_system, SYSTEM_GRAPH_JSON};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ClosureDefect};

/// Divergence threshold: well above fp noise (~5e-8 measured), well
/// below any semantic effect.
const NOISE_CEILING: f32 = 1e-5;

/// Encode the miniature fixture and hand back the container dir.
fn encoded_container(dir: &std::path::Path) -> tempfile::TempDir {
    let inventory = larql_models::inventory::build_inventory(dir).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();
    container
}

/// Edit one component's persisted graph JSON in place.
fn mutate_graph(container: &std::path::Path, mutate: impl FnOnce(&mut serde_json::Value)) {
    let path = container.join(SYSTEM_GRAPH_JSON);
    let mut graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let target = graph["components"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["id"] == "target")
        .unwrap();
    mutate(target);
    std::fs::write(&path, graph.to_string()).unwrap();
}

/// C1 — scalar authority: a mutated persisted `query_scale` must change
/// computation from the first layer onward. The executor and oracle are
/// untouched; only the IR moved.
#[test]
fn c1_query_scale_mutation_changes_computation() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let golden = golden_forward(dir.path());
    let container = encoded_container(dir.path());
    mutate_graph(container.path(), |target| {
        target["execution"]["attention"]["query_scale"] = serde_json::json!(3.5);
    });
    let executed = executor_trace_from(container.path());

    let layer0 = max_abs(
        &executed.layers[0].post_attention,
        &golden.layers[0].post_attention,
    );
    assert!(
        layer0 > NOISE_CEILING,
        "query_scale must be causally load-bearing from layer 0 (diff {layer0:e})"
    );
}

/// C2 — layer-policy authority: flipping layer 1 from NoPE to RoPE must
/// leave layer 0 *identical* to golden and diverge exactly at layer 1 —
/// the location of first divergence is predictable from the mutation.
#[test]
fn c2_position_policy_mutation_diverges_exactly_at_its_layer() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let golden = golden_forward(dir.path());
    let container = encoded_container(dir.path());
    mutate_graph(container.path(), |target| {
        target["attention"][1]["position"] =
            serde_json::json!({ "kind": "rope", "theta": 500000.0 });
    });
    let executed = executor_trace_from(container.path());

    let layer0 = max_abs(&executed.layers[0].post_layer, &golden.layers[0].post_layer);
    assert!(
        layer0 < NOISE_CEILING,
        "layer 0 precedes the mutation and must match golden (diff {layer0:e})"
    );
    let layer1 = max_abs(
        &executed.layers[1].post_attention,
        &golden.layers[1].post_attention,
    );
    assert!(
        layer1 > NOISE_CEILING,
        "layer 1 carries the mutation and must diverge (diff {layer1:e})"
    );
    assert_eq!(G_LAYERS, 2);
}

/// C3 — operation-semantic authority: removing the judged gate semantics
/// from the persisted surface must REFUSE at operand closure (the gate
/// operand still exists), naming the primitive — fail-closed all the way
/// into execution, never a silently ungated forward.
#[test]
fn c3_removing_gate_judgment_refuses_execution() {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let container = encoded_container(dir.path());
    mutate_graph(container.path(), |target| {
        target["execution"]["attention"]
            .as_object_mut()
            .unwrap()
            .remove("output_gate");
    });

    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(
        outcome.plan.is_none(),
        "an unjudged gate operand must not plan"
    );
    let named = outcome.defects.iter().any(|d| {
        matches!(
            d,
            ClosureDefect::OperandImpliesAbsentOp { required_primitive, .. }
                if required_primitive.contains("attention output gate")
        )
    });
    assert!(
        named,
        "the refusal must name the primitive: {:?}",
        outcome.defects
    );
}
