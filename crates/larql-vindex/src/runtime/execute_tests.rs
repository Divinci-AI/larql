//! Colocated tests for `execute` — paths fixture A does not reach.
//!
//! Fixture A covers the direct route against an oracle. These cover the
//! latent route, the per-expert scale policy, the raw-softmax policy and the
//! refusals — all reachable code that a direct f32 fixture never touches.

use larql_compute::{MoeExpertScalePolicy, MoeTopKWeightPolicy};

use super::execute::{execute, execute_traced, execute_with};
use super::operation_tests::{direct, latent};
use super::test_support::{ascending, vector};
use super::trace::{CollectedTrace, NoTrace, TraceSink};
use super::transform::TransformStage;

const RESIDUAL_WIDTH: usize = 6;
const LATENT_WIDTH: usize = 3;
const SCALE: &str = "router_scale";
const LATENT_OUT: &str = "latent_out";

fn residual() -> Vec<f32> {
    vec![0.4, -0.3, 0.9, 0.1, -0.7, 0.2]
}

// ── The latent route ───────────────────────────────────────────────────────

#[test]
fn a_latent_operation_returns_a_residual_width_delta() {
    // The point of the output transform: the banks run narrow, the delta
    // rejoins the residual at full width.
    let out = execute(&latent(), &residual()).unwrap();
    assert_eq!(out.len(), RESIDUAL_WIDTH);
}

#[test]
fn a_latent_operation_reduces_at_the_projected_width() {
    let (out, trace) = execute_traced(&latent(), &residual()).unwrap();
    assert_eq!(trace.routed_input.len(), LATENT_WIDTH);
    assert_eq!(trace.reduced.len(), LATENT_WIDTH);
    assert_eq!(trace.residual_delta.len(), RESIDUAL_WIDTH);
    assert_eq!(out, trace.residual_delta);
}

#[test]
fn a_latent_routed_input_is_not_the_residual() {
    // If the input transform were skipped, these would coincide.
    let (_, trace) = execute_traced(&latent(), &residual()).unwrap();
    assert_ne!(trace.routed_input, residual());
}

#[test]
fn a_direct_operation_routes_on_the_residual_itself() {
    let (_, trace) = execute_traced(&direct(), &residual()).unwrap();
    assert_eq!(trace.routed_input, residual());
}

#[test]
fn latent_and_direct_operations_both_execute_through_one_path() {
    for op in [direct(), latent()] {
        let out = execute(&op, &residual()).unwrap();
        assert_eq!(out.len(), RESIDUAL_WIDTH);
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
    }
}

// ── Weight policies ────────────────────────────────────────────────────────

#[test]
fn raw_softmax_leaves_the_selected_weights_unnormalised() {
    let mut op = direct();
    op.router.selected_weight = MoeTopKWeightPolicy::RawSoftmax;
    let (_, trace) = execute_traced(&op, &residual()).unwrap();
    let total: f32 = trace.gate_weights().iter().sum();
    assert!(total < 1.0, "top-k of a larger population sums to {total}");
}

#[test]
fn renormalised_and_raw_softmax_select_the_same_experts() {
    // The policy changes the weights, never the selection.
    let mut raw = direct();
    raw.router.selected_weight = MoeTopKWeightPolicy::RawSoftmax;
    let (_, raw_trace) = execute_traced(&raw, &residual()).unwrap();
    let (_, renorm_trace) = execute_traced(&direct(), &residual()).unwrap();
    assert_eq!(raw_trace.selected_ids(), renorm_trace.selected_ids());
    assert_ne!(raw_trace.gate_weights(), renorm_trace.gate_weights());
}

#[test]
fn a_per_expert_scale_multiplies_after_renormalisation() {
    // Applying it before would let renormalisation divide the learned scale
    // straight back out, making the whole policy a no-op.
    let mut op = direct();
    let population = op.banks[0].experts.len();
    op.router.expert_scale = MoeExpertScalePolicy::PerExpert;
    op.router.per_expert_scale = Some(vector(SCALE, &vec![2.0f32; population]));

    let (_, scaled) = execute_traced(&op, &residual()).unwrap();
    let (_, plain) = execute_traced(&direct(), &residual()).unwrap();
    let total: f32 = scaled.gate_weights().iter().sum();
    assert!((total - 2.0).abs() < 1e-5, "scaled weights sum to {total}");
    assert_eq!(scaled.selected_ids(), plain.selected_ids());
}

#[test]
fn a_scale_vector_is_ignored_when_the_policy_does_not_ask_for_it() {
    let mut op = direct();
    let population = op.banks[0].experts.len();
    op.router.expert_scale = MoeExpertScalePolicy::None;
    op.router.per_expert_scale = Some(vector(SCALE, &vec![9.0f32; population]));
    let (_, trace) = execute_traced(&op, &residual()).unwrap();
    let total: f32 = trace.gate_weights().iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "{total}");
}

// ── Depth ──────────────────────────────────────────────────────────────────

#[test]
fn top_k_bounds_how_many_experts_run() {
    let mut op = direct();
    op.router.top_k = 1;
    let (_, trace) = execute_traced(&op, &residual()).unwrap();
    assert_eq!(trace.selection.len(), 1);
    assert_eq!(trace.expert_outputs.len(), 1);
    assert!((trace.gate_weights()[0] - 1.0).abs() < 1e-6, "renormalised");
}

#[test]
fn a_top_k_of_zero_selects_nothing_and_contributes_nothing() {
    let mut op = direct();
    op.router.top_k = 0;
    let (out, trace) = execute_traced(&op, &residual()).unwrap();
    assert!(trace.selection.is_empty());
    assert_eq!(out, vec![0.0f32; RESIDUAL_WIDTH]);
}

#[test]
fn a_top_k_beyond_the_population_runs_every_expert() {
    let mut op = direct();
    let population = op.banks[0].experts.len();
    op.router.top_k = population + 5;
    let (_, trace) = execute_traced(&op, &residual()).unwrap();
    assert_eq!(trace.selection.len(), population);
}

// ── Refusals ───────────────────────────────────────────────────────────────

#[test]
fn a_residual_of_the_wrong_width_is_refused() {
    assert!(execute(&direct(), &[1.0, 2.0]).is_err());
    assert!(execute(&latent(), &[1.0, 2.0]).is_err());
}

#[test]
fn a_router_selecting_an_expert_the_bank_lacks_is_refused() {
    // Router and bank disagreeing about the population is a binding fault, and
    // must not degrade into a quietly missing contribution.
    let mut op = direct();
    op.banks[0].experts.truncate(1);
    op.banks[0].experts[0].expert_id = 0;
    op.router.top_k = op.router.population();
    let err = execute(&op, &residual()).unwrap_err();
    assert!(err.to_string().contains("selected expert"), "{err}");
}

#[test]
fn an_output_transform_that_lands_off_the_residual_width_is_refused() {
    // Constructed so the *input* stage still succeeds: widening `residual_dim`
    // instead would fail at the input projection and never reach this check.
    let mut op = latent();
    let output_stage = op
        .transforms
        .iter_mut()
        .find(|t| t.stage == TransformStage::RoutedOutput)
        .expect("the latent operation binds an output stage");
    output_stage.weight = ascending(LATENT_OUT, (RESIDUAL_WIDTH - 1) as u32, LATENT_WIDTH as u32);

    let err = execute(&op, &residual()).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("operation output"), "{text}");
    assert!(text.contains(&RESIDUAL_WIDTH.to_string()), "{text}");
}

// ── The sink is the only difference between the two entry points ───────────

#[test]
fn the_no_op_sink_and_the_collecting_sink_agree_on_the_result() {
    let op = latent();
    let mut none = NoTrace;
    let mut collected = CollectedTrace::default();
    let a = execute_with(&op, &residual(), &mut none).unwrap();
    let b = execute_with(&op, &residual(), &mut collected).unwrap();
    assert_eq!(a, b);
    assert_eq!(b, collected.residual_delta);
}

#[test]
fn the_no_op_sink_accepts_every_checkpoint_without_recording() {
    // Defaulted trait methods: the production sink must stay a single empty
    // impl even as checkpoints are added.
    let mut sink = NoTrace;
    sink.routed_input(&[1.0]);
    sink.router_scores(&[1.0]);
    sink.selection(&[]);
    sink.expert_output(0, &[1.0]);
    sink.reduced(&[1.0]);
    sink.residual_delta(&[1.0]);
}

#[test]
fn a_collected_trace_looks_up_expert_outputs_by_id() {
    let (_, trace) = execute_traced(&direct(), &residual()).unwrap();
    let first = trace.selected_ids()[0];
    assert!(trace.expert_output(first).is_some());
    assert!(trace.expert_output(u32::MAX).is_none());
}

#[test]
fn the_latent_stages_are_the_only_transforms_bound() {
    let op = latent();
    let stages: Vec<TransformStage> = op.transforms.iter().map(|t| t.stage).collect();
    assert_eq!(
        stages,
        vec![TransformStage::RoutedInput, TransformStage::RoutedOutput]
    );
}
