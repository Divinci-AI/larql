//! QW-3: heterogeneous continuation state, over Qwen3.8's real topology.
//!
//! The claim is not that the enum has the right variants. It is that the
//! **memory model** differs: a recurrent layer's footprint is constant in
//! context length and a KV layer's is not. That is checked as arithmetic at
//! two very different context lengths, not asserted from a label.
//!
//! The 64-layer plan here is assembled directly rather than read from a
//! container, because Qwen3.8 is not yet encodable (24 blocking findings,
//! mostly vision). Its geometry and cadence are the real checkpoint's,
//! verified against it in `vindex3 plan`; what is synthetic is the plan
//! object, not the architecture it describes.

use super::super::continuation::{
    plan_continuation_geometry, LayerContinuationGeometry, StateInitialization,
};
use super::super::kv::try_plan_kv_geometry;
use crate::format::vindex3::opplan::{ComponentOpPlan, GatedDeltaOp, LayerAttention};
use larql_models::inventory::report::RecurrentStateDtype;

/// Qwen3.8-27B, from its own config.
const LAYERS: usize = 64;
const FULL_ATTENTION_INTERVAL: usize = 4;
const VALUE_HEADS: usize = 48;
const KEY_HEADS: usize = 16;
const HEAD_DIM: usize = 128;
const CONV_KERNEL: usize = 4;
/// Full-attention layers: GQA 24/4 at head_dim 256.
const FULL_KV_HEADS: usize = 4;
const FULL_HEAD_DIM: usize = 256;

/// 48 · 128 · 128 · 4 bytes ≈ 3 MB per recurrent layer.
const STATE_ELEMENTS_PER_LAYER: usize = VALUE_HEADS * HEAD_DIM * HEAD_DIM;

fn gated_delta_op() -> GatedDeltaOp {
    let operand = || crate::format::vindex3::opplan::OperandRef {
        object: "target.decoder_stack".into(),
        tensor: "synthetic".into(),
        dtype: "BF16".into(),
        shape: vec![1],
    };
    GatedDeltaOp {
        num_key_heads: KEY_HEADS,
        num_value_heads: VALUE_HEADS,
        key_head_dim: HEAD_DIM,
        value_head_dim: HEAD_DIM,
        conv_kernel: CONV_KERNEL,
        state_dtype: Some(RecurrentStateDtype::Float32),
        in_proj_qkv: operand(),
        in_proj_a: operand(),
        in_proj_b: operand(),
        in_proj_z: operand(),
        conv1d: operand(),
        a_log: operand(),
        dt_bias: operand(),
        norm: operand(),
        out_proj: operand(),
    }
}

/// A 64-layer hybrid on Qwen3.8's LLLF cadence.
fn qwen_like_plan() -> ComponentOpPlan {
    let mut plan =
        super::plan_fixtures::softmax_plan_with_layers(LAYERS, FULL_KV_HEADS, FULL_HEAD_DIM);
    for (index, layer) in plan.layers.iter_mut().enumerate() {
        if index % FULL_ATTENTION_INTERVAL != FULL_ATTENTION_INTERVAL - 1 {
            layer.attention = LayerAttention::GatedDelta(Box::new(gated_delta_op()));
        }
    }
    plan
}

/// The census, on the real cadence.
#[test]
fn the_hybrid_topology_resolves_to_the_right_mix() {
    let plan = qwen_like_plan();
    let geometry = plan_continuation_geometry(&plan).expect("every layer declares its state");
    assert_eq!(geometry.len(), LAYERS);

    let recurrent = geometry.iter().filter(|g| g.recurrent().is_some()).count();
    let kv = geometry.iter().filter(|g| g.kv().is_some()).count();
    assert_eq!((recurrent, kv), (48, 16), "Qwen3.8 is 48 recurrent + 16 KV");

    let cadence: String = geometry
        .iter()
        .map(|g| if g.recurrent().is_some() { 'L' } else { 'F' })
        .collect();
    assert!(
        cadence.as_bytes().chunks(4).all(|c| c == b"LLLF"),
        "cadence is not uniform LLLF: {cadence}"
    );

    let r = geometry[0].recurrent().expect("layer 0 is recurrent");
    assert_eq!(r.shape, vec![VALUE_HEADS, HEAD_DIM, HEAD_DIM]);
    assert_eq!(r.dtype, RecurrentStateDtype::Float32);
    assert_eq!(r.initialization, StateInitialization::Zeros);
    assert_eq!(r.elements(), STATE_ELEMENTS_PER_LAYER);
}

/// Claim 1: no fake KV. Claim 2: no phantom recurrence.
///
/// Both are the same failure in opposite directions — a planner that sized
/// every layer the same way would pass a census and still allocate wrongly.
#[test]
fn neither_kind_of_layer_allocates_the_other_kind_of_state() {
    let plan = qwen_like_plan();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    for (index, g) in geometry.iter().enumerate() {
        let recurrent_layer = index % FULL_ATTENTION_INTERVAL != FULL_ATTENTION_INTERVAL - 1;
        if recurrent_layer {
            assert!(
                g.kv().is_none(),
                "layer {index} allocated KV for a recurrence"
            );
        } else {
            assert!(
                g.recurrent().is_none(),
                "layer {index} allocated recurrent state for softmax attention"
            );
        }
    }
}

/// Claims 3 and 4, and the whole architectural point: the two mechanisms
/// respond differently to context length.
#[test]
fn only_the_softmax_layers_grow_with_context() {
    let plan = qwen_like_plan();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let total = |positions: usize, pick: fn(&LayerContinuationGeometry) -> bool| -> usize {
        geometry
            .iter()
            .filter(|g| pick(g))
            .map(|g| g.elements_at(positions))
            .sum()
    };
    let is_recurrent = |g: &LayerContinuationGeometry| g.recurrent().is_some();
    let is_kv = |g: &LayerContinuationGeometry| g.kv().is_some();

    let (short, long) = (128usize, 32_768usize);
    let rec_short = total(short, is_recurrent);
    let rec_long = total(long, is_recurrent);
    let kv_short = total(short, is_kv);
    let kv_long = total(long, is_kv);

    assert_eq!(
        rec_short, rec_long,
        "recurrent footprint moved with context: {rec_short} -> {rec_long}"
    );
    assert!(
        kv_long > kv_short,
        "KV footprint did not grow with context: {kv_short} -> {kv_long}"
    );
    assert_eq!(
        kv_long / kv_short,
        long / short,
        "KV should scale linearly in positions"
    );

    // 48 layers x 48*128*128 f32 ~= 144 MB, whatever the context.
    let bytes = rec_short * 4;
    assert_eq!(bytes, 48 * STATE_ELEMENTS_PER_LAYER * 4);
    assert!(
        (140..=152).contains(&(bytes / 1_000_000)),
        "recurrent total is {} MB, expected ~144",
        bytes / 1_000_000
    );
}

/// The compatibility seam must fail loudly on a hybrid, not answer for the
/// layers it happens to understand.
#[test]
fn the_kv_adapter_refuses_a_hybrid_model() {
    let err = try_plan_kv_geometry(&qwen_like_plan())
        .expect_err("a model with recurrent layers has no flat KV geometry");
    assert!(
        err.contains("plan_continuation_geometry"),
        "the refusal must name the seam that can describe this model: {err}"
    );
}

/// The control: a wholly-softmax model is unchanged, and the adapter still
/// answers for it exactly as before.
#[test]
fn a_pure_softmax_model_is_untouched() {
    let plan = super::plan_fixtures::softmax_plan_with_layers(LAYERS, FULL_KV_HEADS, FULL_HEAD_DIM);
    let geometry = plan_continuation_geometry(&plan).unwrap();
    assert_eq!(geometry.iter().filter(|g| g.kv().is_some()).count(), LAYERS);
    assert!(geometry.iter().all(|g| g.recurrent().is_none()));

    let flat = try_plan_kv_geometry(&plan).expect("a softmax model still flattens");
    assert_eq!(flat.len(), LAYERS);
    for (g, kv) in geometry.iter().zip(&flat) {
        assert_eq!(g.kv().unwrap(), kv, "the adapter changed a KV geometry");
    }
}

/// The type-level control: `RecurrentGeometry` must not have quietly
/// learned what a Gated DeltaNet is.
///
/// Qwen3.8's state happens to be rank-3 and head-shaped. If any of that had
/// leaked into the type — a head count, a rank assumption, a square
/// `Dk × Dv` — then KDA, whose state is a different shape under a different
/// update rule, would need a second variant and this abstraction would have
/// failed at its first real test. So the control states states that are
/// deliberately nothing like DeltaNet's and requires the same arithmetic to
/// hold.
#[test]
fn recurrent_geometry_does_not_assume_the_operator() {
    use super::super::continuation::RecurrentGeometry;

    let shapes: [Vec<usize>; 4] = [
        vec![7],                               // rank 1: a plain vector of state
        vec![13, 5],                           // rank 2, non-square, no head axis
        vec![2, 3, 5, 7],                      // rank 4
        vec![VALUE_HEADS, HEAD_DIM, HEAD_DIM], // and DeltaNet's, for contrast
    ];
    for shape in shapes {
        let expected: usize = shape.iter().product();
        let g = RecurrentGeometry {
            shape: shape.clone(),
            dtype: RecurrentStateDtype::Float32,
            initialization: StateInitialization::Zeros,
        };
        assert_eq!(g.elements(), expected, "{shape:?}");
        assert_eq!(g.bytes(), expected * 4, "{shape:?}");

        // The defining property, independent of shape: constant in context.
        let layer = LayerContinuationGeometry::Recurrent(g);
        assert_eq!(layer.elements_at(1), expected, "{shape:?}");
        assert_eq!(layer.elements_at(1_000_000), expected, "{shape:?}");
    }
}

/// A recurrence whose state precision was never declared cannot be sized,
/// and the planner says so rather than choosing one.
#[test]
fn an_undeclared_state_precision_refuses_rather_than_defaulting() {
    let mut plan = qwen_like_plan();
    if let LayerAttention::GatedDelta(op) = &mut plan.layers[0].attention {
        op.state_dtype = None;
    }
    let err =
        plan_continuation_geometry(&plan).expect_err("an unsized recurrence must not be planned");
    assert!(
        err.contains("layer 0") && err.contains("precision"),
        "the refusal must name the layer and the missing fact: {err}"
    );
}
