//! The plan must pair a format with a kernel that consumes it, and the
//! executor's observation must land on the loader's decision.

use super::super::physical::{
    compact_threshold_bytes, project_matrix, ExecutorProjections, PhysicalProjectionPlan, F32_BYTES,
};
use super::super::projector::WeightRows;
use crate::format::vindex3::opplan::exec::backend::{WeightFormat, WeightSlice};
use crate::format::vindex3::opplan::exec::gated_delta::DenseProjections;

/// Every matrix Qwen3.8-27B decodes through, from the container's own
/// tensor table: `(name, elements)`.
///
/// The whole population rather than a sample, because the claim the
/// policy makes is about the model's residency — and a residency claim
/// that skipped a class would be a claim about part of a model.
///
/// **All thirteen are stored BF16.** The container reports one encoding
/// for the decoder stack, so nothing here is separated by what the
/// checkpoint holds; the two populations below are separated by SIZE
/// alone. A table that marked the delta gates "not stored bf16" would
/// pass the same assertions while testing nothing — the gates would be
/// f32-resident because of the checkpoint, and the threshold could be
/// any number at all.
const REAL_MATRICES: &[(&str, usize)] = &[
    ("mlp.gate_proj", 17408 * 5120),
    ("mlp.up_proj", 17408 * 5120),
    ("mlp.down_proj", 5120 * 17408),
    ("linear_attn.in_proj_qkv", 10240 * 5120),
    ("linear_attn.in_proj_z", 6144 * 5120),
    ("linear_attn.out_proj", 5120 * 6144),
    ("self_attn.q_proj", 12288 * 5120),
    ("self_attn.o_proj", 5120 * 6144),
    ("self_attn.k_proj", 1024 * 5120),
    ("self_attn.v_proj", 1024 * 5120),
    ("linear_attn.in_proj_a", 48 * 5120),
    ("linear_attn.in_proj_b", 48 * 5120),
    ("output_head", 248320 * 5120),
];

/// The stored encoding of every one of them, per the container index.
const STORED_BF16: bool = true;

/// A slab of the plan's own format, so a mispairing cannot be papered
/// over by the test choosing the representation the kernel wanted.
fn slab(plan: PhysicalProjectionPlan, elements: usize) -> (Vec<f32>, Vec<u16>) {
    match plan.format() {
        WeightFormat::F32 => (vec![0.5f32; elements], Vec::new()),
        WeightFormat::Bf16 => (Vec::new(), vec![0x3f00u16; elements]),
        other => panic!("no CPU plan declares {other:?}"),
    }
}

fn rows<'a>(plan: PhysicalProjectionPlan, f: &'a [f32], b: &'a [u16]) -> WeightRows<'a> {
    match plan.format() {
        WeightFormat::F32 => WeightRows::F32(f),
        WeightFormat::Bf16 => WeightRows::Bf16(b),
        other => panic!("no CPU plan declares {other:?}"),
    }
}

/// **The load-bearing invariant.** What the loader made resident is what
/// the executor observes.
///
/// This is the whole reason the plan is one value: if `choose` and
/// `for_resident` could disagree about a matrix, a BF16-resident weight
/// could be handed to a kernel expecting f32 — and the failure mode is
/// not a wrong answer but 100 MB read as garbage.
#[test]
fn the_observation_lands_on_the_decision() {
    for (name, elements) in REAL_MATRICES.iter().copied() {
        let chosen = PhysicalProjectionPlan::choose(elements, STORED_BF16);
        // A one-row stand-in: the round trip is about representation, and
        // allocating 1.3 G elements to prove it would measure the
        // allocator.
        let (f, b) = slab(chosen, 8);
        let observed = PhysicalProjectionPlan::for_resident(rows(chosen, &f, &b));
        assert_eq!(
            observed, chosen,
            "`{name}`: the executor observed {observed:?} where the loader chose {chosen:?} — \
             one matrix, two derivations, and they disagree"
        );
    }
}

/// Each plan's kernel actually consumes each plan's format.
///
/// The kernels panic on the wrong representation, so a mispaired variant
/// fails here loudly rather than at decode on a real container.
#[test]
fn every_plan_runs_its_own_format() {
    let x = vec![1.0f32; 8];
    for plan in [
        PhysicalProjectionPlan::ScalarF32,
        PhysicalProjectionPlan::BlasF32,
        PhysicalProjectionPlan::FusedBf16,
    ] {
        let (f, b) = slab(plan, 8 * 2);
        let mut out = vec![0.0f32; 2];
        plan.kernel().project_rows(rows(plan, &f, &b), &x, &mut out);
        assert!(
            out.iter().all(|v| v.is_finite() && *v != 0.0),
            "{plan:?} produced nothing from its own declared format"
        );
    }
}

/// The oracle is chosen by IDENTITY, not by representation.
///
/// `for_resident` is total over what a CPU kernel can hold, and f32 has
/// two kernels: the production `BlasF32` and the reference `ScalarF32`.
/// It answers `BlasF32`, and that is not an omission — the reference
/// backend declares its plan because of what it IS, so nothing ever asks
/// the bytes which of the two it wanted. Asserting the asymmetry here
/// stops a later reader "fixing" it by making the observation guess.
#[test]
fn the_oracle_is_not_reachable_by_observation() {
    let f = vec![0.5f32; 8];
    assert_eq!(
        PhysicalProjectionPlan::for_resident(WeightRows::F32(&f)),
        PhysicalProjectionPlan::BlasF32
    );
    let at = compact_threshold_bytes() / F32_BYTES;
    for elements in [1, at - 1, at, at * 64] {
        for stored in [false, true] {
            assert_ne!(
                PhysicalProjectionPlan::choose(elements, stored),
                PhysicalProjectionPlan::ScalarF32,
                "the policy must never route production through the oracle"
            );
        }
    }
}

/// An f32 checkpoint never reaches the compact kernel, however large.
///
/// The alternative would be to narrow at load to hit the threshold, which
/// would ROUND — the policy would be quantising a model while reporting a
/// residency win.
#[test]
fn a_checkpoint_without_stored_bf16_stays_f32() {
    let huge = 1_000 * compact_threshold_bytes() / F32_BYTES;
    assert_eq!(
        PhysicalProjectionPlan::choose(huge, false),
        PhysicalProjectionPlan::BlasF32
    );
    assert_eq!(
        PhysicalProjectionPlan::choose(huge, true),
        PhysicalProjectionPlan::FusedBf16
    );
}

/// The boundary is the cache boundary, and it is bracketed both ways.
///
/// A one-sided assertion would pass against a policy that answered
/// `FusedBf16` for everything, which is exactly the failure the
/// `48 x 5120` delta gates exist to catch — they lose 3.8x through the
/// fused kernel.
#[test]
fn the_threshold_is_bracketed_on_both_sides() {
    let at = compact_threshold_bytes() / F32_BYTES;
    assert_eq!(
        PhysicalProjectionPlan::choose(at - 1, true),
        PhysicalProjectionPlan::BlasF32,
        "one element below the cache boundary must still be a BLAS matrix"
    );
    assert_eq!(
        PhysicalProjectionPlan::choose(at, true),
        PhysicalProjectionPlan::FusedBf16,
        "at the cache boundary the widened image no longer fits, so compact wins"
    );
}

/// The real model's two populations land on opposite sides.
///
/// Named separately from the bracket because this is the claim that
/// matters for Qwen3.8: 51.2 GB of matrix stays compact and the 48 MB of
/// delta gates do not, and neither number survives a policy that answers
/// uniformly.
#[test]
fn the_real_model_splits_into_two_populations() {
    let compact: Vec<&str> = REAL_MATRICES
        .iter()
        .filter(|(_, e)| {
            PhysicalProjectionPlan::choose(*e, STORED_BF16) == PhysicalProjectionPlan::FusedBf16
        })
        .map(|(n, _)| *n)
        .collect();
    assert_eq!(
        compact.len(),
        REAL_MATRICES.len() - 2,
        "compact set: {compact:?}"
    );
    assert!(!compact.contains(&"linear_attn.in_proj_a"));
    assert!(!compact.contains(&"linear_attn.in_proj_b"));
}

/// A projection runs through the executor under its own plan, whichever
/// representation it is resident as, and the two agree on the answer.
///
/// The bf16 slab holds exactly the values the f32 one does — bf16 is the
/// top half of f32 — so this prices the KERNELS and nothing else. A
/// disagreement beyond reassociation would mean the widen was not a
/// widen.
#[test]
fn both_representations_project_to_the_same_answer() {
    const OUT: usize = 24;
    const IN: usize = 32;
    let f: Vec<f32> = (0..OUT * IN)
        .map(|i| {
            let v = (i as f32 * 0.013).sin();
            f32::from_bits(v.to_bits() & 0xffff_0000)
        })
        .collect();
    let b: Vec<u16> = f.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let x: Vec<f32> = (0..IN).map(|i| (i as f32 * 0.07).cos()).collect();

    let widened = project_matrix(&WeightSlice::F32(&f), &x, OUT, IN).unwrap();
    let compact = project_matrix(&WeightSlice::Bf16(&b), &x, OUT, IN).unwrap();
    let gated = ExecutorProjections.project(WeightRows::Bf16(&b), &x, OUT);
    assert_eq!(
        compact, gated,
        "the delta seam and the plan seam must agree exactly"
    );

    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (p, q) in compact.iter().zip(&widened) {
        num += (*p as f64 - *q as f64).powi(2);
        den += (*q as f64).powi(2);
    }
    assert!((num / den.max(f64::MIN_POSITIVE)).sqrt() < 1e-5);
}

/// A representation no CPU kernel runs refuses, and names itself.
#[test]
fn a_device_only_representation_refuses_by_name() {
    let err = project_matrix(&WeightSlice::F16(&[0u8; 64]), &[1.0f32; 4], 4, 4)
        .expect_err("no CPU kernel consumes f16")
        .to_string();
    assert!(err.contains("f16"), "{err}");
}

/// The threshold is a real cache size, whatever machine reads it.
#[test]
fn the_threshold_is_a_plausible_cache_size() {
    let bytes = compact_threshold_bytes();
    assert!(
        (1 << 20..=1 << 30).contains(&bytes),
        "{bytes} is not a plausible L2 size — a threshold this far out would put every \
         matrix on one side"
    );
}
