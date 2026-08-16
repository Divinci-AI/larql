//! Decode-vs-batch parity: the two traversals realise one program.
//!
//! A [`DecodeSession`] feeds tokens one at a time through
//! `attention_step` against its own K/V cache; the batch path computes
//! every position in one pass. Per position the arithmetic is the same
//! operations on the same values in the same order, so the final
//! logits must agree **bit-for-bit** per backend — any gap means the
//! step path re-derived something (a rope position, a mask start, a
//! norm placement) instead of inheriting it.
//!
//! The miniature fixture's sliding window (3, over 5 positions)
//! genuinely truncates from position 3, so this parity also pins the
//! step path's masking — the cache may hold a position the span must
//! exclude, and only the span logic keeps it out.

use super::golden::{miniature_glimmer, G_TOKENS};
use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{PlanBackend, WeightFormat};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::device::DevicePlanBackend;
use crate::format::vindex3::opplan::exec::execute_plan;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};

fn fixture() -> (tempfile::TempDir, ComponentOpPlan, OperandStore) {
    let dir = tempfile::tempdir().unwrap();
    miniature_glimmer(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("mini-glimmer".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let outcome = plan_component_ops(&inspection, container.path(), "target").unwrap();
    assert!(outcome.closed(), "defects: {:?}", outcome.defects);
    let plan = outcome.plan.unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

/// Step every fixture token through a fresh session and return the last
/// position's logits.
fn decode_logits<B: PlanBackend>(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    backend: &B,
) -> Vec<f32> {
    let mut session = DecodeSession::new(plan, store, backend).unwrap();
    let mut last = None;
    for &token in G_TOKENS.iter() {
        last = session.step(token).unwrap().logits;
    }
    assert_eq!(session.position(), G_TOKENS.len());
    last.expect("plan carries an output head")
}

fn assert_bit_exact<B: PlanBackend>(backend: &B) {
    let (_c, plan, store) = fixture();
    let batch = execute_plan(&plan, &store, &G_TOKENS, backend).unwrap();
    let stepped = decode_logits(&plan, &store, backend);
    assert_eq!(
        batch.logits.as_deref(),
        Some(stepped.as_slice()),
        "{}: decode-session logits differ from the batch traversal",
        backend.name()
    );
}

#[test]
fn reference_decode_matches_the_batch_traversal_bit_for_bit() {
    assert_bit_exact(&ReferenceBackend::new());
}

#[test]
fn production_decode_matches_the_batch_traversal_bit_for_bit() {
    assert_bit_exact(&ProductionBackend::new());
}

#[test]
fn mxfp4_device_decode_matches_its_own_batch_traversal_bit_for_bit() {
    // The 32-aligned fixture: MXFP4's group constraint correctly
    // refuses the awkward hidden-12 miniature.
    let (_c, plan, store) = super::device::aligned_fixture();
    let backend = DevicePlanBackend::new(
        super::device::LoopDevice,
        "loop-device-mxfp4-decode",
        WeightFormat::Mxfp4,
    );
    let batch = execute_plan(&plan, &store, &G_TOKENS, &backend).unwrap();
    let stepped = decode_logits(&plan, &store, &backend);
    assert_eq!(batch.logits.as_deref(), Some(stepped.as_slice()));
}

#[test]
fn f16_device_decode_matches_its_own_batch_traversal_bit_for_bit() {
    // Same loop device the seam tests use; the point here is that the
    // step path and the batch path convert and consume the *same* f16
    // bytes, so even the lossy realisation self-agrees exactly.
    assert_bit_exact(&DevicePlanBackend::new(
        super::device::LoopDevice,
        "loop-device-f16-decode",
        WeightFormat::F16,
    ));
}
