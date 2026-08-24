//! The ledger must count what ran, and reset to nothing.
//!
//! Every test here uses a LOCAL `ProjectionLedger` rather than the
//! process-wide one. The global is shared with every other test in the
//! suite and `reset` is destructive, so a test that reached for it would
//! be racing the thing it was trying to measure — and would pass or fail
//! depending on what else happened to be running.

use super::super::physical::PhysicalProjectionPlan;
use super::ProjectionLedger;

const PLANS: [PhysicalProjectionPlan; 3] = [
    PhysicalProjectionPlan::ScalarF32,
    PhysicalProjectionPlan::BlasF32,
    PhysicalProjectionPlan::FusedBf16,
];

#[test]
fn each_plan_is_counted_separately() {
    let l = ProjectionLedger::default();
    l.record(PhysicalProjectionPlan::FusedBf16, 1_000, 12);
    l.record(PhysicalProjectionPlan::FusedBf16, 2_000, 12);
    l.record(PhysicalProjectionPlan::BlasF32, 40, 1);

    let fused = l.get(PhysicalProjectionPlan::FusedBf16);
    assert_eq!(fused.calls, 2);
    assert_eq!(fused.bytes, 3_000);
    assert_eq!(fused.slabs, 24);

    let blas = l.get(PhysicalProjectionPlan::BlasF32);
    assert_eq!((blas.calls, blas.bytes, blas.slabs), (1, 40, 1));
    assert_eq!(l.get(PhysicalProjectionPlan::ScalarF32), Default::default());
    assert_eq!(l.total_bytes(), 3_040);
}

/// `all()` enumerates every plan, so a reader cannot silently stop
/// covering one.
#[test]
fn all_enumerates_every_plan() {
    let l = ProjectionLedger::default();
    for (i, plan) in PLANS.iter().enumerate() {
        l.record(*plan, i + 1, 1);
    }
    let seen: Vec<_> = l.all().iter().map(|(p, t)| (*p, t.bytes)).collect();
    assert_eq!(
        seen,
        vec![
            (PhysicalProjectionPlan::ScalarF32, 1),
            (PhysicalProjectionPlan::BlasF32, 2),
            (PhysicalProjectionPlan::FusedBf16, 3),
        ]
    );
    assert_eq!(l.total_bytes(), 6);
}

/// Reset zeroes every plan, not just the one that was busiest.
///
/// A partial reset is the failure that would matter: the CLI resets
/// before the step it prices, so a leftover count would silently fold the
/// weight load and every warm-up step into a per-token number.
#[test]
fn reset_clears_every_plan() {
    let l = ProjectionLedger::default();
    for plan in PLANS {
        l.record(plan, 7, 3);
    }
    assert_eq!(l.total_bytes(), 21);
    l.reset();
    for plan in PLANS {
        assert_eq!(
            l.get(plan),
            Default::default(),
            "{plan:?} survived the reset"
        );
    }
    assert_eq!(l.total_bytes(), 0);
}

/// The process-wide ledger exists and is the same one every time.
#[test]
fn the_shared_ledger_is_one_ledger() {
    assert!(std::ptr::eq(super::ledger(), super::ledger()));
}
