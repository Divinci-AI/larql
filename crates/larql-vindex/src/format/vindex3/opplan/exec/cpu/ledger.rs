//! What the executor ACTUALLY ran — the counterpart to what the loader
//! decided.
//!
//! The residency census reads the loader's own bookkeeping, so on its own
//! it cannot fail the way that matters: a census can report 51 GB compact
//! while every projection quietly widens a tile before computing. The two
//! instruments answer different questions and only agree if both are true.
//!
//! Global rather than per backend, for the reason the pool is:
//! `ProductionBackend` is a zero-sized value that call sites construct
//! freely, so per-instance counters would each see a fraction of a decode
//! and none of them the whole.
//!
//! Cost is two relaxed atomic adds per projection against roughly 400
//! projections and 51 GB of streaming per token — unmeasurable, so it is
//! always on rather than behind a feature that would be off exactly when
//! a number needed explaining.

use std::sync::atomic::{AtomicU64, Ordering};

use super::physical::PhysicalProjectionPlan;

/// One plan's tally.
#[derive(Default)]
struct Tally {
    calls: AtomicU64,
    bytes: AtomicU64,
    /// Row slabs handed to workers. Equal to `calls` for an unpartitioned
    /// kernel, and `calls * workers` for a fully fanned-out one — which is
    /// what makes per-dispatch overhead visible in a decode rather than
    /// only in a bench.
    slabs: AtomicU64,
}

impl Tally {
    fn snapshot(&self) -> PlanTally {
        PlanTally {
            calls: self.calls.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            slabs: self.slabs.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
        self.slabs.store(0, Ordering::Relaxed);
    }
}

/// One plan's tally, read out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanTally {
    pub calls: u64,
    /// Weight bytes read in the representation they were resident as —
    /// directly comparable across plans, and the quantity the roofline is
    /// stated in.
    pub bytes: u64,
    pub slabs: u64,
}

/// Every projection the CPU executor has run, by plan.
#[derive(Default)]
pub struct ProjectionLedger {
    scalar: Tally,
    blas: Tally,
    fused: Tally,
}

impl ProjectionLedger {
    fn tally(&self, plan: PhysicalProjectionPlan) -> &Tally {
        match plan {
            PhysicalProjectionPlan::ScalarF32 => &self.scalar,
            PhysicalProjectionPlan::BlasF32 => &self.blas,
            PhysicalProjectionPlan::FusedBf16 => &self.fused,
        }
    }

    pub(super) fn record(&self, plan: PhysicalProjectionPlan, bytes: usize, slabs: usize) {
        let t = self.tally(plan);
        t.calls.fetch_add(1, Ordering::Relaxed);
        t.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        t.slabs.fetch_add(slabs as u64, Ordering::Relaxed);
    }

    pub fn get(&self, plan: PhysicalProjectionPlan) -> PlanTally {
        self.tally(plan).snapshot()
    }

    /// Every plan, so a reader enumerates rather than remembers. A caller
    /// that listed the plans itself would stop covering a new one on the
    /// day it was added.
    pub fn all(&self) -> [(PhysicalProjectionPlan, PlanTally); 3] {
        [
            PhysicalProjectionPlan::ScalarF32,
            PhysicalProjectionPlan::BlasF32,
            PhysicalProjectionPlan::FusedBf16,
        ]
        .map(|p| (p, self.get(p)))
    }

    /// Weight bytes across every plan — what one decode step streamed.
    pub fn total_bytes(&self) -> u64 {
        self.all().iter().map(|(_, t)| t.bytes).sum()
    }

    /// Zero the counters, so a caller can price ONE step.
    ///
    /// Nothing here is per session, so a reader that forgot this would be
    /// measuring the weight load and every warm-up step as well.
    pub fn reset(&self) {
        self.scalar.reset();
        self.blas.reset();
        self.fused.reset();
    }
}

static LEDGER: ProjectionLedger = ProjectionLedger {
    scalar: Tally {
        calls: AtomicU64::new(0),
        bytes: AtomicU64::new(0),
        slabs: AtomicU64::new(0),
    },
    blas: Tally {
        calls: AtomicU64::new(0),
        bytes: AtomicU64::new(0),
        slabs: AtomicU64::new(0),
    },
    fused: Tally {
        calls: AtomicU64::new(0),
        bytes: AtomicU64::new(0),
        slabs: AtomicU64::new(0),
    },
};

/// The process's projection ledger.
pub fn ledger() -> &'static ProjectionLedger {
    &LEDGER
}
