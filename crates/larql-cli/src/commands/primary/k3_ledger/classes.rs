//! Per-tensor-class byte census and composed throughput ceiling.
//!
//! Pure. Every ceiling quoted before this module existed divided total bytes by
//! a single `eta`, which is wrong in a way that flatters: `eta` is a property of
//! a *kernel*, and different tensor classes run through different kernels with
//! measured efficiencies from 0.59 to 1.00. K3's attention projections are the
//! largest dense term AND run through the worst-measured class, so a scalar
//! quote overstates by ~1.17x.
//!
//! The composed form is a sum of per-class times, not a single division:
//!
//! ```text
//!   t_token = sum_c  bytes_c / (BW * eta_c)
//!   tok/s   = 1 / t_token
//! ```
//!
//! This module also provides the R4 discipline in scenario form: for a proposed
//! change, zero out or improve exactly what it targets and recompute. A lever
//! whose best case still sits under the goal cannot decide the goal, and finding
//! that out costs one division rather than a fortnight.

use serde::Serialize;

use super::geometry::K3Geometry;

/// Measured attainable GPU bandwidth (`project_memory_bandwidth_roofline`).
pub const BW_GB_S: f64 = 367.0;

/// A kernel class, with the efficiency measured for it by the batched profiler.
///
/// **MEASURED at K3 shapes under Q6K** as of `diag_shape_census`, which
/// resolved a confound worth recording. The composed ledger previously assigned
/// `q4k_matvec`'s 0.59 to the attention class — but crossing kernel against
/// shape showed that figure is **kernel-borne, not shape-borne**: `q6k_matvec`
/// reaches 0.87-0.90 at *every* large shape including K3's `[12288, 7168]`
/// attention, while `q4k_matvec` sits at 0.65-0.75 on the same geometry. So a
/// Q6K-transcoded image sidesteps the Wo problem entirely and attention runs at
/// 0.89, not 0.59.
///
/// The same run found the opposite effect where nobody was looking: K3's expert
/// branch `[3584, 3072]` gets only **0.64** from the identical q6k kernel. That
/// one *is* shape-borne — 896 threadgroups versus attention's 3072 leaves too
/// little work to hide latency — and it lands on the single largest byte term,
/// so routed experts now consume over half the time budget.
///
/// Historical note on the superseded figures. They came from
/// `diag_profile_kernels` batched at 34x per command buffer, which is the right
/// instrument — isolated single-dispatch numbers for the same kernels run
/// 3.0-6.7x lower and must never be substituted. But they were measured:
///
///   1. on **Gemma-class shapes** (2560x8192, 2560x10240, 10240x2560), not K3's
///      ([12288, 7168] attention, [3584, 3072] expert branches); and
///   2. in **Q4K / Q6K**, not in whatever format the image actually serves.
///
/// So applying them to a K3 census conflates tensor shape with kernel format
/// twice over. A ceiling computed from them is a *provisional composed ceiling*
/// and must be labelled as such until the real K3 shapes run through the same
/// batched, cold-rotating harness. `Eta::provenance` carries that condition so
/// it cannot be quoted bare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum KernelClass {
    /// `q4k_matvec` (Wo-shaped). Worst measured.
    AttnProjection,
    /// `q4k_ffn_gate_up_8sg`. Compute-bound on dequant at K=2560.
    GateUp,
    /// `q6k_matvec` at a large shape. Bandwidth-bound, best measured class.
    Down,
    /// `q6k_matvec` at K3's small expert-branch shape `[3584, 3072]`. The same
    /// kernel, measurably worse purely from geometry: 896 threadgroups against
    /// the attention shape's 3072, so there is far less work to hide latency
    /// behind. This is a SHAPE-borne penalty on the largest byte term.
    RoutedExpert,
    /// `f32_gemv`. Un-quantised, reaches the roofline.
    Unquantised,
}

impl KernelClass {
    /// What was actually measured, and under what conditions.
    pub fn provenance(self) -> &'static str {
        match self {
            Self::AttnProjection => "q6k_matvec, Q6K, measured at K3 KDA 12288x7168",
            Self::GateUp => "q4k_ffn_gate_up_8sg 10240x2560, Q4K, Gemma shape",
            Self::Down => "q6k_matvec, Q6K, measured at K3 latent-up 7168x3584",
            Self::RoutedExpert => {
                "q6k_matvec ungrouped, K3 expert 3584x3072 (K3a grouped measures 0.89)"
            }
            Self::Unquantised => "f32_gemv lm_head 262Kx2560, f32, Gemma shape",
        }
    }

    /// True while this class's eta is a borrowed proxy rather than a K3-shape,
    /// K3-format measurement.
    ///
    /// Only `GateUp` still is: no `[6144, 7168]` shape was measured directly, so
    /// it borrows the nearest large-shape q6k figure. Everything else now comes
    /// from `diag_shape_census` at the real K3 geometry under Q6K.
    pub fn is_provisional(self) -> bool {
        matches!(self, Self::GateUp)
    }

    pub fn eta(self) -> f64 {
        match self {
            Self::AttnProjection => 0.89,
            Self::GateUp => 0.87,
            Self::Down => 0.87,
            Self::RoutedExpert => 0.64,
            Self::Unquantised => 1.00,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::AttnProjection => "attn-projection (Wo class)",
            Self::GateUp => "gate/up class",
            Self::Down => "down class",
            Self::RoutedExpert => "routed-expert shape (small)",
            Self::Unquantised => "unquantised",
        }
    }
}

/// One row of the census: a group of tensors sharing a kernel class and format.
#[derive(Debug, Clone, Serialize)]
pub struct ClassRow {
    pub name: &'static str,
    pub params: u64,
    pub class: KernelClass,
    /// All-in bits per weight for this class in the current image.
    pub bits: f64,
    /// Efficiency override, when a scenario improves this class specifically.
    pub eta_override: Option<f64>,
}

impl ClassRow {
    pub fn bytes(&self) -> f64 {
        self.params as f64 * self.bits / 8.0
    }
    pub fn eta(&self) -> f64 {
        self.eta_override.unwrap_or_else(|| self.class.eta())
    }
    pub fn seconds(&self, bw: f64) -> f64 {
        self.bytes() / (bw * 1e9 * self.eta())
    }
}

/// Efficiency the routed-expert class reaches once experts are put on the
/// dispatch grid's y axis — MEASURED by K3a, and inside the large-shape band.
pub const GROUPED_ROUTED_ETA: f64 = 0.89;

/// MXFP4 all-in bits (codeword + e8m0 scale per 32).
pub const MXFP4_BITS: f64 = 4.25;
/// Q6_K all-in bits (210 bytes per 256 weights).
pub const Q6K_BITS: f64 = 210.0 * 8.0 / 256.0;

/// The per-class census for a K3 image at a given format.
///
/// Parameter counts are derived from the measured geometry rather than assumed:
/// a KDA layer carries five `[12288, 7168]` attention projections, an MLA layer
/// replaces three of them with the smaller q_a/q_b/kv_a/kv_b set, and every
/// layer carries shared experts, the LatentMoE projections and a router.
pub fn census(geom: &K3Geometry, dense_bits: f64, routed_bits: f64) -> Vec<ClassRow> {
    let h = geom.hidden_size as u64;
    let kda = geom.n_kda_layers as u64;
    let mla = geom.n_mla_layers as u64;
    let moe = geom.n_moe_layers() as u64;

    // KDA attention: q, k, v, g, o each [12288, 7168].
    let kda_attn = kda * 5 * 12288 * h;
    // MLA attention: g + o at full width, plus the compressed q/kv set.
    let mla_attn = mla * (2 * 12288 * h + 1536 * h + 18432 * 1536 + 576 * h + 24576 * 512);
    // Shared experts: gate, up, down at [6144, 7168].
    let shared = moe * 3 * 6144 * h;
    // LatentMoE wrapper: down + up at [3584, 7168].
    let latent = moe * 2 * 3584 * h;
    let router = moe * geom.n_experts as u64 * h;

    vec![
        ClassRow {
            name: "KDA attention projections",
            params: kda_attn,
            class: KernelClass::AttnProjection,
            bits: dense_bits,
            eta_override: None,
        },
        ClassRow {
            name: "MLA attention projections",
            params: mla_attn,
            class: KernelClass::AttnProjection,
            bits: dense_bits,
            eta_override: None,
        },
        ClassRow {
            name: "shared experts",
            params: shared,
            class: KernelClass::GateUp,
            bits: dense_bits,
            eta_override: None,
        },
        ClassRow {
            name: "LatentMoE projections",
            params: latent,
            class: KernelClass::Down,
            bits: dense_bits,
            eta_override: None,
        },
        ClassRow {
            name: "router",
            params: router,
            class: KernelClass::Down,
            bits: dense_bits,
            eta_override: None,
        },
        ClassRow {
            name: "embeddings + lm_head",
            params: geom.embedding_params,
            class: KernelClass::Unquantised,
            bits: 16.0,
            eta_override: None,
        },
        ClassRow {
            name: "routed experts (top-k)",
            params: geom.routed_activated_params(),
            class: KernelClass::RoutedExpert,
            bits: routed_bits,
            eta_override: None,
        },
    ]
}

#[derive(Debug, Clone, Serialize)]
pub struct ComposedCeiling {
    pub total_bytes: f64,
    pub seconds_per_token: f64,
    pub tok_s: f64,
    /// What a single-eta quote would have claimed, and by how much it overstates.
    pub scalar_tok_s: f64,
    pub scalar_overstates_by: f64,
    pub rows: Vec<ClassRow>,
}

/// Compose per-class times into one throughput ceiling.
pub fn compose(rows: Vec<ClassRow>, bw: f64, scalar_eta: f64) -> ComposedCeiling {
    let total: f64 = rows.iter().map(ClassRow::bytes).sum();
    let secs: f64 = rows.iter().map(|r| r.seconds(bw)).sum();
    let scalar = bw * 1e9 * scalar_eta / total;
    ComposedCeiling {
        total_bytes: total,
        seconds_per_token: secs,
        tok_s: 1.0 / secs,
        scalar_tok_s: scalar,
        scalar_overstates_by: scalar * secs,
        rows,
    }
}

/// **R4 in scenario form.** Apply a best-case change and recompute the ceiling.
///
/// Each variant expresses what one proposed experiment would achieve *if it
/// fully succeeded* — not a forecast, a ceiling. Anything whose best case sits
/// under the goal cannot decide the goal.
#[derive(Debug, Clone, Copy)]
pub enum Scenario {
    /// Every class runs at the best measured quantised efficiency.
    AllClassesAtBestEta,
    /// One class's efficiency is lifted to a target.
    LiftClass(KernelClass, f64),
    /// One named row's format changes to a new bit width.
    Rebits(&'static str, f64),
    /// A named row's bytes vanish entirely — the strict R4 zero-out.
    Zero(&'static str),
}

/// Apply several scenarios in sequence — the stacked best case.
///
/// Single levers are informative but no proposal ships alone; the programme
/// question is what a *combination* reaches, and whether the combination clears
/// the target that any one of them misses.
pub fn apply_all(rows: &[ClassRow], scenarios: &[Scenario]) -> Vec<ClassRow> {
    scenarios
        .iter()
        .fold(rows.to_vec(), |acc, s| apply(&acc, *s))
}

pub fn apply(rows: &[ClassRow], s: Scenario) -> Vec<ClassRow> {
    rows.iter()
        .map(|r| {
            let mut r = r.clone();
            match s {
                Scenario::AllClassesAtBestEta => {
                    if r.class != KernelClass::Unquantised {
                        r.eta_override = Some(KernelClass::Down.eta());
                    }
                }
                Scenario::LiftClass(c, eta) => {
                    if r.class == c {
                        r.eta_override = Some(eta);
                    }
                }
                Scenario::Rebits(name, bits) => {
                    if r.name == name {
                        r.bits = bits;
                    }
                }
                Scenario::Zero(name) => {
                    if r.name == name {
                        r.bits = 0.0;
                    }
                }
            }
            r
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::primary::k3_ledger::geometry::k3_reference;

    fn base() -> Vec<ClassRow> {
        census(&k3_reference(), MXFP4_BITS, MXFP4_BITS)
    }

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ~{b}, got {a}");
    }

    #[test]
    fn census_reproduces_the_measured_activated_total() {
        // Independent path to the same 104.83 B figure the ledger derives from
        // per-layer headers — a cross-check that the class split is complete.
        let total: u64 = base().iter().map(|r| r.params).sum();
        approx(total as f64 / 1e9, 104.83, 1.5);
    }

    #[test]
    fn attention_is_the_largest_dense_class() {
        let rows = base();
        let attn: u64 = rows
            .iter()
            .filter(|r| r.class == KernelClass::AttnProjection)
            .map(|r| r.params)
            .sum();
        let others: u64 = rows
            .iter()
            .filter(|r| {
                r.class != KernelClass::AttnProjection && r.name != "routed experts (top-k)"
            })
            .map(|r| r.params)
            .sum();
        assert!(attn > others, "attention {attn} vs other dense {others}");
    }

    #[test]
    fn composing_is_stricter_than_a_scalar_quote() {
        // A single eta flatters whenever any class sits below it, and one
        // always does. Quoting the best-class eta over total bytes is the
        // specific mistake this module exists to prevent.
        let c = compose(base(), BW_GB_S, KernelClass::Down.eta());
        assert!(c.scalar_overstates_by > 1.0, "{}", c.scalar_overstates_by);
        assert!(c.tok_s < c.scalar_tok_s);
    }

    #[test]
    fn the_routed_expert_shape_is_now_the_worst_class() {
        // Reordered by measurement: before diag_shape_census the attention
        // class held 0.59 and routed experts a borrowed 0.85. Crossing kernel
        // against shape moved attention to 0.89 (its 0.59 was q4k's problem,
        // not the geometry's) and routed experts DOWN to 0.64 (896 threadgroups
        // against attention's 3072). The worst class changed hands.
        assert!(KernelClass::RoutedExpert.eta() < KernelClass::AttnProjection.eta());
        assert!(KernelClass::RoutedExpert.eta() < KernelClass::Down.eta());
    }

    #[test]
    fn fixing_the_routed_expert_shape_is_the_largest_single_lever() {
        // Consequence of the reordering: routed experts carry over half the
        // time budget at the worst eta, so lifting THAT class beats lifting any
        // other. This test is the programme's ordering, made executable.
        let rows = base();
        let ceiling = |c: KernelClass| {
            compose(apply(&rows, Scenario::LiftClass(c, 0.89)), BW_GB_S, 0.87).tok_s
        };
        let routed = ceiling(KernelClass::RoutedExpert);
        for other in [
            KernelClass::AttnProjection,
            KernelClass::GateUp,
            KernelClass::Down,
        ] {
            assert!(
                routed > ceiling(other),
                "lifting routed ({routed}) should beat lifting {:?} ({})",
                other,
                ceiling(other)
            );
        }
    }

    #[test]
    fn routed_experts_dominate_the_time_budget() {
        let c = compose(base(), BW_GB_S, 0.87);
        let routed = c
            .rows
            .iter()
            .find(|r| r.name == "routed experts (top-k)")
            .unwrap()
            .seconds(BW_GB_S);
        assert!(
            routed / c.seconds_per_token > 0.5,
            "routed share {:.2} should exceed half",
            routed / c.seconds_per_token
        );
    }

    #[test]
    fn zeroing_the_routed_bank_leaves_the_dense_ceiling() {
        // R4: no expert-side lever can lift a target above this.
        let c = compose(
            apply(&base(), Scenario::Zero("routed experts (top-k)")),
            BW_GB_S,
            0.85,
        );
        assert!(
            c.tok_s < 20.0,
            "dense-only ceiling {} should refuse 20",
            c.tok_s
        );
    }

    #[test]
    fn q6k_transcode_costs_bytes_and_therefore_throughput() {
        let mxfp4 = compose(base(), BW_GB_S, 0.85);
        let q6k = compose(census(&k3_reference(), Q6K_BITS, Q6K_BITS), BW_GB_S, 0.85);
        assert!(q6k.total_bytes > mxfp4.total_bytes);
        assert!(q6k.tok_s < mxfp4.tok_s);
        approx(Q6K_BITS / MXFP4_BITS, 1.544, 0.01);
    }

    #[test]
    fn three_bit_kda_attention_is_a_real_lever() {
        // Exp 3's target: the largest dense class at 3 bits instead of 4.25.
        let rows = base();
        let before = compose(rows.clone(), BW_GB_S, 0.85);
        let after = compose(
            apply(&rows, Scenario::Rebits("KDA attention projections", 3.0)),
            BW_GB_S,
            0.85,
        );
        assert!(after.tok_s > before.tok_s * 1.05, "should move the needle");
    }
}
