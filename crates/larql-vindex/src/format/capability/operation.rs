//! Operation admission (spec §9, §11, §15).
//!
//! Three questions, kept apart:
//!
//! ```text
//! Can this operation run?          → admission (here)
//! How faithfully would it run?     → authority, folded over the regions THIS
//!                                    operation consumes
//! Which implementation runs it?    → kernel binding, elsewhere
//! ```
//!
//! Admission is inferred first and authority attached afterwards, because the
//! two answer different questions and one of them has no answer when the
//! selection is contradictory.
//!
//! # The decisive property: non-interference
//!
//! Changing a component an operation does not use must not change that
//! operation's capability. WALK reads gate rows; making every `down` region
//! unreadable must leave the WALK report byte-for-byte identical. That is what
//! proves traversal facts are *projected per operation* rather than globally
//! summarised, and it is why each operation declares a narrow dependency
//! surface rather than consulting a shared "is the index healthy" verdict.

use super::authority::DerivedAuthority;
use super::coordinate::RegionCoordinate;

/// Why an operation cannot run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationFailure {
    /// A region this operation needs is unusable, with its exact coordinate.
    RequiredRegionUnusable {
        coordinate: RegionCoordinate,
        cause: String,
    },
    /// The selection is self-contradictory. Fails closed — never downgraded
    /// into a weak authority.
    InvalidSelection { detail: String },
    /// A document-level input this operation needs is absent.
    MissingDocumentInput { what: &'static str },
    /// The operation needs a policy declaration the caller did not supply.
    /// Notably: remote execution cannot be inferred from local absence.
    MissingContract { what: &'static str },
    /// No layer offers an executable route for this operation.
    NoExecutableRoute { layer: u32 },
}

impl OperationFailure {
    pub fn describe(&self) -> String {
        match self {
            Self::RequiredRegionUnusable { coordinate, cause } => {
                format!("{}: {cause}", coordinate.describe())
            }
            Self::InvalidSelection { detail } => format!("invalid selection: {detail}"),
            Self::MissingDocumentInput { what } => format!("missing document input: {what}"),
            Self::MissingContract { what } => {
                format!("no declared contract for {what}; it cannot be inferred")
            }
            Self::NoExecutableRoute { layer } => {
                format!("layer {layer} has no executable route")
            }
        }
    }
}

/// Something absent that reduces richness without preventing the operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Degradation {
    /// Optional query metadata is absent. Reduces label richness; never
    /// affects correctness (§15.3).
    MissingQueryMetadata { what: &'static str },
    /// A bank cannot be browsed, so its features are outside query reach while
    /// other banks remain reachable.
    BankNotBrowsable { bank_id: u16, reason: String },
}

impl Degradation {
    pub fn describe(&self) -> String {
        match self {
            Self::MissingQueryMetadata { what } => {
                format!("{what} absent — reduced richness, unchanged correctness")
            }
            Self::BankNotBrowsable { bank_id, reason } => {
                format!("bank {bank_id} not browsable: {reason}")
            }
        }
    }
}

/// What an operation would actually use if it ran.
///
/// Carrying the region list is what lets authority be folded over *this
/// operation's* regions rather than the whole index.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OperationPlan {
    /// Regions this operation consumes. The authority fold's domain.
    pub regions: Vec<RegionCoordinate>,
    /// Layers this operation touches.
    pub layers: Vec<u32>,
}

impl OperationPlan {
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
}

/// Whether an operation can run, and how completely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationAdmission {
    Available(OperationPlan),
    Degraded {
        plan: OperationPlan,
        reasons: Vec<Degradation>,
    },
    Unavailable {
        reasons: Vec<OperationFailure>,
    },
}

impl OperationAdmission {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available(_) | Self::Degraded { .. })
    }

    pub fn plan(&self) -> Option<&OperationPlan> {
        match self {
            Self::Available(p) | Self::Degraded { plan: p, .. } => Some(p),
            Self::Unavailable { .. } => None,
        }
    }

    /// Whether admission failed because the selection is contradictory, as
    /// opposed to incomplete. Contradictions must fail closed.
    pub fn is_invalid_selection(&self) -> bool {
        match self {
            Self::Unavailable { reasons } => reasons
                .iter()
                .any(|r| matches!(r, OperationFailure::InvalidSelection { .. })),
            _ => false,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Available(_) => "available".into(),
            Self::Degraded { reasons, .. } => format!(
                "available, degraded: {}",
                reasons
                    .iter()
                    .map(|r| r.describe())
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
            Self::Unavailable { reasons } => format!(
                "unavailable: {}",
                reasons
                    .iter()
                    .map(|r| r.describe())
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        }
    }
}

/// Admission plus the fidelity of the regions this operation actually reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationCapability {
    pub admission: OperationAdmission,
    /// `None` when there is nothing honest to say.
    ///
    /// Two cases, and the second is the subtle one. An unavailable operation
    /// has no fidelity because it does not run. A **contradictory selection**
    /// also has none — and must not be handed `analysis-only` or
    /// `structurally-approximate` merely because those are low lattice values.
    /// Assigning a weak-but-valid authority to an invalid selection would
    /// launder a refusal into a downgrade, which is exactly what the
    /// fail-closed rule exists to prevent.
    pub authority: Option<DerivedAuthority>,
}

impl OperationCapability {
    pub fn unavailable(reasons: Vec<OperationFailure>) -> Self {
        Self {
            admission: OperationAdmission::Unavailable { reasons },
            authority: None,
        }
    }

    pub fn is_available(&self) -> bool {
        self.admission.is_available()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::capability::authority::{AuthorityInputs, Fidelity};
    use crate::format::lyrw2::region_role::RegionRole;

    fn coordinate() -> RegionCoordinate {
        RegionCoordinate::new(3, 0, Some(1), RegionRole::Down)
    }

    fn plan() -> OperationPlan {
        OperationPlan {
            regions: vec![coordinate()],
            layers: vec![3],
        }
    }

    #[test]
    fn available_and_degraded_both_admit() {
        assert!(OperationAdmission::Available(plan()).is_available());
        assert!(OperationAdmission::Degraded {
            plan: plan(),
            reasons: vec![Degradation::MissingQueryMetadata { what: "labels" }],
        }
        .is_available());
    }

    #[test]
    fn unavailable_does_not_admit_and_has_no_plan() {
        let a = OperationAdmission::Unavailable {
            reasons: vec![OperationFailure::NoExecutableRoute { layer: 3 }],
        };
        assert!(!a.is_available());
        assert!(a.plan().is_none());
    }

    #[test]
    fn a_degraded_operation_still_carries_its_plan() {
        // Degradation reduces richness, not reach — the regions are still read.
        let a = OperationAdmission::Degraded {
            plan: plan(),
            reasons: vec![Degradation::MissingQueryMetadata { what: "down_meta" }],
        };
        assert_eq!(a.plan(), Some(&plan()));
    }

    #[test]
    fn an_invalid_selection_gets_no_authority_rather_than_a_weak_one() {
        // The subtle case: analysis-only is a *valid* low fidelity. Handing it
        // to a contradictory selection would launder a refusal into a
        // downgrade, and a caller checking "is this at least analysis-only?"
        // would wrongly proceed.
        let c = OperationCapability::unavailable(vec![OperationFailure::InvalidSelection {
            detail: "gate covers 0, down covers 1".into(),
        }]);
        assert_eq!(c.authority, None);
        assert!(c.admission.is_invalid_selection());
        assert!(!c.is_available());
    }

    #[test]
    fn an_ordinary_unavailability_is_not_an_invalid_selection() {
        let c = OperationCapability::unavailable(vec![OperationFailure::MissingDocumentInput {
            what: "tokenizer",
        }]);
        assert!(!c.admission.is_invalid_selection());
    }

    #[test]
    fn a_missing_contract_is_distinct_from_missing_bytes() {
        // Remote execution cannot be inferred from local absence; the two must
        // never read alike.
        let contract = OperationFailure::MissingContract {
            what: "routed wire contract",
        };
        let bytes = OperationFailure::RequiredRegionUnusable {
            coordinate: coordinate(),
            cause: "absent".into(),
        };
        assert!(contract.describe().contains("cannot be inferred"));
        assert!(!bytes.describe().contains("cannot be inferred"));
    }

    #[test]
    fn failures_carry_exact_coordinates() {
        let f = OperationFailure::RequiredRegionUnusable {
            coordinate: coordinate(),
            cause: "absent from every selected segment".into(),
        };
        let s = f.describe();
        assert!(s.contains("layer 3"), "{s}");
        assert!(s.contains("bank 0"), "{s}");
        assert!(s.contains("role down"), "{s}");
        assert!(s.contains("segment 1"), "{s}");
    }

    #[test]
    fn missing_query_metadata_says_correctness_is_unchanged() {
        // §15.3: absent query metadata downgrades richness, never WALK
        // correctness. The wording matters — an operator must not read it as
        // "results may be wrong".
        let d = Degradation::MissingQueryMetadata {
            what: "feature_labels.json",
        };
        assert!(
            d.describe().contains("unchanged correctness"),
            "{}",
            d.describe()
        );
    }

    #[test]
    fn an_available_operation_can_carry_a_derived_authority() {
        let c = OperationCapability {
            admission: OperationAdmission::Available(plan()),
            authority: Some(DerivedAuthority::of(&AuthorityInputs {
                selected_fidelities: vec![Fidelity::SourceExact],
                execution_complete: true,
                structural: None,
            })),
        };
        assert!(c.is_available());
        assert_eq!(c.authority.unwrap().level, Fidelity::SourceExact);
    }

    #[test]
    fn a_non_browsable_bank_degrades_rather_than_failing() {
        // Other banks stay reachable, so the operation runs with reduced reach.
        let d = Degradation::BankNotBrowsable {
            bank_id: 4,
            reason: "fused region cannot be strided".into(),
        };
        let s = d.describe();
        assert!(s.contains("bank 4"), "{s}");
        assert!(s.contains("cannot be strided"), "{s}");
    }

    #[test]
    fn every_failure_kind_renders_distinguishably() {
        let rendered: Vec<String> = [
            OperationFailure::RequiredRegionUnusable {
                coordinate: coordinate(),
                cause: "absent".into(),
            },
            OperationFailure::InvalidSelection {
                detail: "disjoint segments".into(),
            },
            OperationFailure::MissingDocumentInput { what: "embeddings" },
            OperationFailure::MissingContract {
                what: "routed wire",
            },
            OperationFailure::NoExecutableRoute { layer: 7 },
        ]
        .iter()
        .map(|f| f.describe())
        .collect();
        // No two failures should read alike — an operator triages from these.
        for i in 0..rendered.len() {
            for j in (i + 1)..rendered.len() {
                assert_ne!(rendered[i], rendered[j]);
            }
        }
        assert!(rendered[1].contains("invalid selection"));
        assert!(rendered[2].contains("embeddings"));
        assert!(rendered[4].contains("layer 7"));
    }

    #[test]
    fn admission_descriptions_state_the_verdict_first() {
        assert!(OperationAdmission::Available(plan())
            .describe()
            .starts_with("available"));
        assert!(OperationAdmission::Degraded {
            plan: plan(),
            reasons: vec![Degradation::MissingQueryMetadata { what: "labels" }],
        }
        .describe()
        .starts_with("available, degraded"));
        assert!(OperationAdmission::Unavailable {
            reasons: vec![OperationFailure::NoExecutableRoute { layer: 0 }],
        }
        .describe()
        .starts_with("unavailable"));
    }

    #[test]
    fn an_available_operation_is_never_an_invalid_selection() {
        assert!(!OperationAdmission::Available(plan()).is_invalid_selection());
        assert!(!OperationAdmission::Degraded {
            plan: plan(),
            reasons: Vec::new(),
        }
        .is_invalid_selection());
    }

    #[test]
    fn a_plan_names_the_regions_authority_will_fold_over() {
        // The whole point of carrying a plan: authority is per-operation, so
        // its domain has to be the operation's own regions.
        assert_eq!(plan().regions, vec![coordinate()]);
        assert!(!plan().is_empty());
        assert!(OperationPlan::default().is_empty());
    }
}
