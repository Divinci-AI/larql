//! Local decode inference (spec §11).
//!
//! Assembly, by design. Every hard question was answered upstream:
//!
//! ```text
//! adapter             which tensors and banks a layer needs
//! programme traversal which bank-region arrangements execute
//! this function       joins them and folds authority
//! ```
//!
//! It never consults kernel maturity — a Production kernel and the reference
//! path over the same bytes have identical fidelity, and admission does not
//! depend on either.
//!
//! # Failures aggregate
//!
//! A missing router at layer 12 and an unreadable norm at layer 19 are
//! independent facts, and reporting only the first would make fixing an index
//! an iterative guessing game. Resolution collects every failure whose
//! diagnosis does not depend on interpreting another.

use std::collections::BTreeMap;

use super::component::{ComponentUsability, SelectedComponent, SelectedTensor};
use super::coordinate::BankCoordinate;
use super::decode_requirements::{DecodeRequirements, LocalDecodeRequest, Requirement};
use super::operation::{OperationCapability, OperationFailure};
use super::plan::{OperationPlan, PlanChoice, QualifiedAlternative};
use super::traversal::BankCapabilityReport;

/// The physical components a profile resolved to.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResolvedDocumentSelection {
    /// Manifest-addressed tensors, keyed by their coordinate's description so
    /// lookup is by identity rather than by position.
    pub tensors: Vec<SelectedTensor>,
    /// Per-bank traversal results, already qualified.
    pub bank_reports: BTreeMap<BankCoordinate, BankCapabilityReport>,
}

impl ResolvedDocumentSelection {
    fn tensor(&self, coordinate_description: &str) -> Option<&SelectedTensor> {
        self.tensors
            .iter()
            .find(|t| t.coordinate.describe() == coordinate_description)
    }
}

/// Resolve one requirement, preserving every satisfied alternative.
///
/// Returns the first satisfied candidate for the plan, plus a failure built
/// from the *first declared* candidate when none succeed — declaration order
/// decides only which failure is reported, never which success is used.
fn resolve_requirement(
    selection: &ResolvedDocumentSelection,
    requirement: &Requirement,
) -> Result<SelectedComponent, OperationFailure> {
    let mut first_failure: Option<OperationFailure> = None;

    for candidate in requirement.candidates() {
        let key = candidate.coordinate.describe();
        let Some(tensor) = selection.tensor(&key) else {
            first_failure.get_or_insert_with(|| OperationFailure::MissingDocumentInput {
                what: requirement.purpose(),
            });
            continue;
        };
        let usability = tensor.usability(candidate.contract.as_ref());
        if usability.is_usable() {
            return Ok(SelectedComponent::ManifestTensor(tensor.clone()));
        }
        first_failure.get_or_insert_with(|| failure_for(requirement, &key, &usability));
    }

    Err(
        first_failure.unwrap_or(OperationFailure::MissingDocumentInput {
            what: requirement.purpose(),
        }),
    )
}

fn failure_for(
    requirement: &Requirement,
    coordinate: &str,
    usability: &ComponentUsability,
) -> OperationFailure {
    OperationFailure::InvalidSelection {
        detail: format!(
            "{} ({coordinate}): {}",
            requirement.purpose(),
            usability.describe()
        ),
    }
}

/// Infer whether the selection can run a complete forward pass locally.
pub fn infer_local_decode(
    selection: &ResolvedDocumentSelection,
    requirements: &DecodeRequirements,
    _request: &LocalDecodeRequest,
) -> OperationCapability {
    let mut fixed_components = Vec::new();
    let mut failures = Vec::new();

    // ── Model-global requirements ──
    for requirement in &requirements.fixed_components {
        match resolve_requirement(selection, requirement) {
            Ok(component) => fixed_components.push(component),
            Err(f) => failures.push(f),
        }
    }

    // ── Per-layer fixed path ──
    //
    // Collected across every layer before returning, so one report can name
    // layer 12's missing router and layer 19's unreadable norm together.
    for layer in &requirements.layers {
        for requirement in layer.fixed() {
            match resolve_requirement(selection, requirement) {
                Ok(component) => fixed_components.push(component),
                Err(f) => failures.push(f),
            }
        }
    }

    // ── Expert banks, from traversal ──
    let mut choices = Vec::new();
    for bank in requirements.all_banks() {
        match selection.bank_reports.get(&bank) {
            Some(report) if report.is_executable() => {
                choices.push(PlanChoice {
                    bank,
                    alternatives: report
                        .successful_alternatives()
                        .iter()
                        .map(|a| QualifiedAlternative {
                            alternative: a.alternative,
                            regions: Vec::new(),
                        })
                        .collect(),
                });
            }
            Some(report) => {
                let detail = report
                    .closest_failure()
                    .map(|a| a.reference_execution.describe())
                    .unwrap_or_else(|| "no executable alternative".into());
                failures.push(OperationFailure::RequiredRegionUnusable {
                    coordinate: super::coordinate::RegionCoordinate::new(
                        bank.layer,
                        bank.bank_id,
                        None,
                        crate::format::lyrw2::region_role::RegionRole::Down,
                    ),
                    cause: detail,
                });
            }
            None => failures.push(OperationFailure::NoExecutableRoute { layer: bank.layer }),
        }
    }

    if !failures.is_empty() {
        return OperationCapability::unavailable(failures);
    }

    OperationCapability::available(vec![super::plan::QualifiedOperationRoute::new(
        OperationPlan {
            fixed_components,
            choices,
        },
    )])
}
