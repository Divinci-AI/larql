//! What this implementation can do with one required operand (spec §10, §11).
//!
//! Four states, and the distinctions between them must survive to the caller.
//! Collapsing them into a boolean is the specific failure §11 legislates
//! against: *representable-but-no-kernel* falls back to the reference executor
//! and is flagged, while *operand-absent* is a hard refusal. If those look the
//! same, the reference executor becomes a fallback for corruption rather than
//! for kernel immaturity — and an index that should have been refused gets
//! served slowly instead.
//!
//! The third state exists for the same reason unknown tags are preserved at
//! parse time (§6.5): bytes this binary cannot interpret are not missing
//! bytes. The index may be perfectly intact and simply newer than the reader.

use crate::format::lyrw2::region_format::{Packing, RegionFormat};

use super::coordinate::AbsenceKind;

/// How mature the execution path for an operand is (§10's ladder).
///
/// Deliberately excludes `Representable` and `Reference`: those are not kernel
/// bindings, and are carried by [`OperandCapability`] variants instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KernelMaturity {
    Grouped,
    Dispatched,
    Production,
}

impl KernelMaturity {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Grouped => "grouped",
            Self::Dispatched => "dispatched",
            Self::Production => "production",
        }
    }
}

/// Opaque handle to a registered kernel.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId(pub String);

impl KernelId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

/// What can be done with one required operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandCapability {
    /// The selected physical index cannot perform operations needing this
    /// region. Hard refusal — never a reference-executor fallback.
    Absent(AbsenceKind),

    /// Bytes are present and the index is structurally intact, but this
    /// implementation cannot interpret the codec or packing. Not a corrupt
    /// index; possibly a newer one.
    PresentUnsupported {
        format: RegionFormat,
        packing: Packing,
    },

    /// Correct execution is available through the generic path. Optimisation
    /// maturity is incomplete, which is a performance fact, not a fidelity one.
    ReferenceExecutable,

    /// A specialised kernel is available.
    KernelExecutable {
        maturity: KernelMaturity,
        kernel_id: KernelId,
    },
}

impl OperandCapability {
    /// Whether the operand can be computed at all, by any path.
    pub fn is_executable(&self) -> bool {
        matches!(
            self,
            Self::ReferenceExecutable | Self::KernelExecutable { .. }
        )
    }

    /// Whether the *bytes* are present, regardless of whether this binary can
    /// interpret them.
    ///
    /// Distinct from `is_executable`: an unsupported codec is present and
    /// unusable, which is the state that must not read as absent.
    pub fn bytes_present(&self) -> bool {
        !matches!(self, Self::Absent(_))
    }

    /// Whether this state indicates something wrong with the index.
    ///
    /// An operand omitted by the active selection is not a defect, and neither
    /// is an unsupported codec — the first is deliberate scoping, the second is
    /// a limitation of this binary rather than of the artifact.
    pub fn indicates_index_defect(&self) -> bool {
        match self {
            Self::Absent(kind) => kind.is_defect(),
            Self::PresentUnsupported { .. } => false,
            Self::ReferenceExecutable | Self::KernelExecutable { .. } => false,
        }
    }

    /// The kernel maturity backing this operand, if any. `None` covers both
    /// "reference only" and "not executable" — callers that care about the
    /// difference must ask `is_executable` too.
    pub fn maturity(&self) -> Option<KernelMaturity> {
        match self {
            Self::KernelExecutable { maturity, .. } => Some(*maturity),
            _ => None,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Absent(kind) => format!("absent — {}", kind.describe()),
            Self::PresentUnsupported { format, packing } => format!(
                "present but not interpretable by this build (format {}, packing {})",
                format.name(),
                packing.name()
            ),
            Self::ReferenceExecutable => {
                "executable through the reference path; no specialised kernel".into()
            }
            Self::KernelExecutable {
                maturity,
                kernel_id,
            } => format!(
                "executable by kernel '{}' ({})",
                kernel_id.0,
                maturity.name()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(maturity: KernelMaturity) -> OperandCapability {
        OperandCapability::KernelExecutable {
            maturity,
            kernel_id: KernelId::new("grouped_experts_q6k"),
        }
    }

    #[test]
    fn only_executable_states_are_executable() {
        assert!(OperandCapability::ReferenceExecutable.is_executable());
        assert!(kernel(KernelMaturity::Production).is_executable());
        assert!(!OperandCapability::Absent(AbsenceKind::AbsentEverywhere).is_executable());
        assert!(!OperandCapability::PresentUnsupported {
            format: RegionFormat::Unknown(900),
            packing: Packing::RowMajor,
        }
        .is_executable());
    }

    #[test]
    fn an_unsupported_codec_is_present_not_absent() {
        // The distinction §6.5 preserves at parse time has to survive here:
        // bytes this binary cannot read are not missing bytes.
        let c = OperandCapability::PresentUnsupported {
            format: RegionFormat::Nvfp4,
            packing: Packing::BlocksValues,
        };
        assert!(c.bytes_present());
        assert!(!c.is_executable());
    }

    #[test]
    fn an_absent_operand_has_no_bytes() {
        assert!(!OperandCapability::Absent(AbsenceKind::AbsentEverywhere).bytes_present());
    }

    #[test]
    fn a_real_absence_is_an_index_defect() {
        assert!(OperandCapability::Absent(AbsenceKind::AbsentEverywhere).indicates_index_defect());
    }

    #[test]
    fn a_deliberate_omission_is_not_an_index_defect() {
        // A browse slice lacking `down` is working as designed.
        assert!(
            !OperandCapability::Absent(AbsenceKind::OmittedBySelection).indicates_index_defect()
        );
    }

    #[test]
    fn an_unsupported_codec_is_not_an_index_defect() {
        // The artifact may be intact and simply newer than this reader.
        assert!(!OperandCapability::PresentUnsupported {
            format: RegionFormat::Unknown(901),
            packing: Packing::RowMajor,
        }
        .indicates_index_defect());
    }

    #[test]
    fn reference_execution_reports_no_kernel_maturity() {
        // "Executable but unoptimised" must not read as a kernel binding.
        assert_eq!(OperandCapability::ReferenceExecutable.maturity(), None);
    }

    #[test]
    fn kernel_execution_reports_its_maturity() {
        assert_eq!(
            kernel(KernelMaturity::Grouped).maturity(),
            Some(KernelMaturity::Grouped)
        );
    }

    #[test]
    fn maturity_orders_from_grouped_to_production() {
        assert!(KernelMaturity::Grouped < KernelMaturity::Dispatched);
        assert!(KernelMaturity::Dispatched < KernelMaturity::Production);
    }

    #[test]
    fn absence_and_no_kernel_are_never_confused_in_diagnostics() {
        // The §11 separation, as text: one says absent, the other says
        // executable. An operator reading these must not have to guess.
        let absent = OperandCapability::Absent(AbsenceKind::AbsentEverywhere).describe();
        let reference = OperandCapability::ReferenceExecutable.describe();
        assert!(absent.starts_with("absent"), "{absent}");
        assert!(reference.contains("executable"), "{reference}");
        assert!(!reference.contains("absent"), "{reference}");
    }

    #[test]
    fn an_unsupported_codec_names_both_format_and_packing() {
        let s = OperandCapability::PresentUnsupported {
            format: RegionFormat::Mxfp8,
            packing: Packing::BlocksScales,
        }
        .describe();
        assert!(s.contains("mxfp8"), "{s}");
        assert!(s.contains("blocks_scales"), "{s}");
    }

    #[test]
    fn a_kernel_binding_names_the_kernel_and_its_rung() {
        let s = kernel(KernelMaturity::Dispatched).describe();
        assert!(s.contains("grouped_experts_q6k"), "{s}");
        assert!(s.contains("dispatched"), "{s}");
    }

    #[test]
    fn partial_coverage_describes_which_segments_are_missing() {
        let s = OperandCapability::Absent(AbsenceKind::PartialSegmentCoverage {
            present: vec![0],
            missing: vec![1],
        })
        .describe();
        assert!(s.contains("missing from 1"), "{s}");
    }
}
