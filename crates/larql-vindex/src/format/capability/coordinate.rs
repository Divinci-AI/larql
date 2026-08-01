//! Exact coordinates of a region within an index (spec §11).
//!
//! V2-0's acceptance contract requires missing operands to be diagnosed with
//! `{layer, bank, role, segment}` precision. "Some segment is missing `down`"
//! is not that: on a two-segment K3 routed layer it leaves the reader to
//! bisect 896 experts to find which half is broken.
//!
//! Segment identity is kept **individual** in the report even when several
//! adjacent segments fail. Presentation may compact a run into `segments 1–4`;
//! the report itself must not, because compaction is lossy and a consumer that
//! wants to re-fetch exactly the broken segments needs the list.

use crate::format::lyrw2::region_role::RegionRole;

/// Where a region lives, precisely enough to act on.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionCoordinate {
    pub layer: u32,
    pub bank_id: u16,
    /// Segment index within the bank. `None` for a bank the selection treats
    /// as unsegmented — distinct from segment 0, which is one segment of many.
    pub segment: Option<u16>,
    pub role: RegionRole,
}

impl RegionCoordinate {
    pub fn new(layer: u32, bank_id: u16, segment: Option<u16>, role: RegionRole) -> Self {
        Self {
            layer,
            bank_id,
            segment,
            role,
        }
    }

    /// Diagnostic form, in the order §11 names: layer, bank, role, segment.
    pub fn describe(&self) -> String {
        let seg = match self.segment {
            Some(s) => format!(" segment {s}"),
            None => String::new(),
        };
        format!(
            "layer {} bank {} role {}{}",
            self.layer,
            self.bank_id,
            self.role.name(),
            seg
        )
    }
}

/// Why a required region is not usable. Similar execution outcomes, different
/// fixes — which is the whole reason these are not collapsed into "absent".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbsenceKind {
    /// No segment carries this role. The variant was never extracted, or the
    /// slice dropped it. Fix: extract or fetch the variant.
    AbsentEverywhere,
    /// Some segments carry it, others do not. Almost always a partial fetch or
    /// an interrupted incremental pack. Fix: complete the transfer — the
    /// variant exists.
    PartialSegmentCoverage {
        present: Vec<u16>,
        missing: Vec<u16>,
    },
    /// Present across a segment set that does not match another selected
    /// role's. The layer cannot be executed even though nothing is strictly
    /// missing. Fix: reconcile the selection, not the storage.
    IncompatibleSegmentSet {
        role_segments: Vec<u16>,
        other_role: RegionRole,
        other_segments: Vec<u16>,
    },
    /// The active profile or slice deliberately omits it. Not a defect — an
    /// analysis-only browse slice has no `down` by design. Fix: none; use a
    /// profile that selects it.
    OmittedBySelection,
}

impl AbsenceKind {
    /// Whether this absence indicates something is wrong with the index, as
    /// opposed to a deliberate scoping decision.
    ///
    /// A browse slice missing `down` is working as designed; reporting it as
    /// corruption would teach operators to ignore the diagnostic.
    pub fn is_defect(&self) -> bool {
        !matches!(self, Self::OmittedBySelection)
    }

    pub fn describe(&self) -> String {
        match self {
            Self::AbsentEverywhere => "absent from every selected segment".into(),
            Self::PartialSegmentCoverage { present, missing } => format!(
                "present in segments {}, missing from {}",
                join(present),
                join(missing)
            ),
            Self::IncompatibleSegmentSet {
                role_segments,
                other_role,
                other_segments,
            } => format!(
                "covers segments {} but '{}' covers {} — no segment carries both",
                join(role_segments),
                other_role.name(),
                join(other_segments)
            ),
            Self::OmittedBySelection => "omitted by the active selection".into(),
        }
    }
}

fn join(segments: &[u16]) -> String {
    if segments.is_empty() {
        return "none".into();
    }
    segments
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_coordinate_describes_every_axis_section_eleven_names() {
        let c = RegionCoordinate::new(37, 0, Some(1), RegionRole::Down);
        let s = c.describe();
        assert!(s.contains("layer 37"), "{s}");
        assert!(s.contains("bank 0"), "{s}");
        assert!(s.contains("role down"), "{s}");
        assert!(s.contains("segment 1"), "{s}");
    }

    #[test]
    fn an_unsegmented_coordinate_omits_the_segment_axis() {
        let c = RegionCoordinate::new(3, 1, None, RegionRole::Gate);
        assert!(!c.describe().contains("segment"), "{}", c.describe());
    }

    #[test]
    fn segment_zero_is_distinct_from_unsegmented() {
        // Segment 0 is one segment of several; None means the selection treats
        // the bank as a whole. Conflating them loses which half is broken.
        let zero = RegionCoordinate::new(0, 0, Some(0), RegionRole::Down);
        let none = RegionCoordinate::new(0, 0, None, RegionRole::Down);
        assert_ne!(zero, none);
        assert!(zero.describe().contains("segment 0"));
    }

    #[test]
    fn coordinates_sort_by_layer_then_bank_then_segment() {
        let mut v = [
            RegionCoordinate::new(1, 0, Some(1), RegionRole::Down),
            RegionCoordinate::new(0, 0, Some(0), RegionRole::Down),
            RegionCoordinate::new(1, 0, Some(0), RegionRole::Down),
        ];
        v.sort();
        assert_eq!(v[0].layer, 0);
        assert_eq!(v[1].segment, Some(0));
        assert_eq!(v[2].segment, Some(1));
    }

    #[test]
    fn absent_everywhere_is_a_defect() {
        assert!(AbsenceKind::AbsentEverywhere.is_defect());
        assert!(AbsenceKind::AbsentEverywhere
            .describe()
            .contains("every selected segment"));
    }

    #[test]
    fn partial_coverage_names_both_sides() {
        let a = AbsenceKind::PartialSegmentCoverage {
            present: vec![0],
            missing: vec![1, 2],
        };
        assert!(a.is_defect());
        let s = a.describe();
        assert!(s.contains("segments 0"), "{s}");
        assert!(s.contains("1, 2"), "{s}");
    }

    #[test]
    fn an_incompatible_segment_set_names_the_conflicting_role() {
        let a = AbsenceKind::IncompatibleSegmentSet {
            role_segments: vec![0],
            other_role: RegionRole::Gate,
            other_segments: vec![1],
        };
        assert!(a.is_defect());
        let s = a.describe();
        assert!(s.contains("gate"), "{s}");
        assert!(s.contains("no segment carries both"), "{s}");
    }

    #[test]
    fn a_deliberate_omission_is_not_a_defect() {
        // A browse slice has no `down` by design. Reporting that as corruption
        // teaches operators to ignore the diagnostic.
        assert!(!AbsenceKind::OmittedBySelection.is_defect());
    }

    #[test]
    fn an_empty_segment_list_reads_as_none_not_as_blank() {
        let a = AbsenceKind::PartialSegmentCoverage {
            present: Vec::new(),
            missing: vec![0],
        };
        assert!(a.describe().contains("segments none"), "{}", a.describe());
    }
}
