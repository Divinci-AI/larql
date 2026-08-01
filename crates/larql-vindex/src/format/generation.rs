//! Container generation detection and dispatch (format spec §12.1).
//!
//! One larql binary supports both vindex generations, indefinitely, for reading
//! and serving. `index.json`'s `version` field is the **sole** discriminator —
//! no filename sniffing, no directory-shape heuristics:
//!
//! | `index.json.version` | generation | layer format |
//! | -------------------- | ---------- | ------------ |
//! | 1                    | VINDEX2    | LYRW `format_version` 1 |
//! | 2                    | VINDEX2    | LYRW `format_version` 1 |
//! | 3                    | VINDEX3    | LYRW `format_version` 2 |
//!
//! The generation is named for its *current* `index.json.version`. An earlier
//! draft called the shipped generation "VINDEX1" while its version was already
//! 2, which put a permanent off-by-one between the name and the discriminator
//! — the single most likely way to mis-detect a directory. Both were renamed
//! so the two agree.
//!
//! **The mapping is not a bijection below 2.** `index.json.version` 1 is a
//! *legacy schema of the same shipped generation*, not a pre-generation
//! artifact: such indexes exist in the wild and the loader reads them by
//! filling absent fields with defaults. Refusing them would break VINDEX2
//! compatibility, which is the one thing dual-generation support exists to
//! protect. This was caught by the E0 preservation matrix, which is why that
//! matrix runs on every commit.
//!
//! The layer format keeps its own sequence and is deliberately *not* aligned to
//! either: LYRW is a different artifact with a different lifetime, and its own
//! numbering was already correct. A VINDEX3 container holds LYRW v2 files.
//!
//! Detection fails closed. An unknown version names the version found and the
//! versions this binary supports, before any weight byte is read — a loader
//! that guesses produces a served model with wrong weights, not an error.

use std::path::Path;

use crate::format::filenames::INDEX_JSON;
use crate::VindexError;

/// Current `index.json.version` for a VINDEX2 container — what a fresh
/// extraction of the shipped generation writes.
pub const V2_INDEX_VERSION: u32 = 2;

/// Oldest `index.json.version` still recognised as the shipped generation.
///
/// Version 1 predates several `index.json` fields and loads with defaults.
/// It is the same container generation, not an older one.
pub const V2_LEGACY_INDEX_VERSION: u32 = 1;
/// `index.json.version` for a VINDEX3 container.
pub const V3_INDEX_VERSION: u32 = 3;

/// Which container generation a directory holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContainerGeneration {
    /// The shipped generation: `index.json.version` 2, LYRW `format_version` 1.
    V2,
    /// The successor: `index.json.version` 3, LYRW `format_version` 2.
    V3,
}

impl ContainerGeneration {
    /// The `index.json.version` this generation writes. Equal to the
    /// generation number — that equality is the point of the naming.
    pub const fn index_version(self) -> u32 {
        match self {
            Self::V2 => V2_INDEX_VERSION,
            Self::V3 => V3_INDEX_VERSION,
        }
    }

    /// The LYRW `format_version` this generation's layer files carry.
    ///
    /// One behind the container generation, permanently: LYRW started at 1
    /// while `index.json` was already at 2.
    pub const fn lyrw_format_version(self) -> u32 {
        self.index_version() - 1
    }

    /// Human-readable name used in diagnostics — "VINDEX2" / "VINDEX3".
    pub const fn name(self) -> &'static str {
        match self {
            Self::V2 => "VINDEX2",
            Self::V3 => "VINDEX3",
        }
    }

    /// Map an `index.json.version` to a generation, or refuse by naming both
    /// the version found and what this binary understands.
    pub fn from_index_version(found: u32) -> Result<Self, VindexError> {
        match found {
            V2_LEGACY_INDEX_VERSION | V2_INDEX_VERSION => Ok(Self::V2),
            V3_INDEX_VERSION => Ok(Self::V3),
            other => Err(VindexError::UnknownContainerGeneration {
                found: other,
                supported: format!(
                    "{V2_LEGACY_INDEX_VERSION}-{V2_INDEX_VERSION} (VINDEX2), \
                     {V3_INDEX_VERSION} (VINDEX3)"
                ),
            }),
        }
    }

    /// Map a LYRW `format_version` to the container generation that writes it.
    ///
    /// Used by the layer parsers so a wrong-generation layer file names the
    /// loader the caller actually needs, rather than echoing a LYRW version
    /// number that no CLI flag or directory is labelled with.
    pub fn from_lyrw_format_version(found: u32) -> Result<Self, VindexError> {
        Self::from_index_version(found.saturating_add(1))
    }

    /// Refuse if this is not the generation the caller's path handles.
    ///
    /// Used by each loader at its own entry point, so "the VINDEX2 loader never
    /// opens a VINDEX3 directory" is enforced rather than merely documented.
    pub fn require(self, required: Self) -> Result<(), VindexError> {
        if self == required {
            return Ok(());
        }
        Err(VindexError::WrongContainerGeneration {
            found: self.name(),
            required: required.name(),
        })
    }
}

/// Read `index.json` and report which generation the directory holds.
///
/// Deliberately parses only the `version` field: a VINDEX3 `index.json` carries
/// keys the VINDEX2 config struct does not model, and full deserialisation
/// would fail on shape before it could report the far more useful "wrong
/// generation".
pub fn detect_generation(dir: &Path) -> Result<ContainerGeneration, VindexError> {
    let path = dir.join(INDEX_JSON);
    let text = std::fs::read_to_string(&path)?;
    let probe: VersionProbe =
        serde_json::from_str(&text).map_err(|e| VindexError::Parse(e.to_string()))?;
    match probe.version {
        Some(v) => ContainerGeneration::from_index_version(v),
        None => Err(VindexError::UnknownContainerGeneration {
            found: 0,
            supported: format!(
                "{V2_INDEX_VERSION} (VINDEX2), {V3_INDEX_VERSION} (VINDEX3); \
                 index.json declared no version field"
            ),
        }),
    }
}

/// Minimal view over `index.json` — the version field and nothing else.
#[derive(serde::Deserialize)]
struct VersionProbe {
    version: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_index(dir: &Path, body: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(INDEX_JSON), body).unwrap();
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join("vindex-generation-tests")
            .join(name);
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn the_generation_number_is_the_index_version() {
        // The whole reason both generations were renamed.
        for gen in [ContainerGeneration::V2, ContainerGeneration::V3] {
            let digit: u32 = gen.name().trim_start_matches("VINDEX").parse().unwrap();
            assert_eq!(digit, gen.index_version(), "{}", gen.name());
        }
    }

    #[test]
    fn lyrw_version_trails_the_generation_by_one() {
        assert_eq!(ContainerGeneration::V2.lyrw_format_version(), 1);
        assert_eq!(ContainerGeneration::V3.lyrw_format_version(), 2);
    }

    #[test]
    fn versions_round_trip_to_generations() {
        for gen in [ContainerGeneration::V2, ContainerGeneration::V3] {
            assert_eq!(
                ContainerGeneration::from_index_version(gen.index_version()).unwrap(),
                gen
            );
        }
    }

    #[test]
    fn lyrw_versions_round_trip_to_generations() {
        for gen in [ContainerGeneration::V2, ContainerGeneration::V3] {
            assert_eq!(
                ContainerGeneration::from_lyrw_format_version(gen.lyrw_format_version()).unwrap(),
                gen
            );
        }
    }

    #[test]
    fn names_are_the_ones_diagnostics_promise() {
        assert_eq!(ContainerGeneration::V2.name(), "VINDEX2");
        assert_eq!(ContainerGeneration::V3.name(), "VINDEX3");
    }

    #[test]
    fn an_unknown_version_names_both_sides() {
        let err = ContainerGeneration::from_index_version(99).unwrap_err();
        let s = err.to_string();
        assert!(s.contains("99"), "{s}");
        assert!(s.contains("VINDEX2"), "{s}");
        assert!(s.contains("VINDEX3"), "{s}");
    }

    #[test]
    fn the_legacy_index_schema_is_still_the_shipped_generation() {
        // Caught by E0: version 1 indexes exist in the wild and load with
        // defaults. Treating the version as a generation *identifier* rather
        // than a generation *floor* refused them, breaking the compatibility
        // dual-generation support exists to protect.
        assert_eq!(
            ContainerGeneration::from_index_version(V2_LEGACY_INDEX_VERSION).unwrap(),
            ContainerGeneration::V2
        );
    }

    #[test]
    fn version_zero_is_not_a_container_generation() {
        assert!(ContainerGeneration::from_index_version(0).is_err());
    }

    #[test]
    fn an_unknown_lyrw_version_is_refused() {
        // LYRW format_version 9 would imply a VINDEX10 container.
        assert!(ContainerGeneration::from_lyrw_format_version(9).is_err());
    }

    #[test]
    fn require_accepts_a_matching_generation() {
        assert!(ContainerGeneration::V2
            .require(ContainerGeneration::V2)
            .is_ok());
    }

    #[test]
    fn require_refuses_a_v3_directory_on_the_v2_path() {
        let err = ContainerGeneration::V3
            .require(ContainerGeneration::V2)
            .unwrap_err();
        let s = err.to_string();
        assert!(s.contains("VINDEX3"), "{s}");
        assert!(s.contains("VINDEX2"), "{s}");
    }

    #[test]
    fn require_refuses_a_v2_directory_on_the_v3_path() {
        let err = ContainerGeneration::V2
            .require(ContainerGeneration::V3)
            .unwrap_err();
        assert!(err.to_string().contains("VINDEX3"), "{err}");
    }

    #[test]
    fn detect_reads_a_shipped_generation_directory() {
        let dir = temp_dir("v2");
        write_index(&dir, r#"{"version": 2, "model_name": "x"}"#);
        assert_eq!(detect_generation(&dir).unwrap(), ContainerGeneration::V2);
    }

    #[test]
    fn detect_reads_a_successor_directory() {
        let dir = temp_dir("v3");
        write_index(
            &dir,
            r#"{"version": 3, "moe_manifest": "moe_manifest.json"}"#,
        );
        assert_eq!(detect_generation(&dir).unwrap(), ContainerGeneration::V3);
    }

    #[test]
    fn detect_ignores_keys_the_shipped_config_does_not_model() {
        // A VINDEX3 index.json carries fields VindexConfig has never seen.
        // Detection must still report the generation rather than fail on shape.
        let dir = temp_dir("v3-rich");
        write_index(
            &dir,
            r#"{"version": 3, "profiles": ["exact"], "segments": {"routed/layer_0": 2}}"#,
        );
        assert_eq!(detect_generation(&dir).unwrap(), ContainerGeneration::V3);
    }

    #[test]
    fn detect_refuses_an_unknown_version() {
        let dir = temp_dir("v9");
        write_index(&dir, r#"{"version": 9}"#);
        assert!(matches!(
            detect_generation(&dir),
            Err(VindexError::UnknownContainerGeneration { found: 9, .. })
        ));
    }

    #[test]
    fn detect_refuses_a_missing_version_field() {
        let dir = temp_dir("no-version");
        write_index(&dir, r#"{"model_name": "x"}"#);
        let err = detect_generation(&dir).unwrap_err();
        assert!(err.to_string().contains("no version field"), "{err}");
    }

    #[test]
    fn detect_reports_a_missing_index_json_as_io() {
        let dir = temp_dir("absent");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(matches!(detect_generation(&dir), Err(VindexError::Io(_))));
    }

    #[test]
    fn detect_reports_malformed_json_as_parse() {
        let dir = temp_dir("malformed");
        write_index(&dir, "{not json");
        assert!(matches!(
            detect_generation(&dir),
            Err(VindexError::Parse(_))
        ));
    }
}
