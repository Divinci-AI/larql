//! Judge the file that will ship, not the model that was in memory.
//!
//! The collateral gate used to re-measure its controls on the IN-MEMORY weights after installing
//! the edge, and then write a checkpoint from those weights. On 2026-09-23 (the erasure
//! programme's §48) that measurement and the file disagreed: on gemma-4-E2B-it at --alpha 1.5 the
//! in-memory model answered the Japan control with a blank top-1 and the gate refused, while the
//! very bytes it would have written — reproduced exactly, sha 3873371745c1… — answered Tokyo at
//! p 0.9999 under larql's own `forward::predict` and under HF transformers in bf16 and f32.
//!
//! Whatever the mechanism (the in-memory edge is unrounded f32, the file stores bf16), a gate that
//! judges one object and writes another cannot make "refused" and "written" statements about the
//! same thing. So compile now:
//!
//!   1. writes the checkpoint into a STAGING directory beside the output,
//!   2. reloads it from disk with the same loader everything else uses,
//!   3. judges the controls (and the fluency bound) on that reload,
//!   4. PROMOTES staging to the output only on a pass, and discards it on a refusal,
//!
//! so a refusal still leaves no output directory behind, and a written output is exactly the file
//! that was judged.

use std::path::{Path, PathBuf};

/// A sibling of `output`, hidden, unique to this process. Beside the output rather than in a temp
/// dir so the final rename stays on one filesystem (a cross-device rename is a copy, and a copy
/// can die half way).
pub fn staging_dir(output: &Path) -> PathBuf {
    let name = output.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "out".into());
    let parent = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    parent.join(format!(".{name}.larql-staging-{}", std::process::id()))
}

/// Refuse up front if the output already holds anything: the gate must never overwrite a previous
/// result it did not judge. An existing EMPTY directory is allowed (and replaced on promote).
pub fn check_output_free(output: &Path) -> Result<(), String> {
    match std::fs::read_dir(output) {
        Ok(mut it) => {
            if it.next().is_some() {
                Err(format!("{} already exists and is not empty; refusing to overwrite it", output.display()))
            } else {
                Ok(())
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotADirectory => {
            Err(format!("{} exists and is not a directory", output.display()))
        }
        Err(e) => Err(format!("cannot inspect {}: {}", output.display(), e)),
    }
}

/// Create an empty staging directory. A leftover from a crashed run with the same pid is removed
/// first: it can only be ours.
pub fn create(staging: &Path) -> std::io::Result<()> {
    if staging.exists() {
        std::fs::remove_dir_all(staging)?;
    }
    std::fs::create_dir_all(staging)
}

/// Move the judged staging directory into place.
pub fn promote(staging: &Path, output: &Path) -> Result<(), String> {
    check_output_free(output)?;
    if output.exists() {
        std::fs::remove_dir(output).map_err(|e| format!("remove empty {}: {}", output.display(), e))?;
    }
    std::fs::rename(staging, output).map_err(|e| format!("promote {} -> {}: {}", staging.display(), output.display(), e))
}

/// Remove the staging directory after a refusal. Errors are reported, not swallowed: a refusal
/// that leaves a checkpoint lying around is the half-built-directory failure the gate exists to
/// prevent, so the caller should hear about it.
pub fn discard(staging: &Path) -> Result<(), String> {
    match std::fs::remove_dir_all(staging) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("could not remove staging {}: {}", staging.display(), e)),
    }
}

/// `values` as they will read back from a bf16 file. Diagnostic only (LARQL_COMPILE_DIAG): lets
/// one compile show whether the in-memory/file disagreement is the rounding of the edited slot.
pub fn round_through_bf16(values: &[f32]) -> Vec<f32> {
    larql_models::quant::half::decode_bf16(&larql_models::quant::half::encode_bf16(values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_is_a_hidden_sibling_unique_to_the_process() {
        let s = staging_dir(Path::new("/tmp/runs/out-a"));
        assert_eq!(s.parent(), Some(Path::new("/tmp/runs")));
        let n = s.file_name().unwrap().to_string_lossy().into_owned();
        assert!(n.starts_with(".out-a.larql-staging-"), "{n}");
        assert!(n.ends_with(&std::process::id().to_string()));
    }

    #[test]
    fn a_bare_relative_output_stages_in_the_current_directory() {
        assert_eq!(staging_dir(Path::new("out")).parent(), Some(Path::new(".")));
    }

    #[test]
    fn a_missing_or_empty_output_is_free_and_a_populated_one_is_not() {
        let d = tempfile::tempdir().unwrap();
        let out = d.path().join("out");
        assert!(check_output_free(&out).is_ok());
        std::fs::create_dir(&out).unwrap();
        assert!(check_output_free(&out).is_ok());
        std::fs::write(out.join("model.safetensors"), b"x").unwrap();
        assert!(check_output_free(&out).unwrap_err().contains("not empty"));
    }

    #[test]
    fn a_file_where_the_output_should_be_is_refused() {
        let d = tempfile::tempdir().unwrap();
        let out = d.path().join("out");
        std::fs::write(&out, b"x").unwrap();
        assert!(check_output_free(&out).is_err());
    }

    #[test]
    fn promote_moves_exactly_the_judged_files_into_place() {
        let d = tempfile::tempdir().unwrap();
        let out = d.path().join("out");
        let st = staging_dir(&out);
        create(&st).unwrap();
        std::fs::write(st.join("model.safetensors"), b"judged").unwrap();
        promote(&st, &out).unwrap();
        assert_eq!(std::fs::read(out.join("model.safetensors")).unwrap(), b"judged");
        assert!(!st.exists(), "staging is gone after promotion");
    }

    #[test]
    fn promote_replaces_an_empty_output_directory() {
        let d = tempfile::tempdir().unwrap();
        let out = d.path().join("out");
        std::fs::create_dir(&out).unwrap();
        let st = staging_dir(&out);
        create(&st).unwrap();
        std::fs::write(st.join("f"), b"1").unwrap();
        promote(&st, &out).unwrap();
        assert!(out.join("f").exists());
    }

    #[test]
    fn promote_never_overwrites_a_populated_output() {
        let d = tempfile::tempdir().unwrap();
        let out = d.path().join("out");
        std::fs::create_dir(&out).unwrap();
        std::fs::write(out.join("keep"), b"previous").unwrap();
        let st = staging_dir(&out);
        create(&st).unwrap();
        assert!(promote(&st, &out).is_err());
        assert_eq!(std::fs::read(out.join("keep")).unwrap(), b"previous");
    }

    #[test]
    fn discard_after_a_refusal_leaves_nothing_behind() {
        let d = tempfile::tempdir().unwrap();
        let out = d.path().join("out");
        let st = staging_dir(&out);
        create(&st).unwrap();
        std::fs::write(st.join("model.safetensors"), b"refused").unwrap();
        discard(&st).unwrap();
        assert!(!st.exists());
        assert!(!out.exists(), "a refusal creates no output directory");
        assert!(discard(&st).is_ok(), "discarding twice is fine");
    }

    #[test]
    fn create_clears_a_leftover_from_a_crashed_run() {
        let d = tempfile::tempdir().unwrap();
        let st = staging_dir(&d.path().join("out"));
        std::fs::create_dir_all(&st).unwrap();
        std::fs::write(st.join("stale"), b"x").unwrap();
        create(&st).unwrap();
        assert!(!st.join("stale").exists());
    }

    #[test]
    fn bf16_round_trip_is_what_a_bf16_file_reads_back() {
        let v = [1.0f32, 0.1, -3.14159, 1e-3];
        let r = round_through_bf16(&v);
        assert_eq!(r[0], 1.0);
        assert!(r[1] != 0.1 && (r[1] - 0.1).abs() < 1e-3, "0.1 is not representable in bf16");
        assert_eq!(round_through_bf16(&r), r, "rounding is idempotent");
    }
}
