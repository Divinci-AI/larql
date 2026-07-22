//! On-disk residual capture pool for the DEC replay loadgen.
//!
//! A capture directory holds two files:
//!   * `manifest.json` — [`CaptureManifest`]: model identity, shape, and
//!     per-prompt step counts.
//!   * `residuals.bin` — f32 LE, layout `[prompt][step][layer][hidden]`,
//!     `steps` = the minimum step count across prompts (ragged tails are
//!     truncated at write time so every `(step, layer)` cell has a row from
//!     every prompt).
//!
//! Rows are captured **pre-normed** (see `ResidualCaptureSink`), i.e. exactly
//! the bytes the f32 wire carries and exactly the input Q8K quantisation
//! consumes — so replay needs no model weights and the pool is portable
//! across hosts (Mac capture → x86 replay).

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

/// Bump when the binary layout changes. `open` rejects other versions.
pub const CAPTURE_VERSION: u32 = 1;

const MANIFEST_FILE: &str = "manifest.json";
const RESIDUALS_FILE: &str = "residuals.bin";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMeta {
    pub id: usize,
    pub text: String,
    /// Steps this prompt actually produced before truncation to `steps`.
    pub steps_captured: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureManifest {
    pub version: u32,
    /// Model path / id the pool was captured from.
    pub model: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    /// Replayable steps: min over prompts of steps captured.
    pub steps: usize,
    /// Always `"f32-le"` at version 1.
    pub dtype: String,
    pub prompts: Vec<PromptMeta>,
    pub created_unix: u64,
}

impl CaptureManifest {
    /// Expected `residuals.bin` byte length for this manifest.
    pub fn expected_bytes(&self) -> u64 {
        self.prompts.len() as u64
            * self.steps as u64
            * self.num_layers as u64
            * self.hidden_size as u64
            * 4
    }
}

/// An open, validated capture pool.
pub struct CapturePool {
    pub manifest: CaptureManifest,
    /// Raw `residuals.bin` contents; decoded to f32 on demand in [`Self::rows`].
    data: Vec<u8>,
}

impl CapturePool {
    /// Write a pool from per-prompt captured steps.
    ///
    /// `per_prompt[p][step][layer]` is the pre-normed residual for prompt `p`.
    /// Prompts with more steps than the shortest are truncated so the binary
    /// layout is rectangular; the pre-truncation count is kept in
    /// [`PromptMeta::steps_captured`].
    pub fn write(
        dir: &Path,
        model: &str,
        hidden_size: usize,
        num_layers: usize,
        prompt_texts: &[String],
        per_prompt: &[Vec<Vec<Vec<f32>>>],
        created_unix: u64,
    ) -> Result<CaptureManifest, String> {
        if per_prompt.is_empty() {
            return Err("capture pool: no prompts captured".into());
        }
        if per_prompt.len() != prompt_texts.len() {
            return Err(format!(
                "capture pool: {} prompt texts but {} capture sinks",
                prompt_texts.len(),
                per_prompt.len()
            ));
        }
        let steps = per_prompt.iter().map(|s| s.len()).min().unwrap_or(0);
        if steps == 0 {
            return Err("capture pool: a prompt produced zero decode steps".into());
        }
        for (p, prompt_steps) in per_prompt.iter().enumerate() {
            for (s, layers) in prompt_steps.iter().take(steps).enumerate() {
                if layers.len() != num_layers {
                    return Err(format!(
                        "capture pool: prompt {p} step {s} has {} layers, expected {num_layers}",
                        layers.len()
                    ));
                }
                for (l, row) in layers.iter().enumerate() {
                    if row.len() != hidden_size {
                        return Err(format!(
                            "capture pool: prompt {p} step {s} layer {l} has {} floats, \
                             expected hidden {hidden_size}",
                            row.len()
                        ));
                    }
                }
            }
        }

        let manifest = CaptureManifest {
            version: CAPTURE_VERSION,
            model: model.to_string(),
            hidden_size,
            num_layers,
            steps,
            dtype: "f32-le".into(),
            prompts: prompt_texts
                .iter()
                .enumerate()
                .map(|(id, text)| PromptMeta {
                    id,
                    text: text.clone(),
                    steps_captured: per_prompt[id].len(),
                })
                .collect(),
            created_unix,
        };

        std::fs::create_dir_all(dir).map_err(|e| format!("capture pool: mkdir: {e}"))?;
        let bin_path = dir.join(RESIDUALS_FILE);
        let f = std::fs::File::create(&bin_path)
            .map_err(|e| format!("capture pool: create {}: {e}", bin_path.display()))?;
        let mut w = std::io::BufWriter::new(f);
        for prompt_steps in per_prompt {
            for layers in prompt_steps.iter().take(steps) {
                for row in layers {
                    for &v in row {
                        w.write_all(&v.to_le_bytes())
                            .map_err(|e| format!("capture pool: write: {e}"))?;
                    }
                }
            }
        }
        w.flush().map_err(|e| format!("capture pool: flush: {e}"))?;

        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| format!("capture pool: manifest serialize: {e}"))?;
        std::fs::write(dir.join(MANIFEST_FILE), manifest_json)
            .map_err(|e| format!("capture pool: write manifest: {e}"))?;
        Ok(manifest)
    }

    /// Open and validate a pool directory.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let manifest_path = dir.join(MANIFEST_FILE);
        let manifest_json = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("capture pool: read {}: {e}", manifest_path.display()))?;
        let manifest: CaptureManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| format!("capture pool: parse manifest: {e}"))?;
        if manifest.version != CAPTURE_VERSION {
            return Err(format!(
                "capture pool: version {} unsupported (expected {CAPTURE_VERSION})",
                manifest.version
            ));
        }
        let bin_path = dir.join(RESIDUALS_FILE);
        let data = std::fs::read(&bin_path)
            .map_err(|e| format!("capture pool: read {}: {e}", bin_path.display()))?;
        if data.len() as u64 != manifest.expected_bytes() {
            return Err(format!(
                "capture pool: residuals.bin is {} bytes, manifest expects {}",
                data.len(),
                manifest.expected_bytes()
            ));
        }
        Ok(Self { manifest, data })
    }

    /// Number of prompts in the pool — the maximum replay batch size.
    pub fn num_prompts(&self) -> usize {
        self.manifest.prompts.len()
    }

    /// Build a `batch × hidden` contiguous row block for `(step, layer)`:
    /// row `i` is prompt `i`'s pre-normed residual at that step and layer.
    /// Distinct prompts per row keep MoE routing union realistic.
    pub fn rows(&self, batch: usize, step: usize, layer: usize) -> Result<Vec<f32>, String> {
        let m = &self.manifest;
        if batch == 0 || batch > m.prompts.len() {
            return Err(format!(
                "capture pool: batch {batch} out of range (pool has {} prompts)",
                m.prompts.len()
            ));
        }
        if step >= m.steps {
            return Err(format!(
                "capture pool: step {step} out of range (pool has {} steps)",
                m.steps
            ));
        }
        if layer >= m.num_layers {
            return Err(format!(
                "capture pool: layer {layer} out of range (pool has {} layers)",
                m.num_layers
            ));
        }
        let row_bytes = m.hidden_size * 4;
        let mut out = Vec::with_capacity(batch * m.hidden_size);
        for prompt in 0..batch {
            let row_idx = (prompt * m.steps + step) * m.num_layers + layer;
            let start = row_idx * row_bytes;
            out.extend(
                self.data[start..start + row_bytes]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_capture(
        prompts: usize,
        steps: usize,
        layers: usize,
        hidden: usize,
    ) -> Vec<Vec<Vec<Vec<f32>>>> {
        (0..prompts)
            .map(|p| {
                (0..steps)
                    .map(|s| {
                        (0..layers)
                            .map(|l| {
                                (0..hidden)
                                    .map(|h| (p * 1000 + s * 100 + l * 10 + h) as f32)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn write_synthetic(dir: &Path, prompts: usize, steps: usize, layers: usize, hidden: usize) {
        let cap = synthetic_capture(prompts, steps, layers, hidden);
        let texts: Vec<String> = (0..prompts).map(|i| format!("prompt {i}")).collect();
        CapturePool::write(dir, "test-model", hidden, layers, &texts, &cap, 12345).unwrap();
    }

    #[test]
    fn manifest_round_trips_through_serde() {
        let m = CaptureManifest {
            version: CAPTURE_VERSION,
            model: "m".into(),
            hidden_size: 8,
            num_layers: 2,
            steps: 3,
            dtype: "f32-le".into(),
            prompts: vec![PromptMeta {
                id: 0,
                text: "p".into(),
                steps_captured: 3,
            }],
            created_unix: 99,
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: CaptureManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.version, m.version);
        assert_eq!(back.hidden_size, 8);
        assert_eq!(back.expected_bytes(), 3 * 2 * 8 * 4);
    }

    #[test]
    fn write_open_rows_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic(dir.path(), 2, 2, 2, 4);
        let pool = CapturePool::open(dir.path()).unwrap();
        assert_eq!(pool.num_prompts(), 2);

        // Row i of a batch must be prompt i at the same (step, layer).
        let rows = pool.rows(2, 1, 1).unwrap();
        assert_eq!(rows.len(), 2 * 4);
        let expect_p0: Vec<f32> = (0..4).map(|h| (100 + 10 + h) as f32).collect();
        let expect_p1: Vec<f32> = (0..4).map(|h| (1000 + 100 + 10 + h) as f32).collect();
        assert_eq!(&rows[..4], &expect_p0[..]);
        assert_eq!(&rows[4..], &expect_p1[..]);
    }

    #[test]
    fn write_truncates_ragged_prompts_to_min_steps() {
        let dir = tempfile::tempdir().unwrap();
        let mut cap = synthetic_capture(2, 3, 1, 2);
        cap[1].truncate(1); // prompt 1 stopped early (EOS)
        let texts = vec!["a".into(), "b".into()];
        let m = CapturePool::write(dir.path(), "m", 2, 1, &texts, &cap, 0).unwrap();
        assert_eq!(m.steps, 1);
        assert_eq!(m.prompts[0].steps_captured, 3);
        assert_eq!(m.prompts[1].steps_captured, 1);
        let pool = CapturePool::open(dir.path()).unwrap();
        assert!(pool.rows(2, 0, 0).is_ok());
        assert!(pool.rows(2, 1, 0).is_err(), "step beyond min must fail");
    }

    #[test]
    fn write_rejects_empty_and_mismatched_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let texts = vec!["a".into()];
        assert!(CapturePool::write(dir.path(), "m", 2, 1, &texts, &[], 0).is_err());

        // Wrong layer count.
        let cap = synthetic_capture(1, 1, 2, 2);
        assert!(CapturePool::write(dir.path(), "m", 2, 1, &texts, &cap, 0).is_err());

        // Wrong hidden size.
        let cap = synthetic_capture(1, 1, 1, 3);
        assert!(CapturePool::write(dir.path(), "m", 2, 1, &texts, &cap, 0).is_err());

        // Zero steps.
        let cap: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![]];
        assert!(CapturePool::write(dir.path(), "m", 2, 1, &texts, &cap, 0).is_err());
    }

    #[test]
    fn open_rejects_truncated_bin_and_bad_version() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic(dir.path(), 1, 1, 1, 4);

        // Truncate residuals.bin → open must fail on size mismatch.
        let bin = dir.path().join("residuals.bin");
        let data = std::fs::read(&bin).unwrap();
        std::fs::write(&bin, &data[..data.len() - 4]).unwrap();
        assert!(CapturePool::open(dir.path()).is_err());
        std::fs::write(&bin, &data).unwrap();
        assert!(CapturePool::open(dir.path()).is_ok());

        // Corrupt the version → open must fail.
        let mf = dir.path().join("manifest.json");
        let json = std::fs::read_to_string(&mf).unwrap();
        std::fs::write(&mf, json.replace("\"version\": 1", "\"version\": 999")).unwrap();
        assert!(CapturePool::open(dir.path()).is_err());
    }

    #[test]
    fn rows_bounds_checks() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic(dir.path(), 2, 2, 2, 2);
        let pool = CapturePool::open(dir.path()).unwrap();
        assert!(pool.rows(0, 0, 0).is_err());
        assert!(pool.rows(3, 0, 0).is_err(), "batch > prompts");
        assert!(pool.rows(1, 2, 0).is_err(), "step out of range");
        assert!(pool.rows(1, 0, 2).is_err(), "layer out of range");
    }
}
