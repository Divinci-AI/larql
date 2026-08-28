//! Mechanistic fact-editing primitives.
//!
//! Implements the rank-1 ROME update and the multi-fact MEMIT patch format,
//! wrapping the algorithms validated in Python in Divinci-AI/server
//! notebooks/CHAPTER_20_HONEY.md (Phase 140) and CHAPTER_22_DISTRIBUTED_STACK.md
//! (Phase 142).
//!
//! Two patch kinds share the same on-disk envelope:
//!
//!   `RankOne` — ΔW = d ⊗ k_norm (stored as two f32 vectors). ~55 KB for
//!   Gemma 4 4B. Emitted by `larql edit`.
//!
//!   `Dense`   — ΔW stored flat row-major (hidden × intermediate). Larger
//!   (~72 MB for Gemma 4 4B) but exact. Emitted by `larql memit` when the
//!   covariance-based MEMIT solver produces a delta that isn't natively
//!   a rank-1 outer product.
//!
//! `apply_patch` dispatches on the kind and adds the resulting ΔW into
//! `down_proj.weight` in place.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use larql_models::{ModelWeights, WeightArray};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Envelope metadata written into every `.lqpatch` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditPatch {
    pub version: u32,
    pub layer: usize,
    pub module: String,
    /// Hidden size of the target model.
    pub hidden_size: usize,
    /// Intermediate size of the target model.
    pub intermediate_size: usize,
    /// Kind tag (also implicit in the binary body layout). Default
    /// "rank_one" for older (version=1) files.
    #[serde(default = "default_kind")]
    pub kind: String,
    /// Scale factor used during creation (informational).
    #[serde(default)]
    pub scale: f32,
    /// Provenance.
    #[serde(default)]
    pub provenance: PatchProvenance,

    // ── Binary body (not serialised to JSON; written separately) ──
    #[serde(skip)]
    pub d: Vec<f32>, // hidden_size — populated only for kind="rank_one"
    #[serde(skip)]
    pub k_norm: Vec<f32>, // intermediate_size — populated only for kind="rank_one"
    #[serde(skip)]
    pub delta_w: Vec<f32>, // hidden_size * intermediate_size (row-major) — populated only for kind="dense"
}

fn default_kind() -> String {
    "rank_one".to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatchProvenance {
    pub src_prompt: String,
    pub tgt_prompt: String,
    pub old_token: String,
    pub new_token: String,
    pub crown_delta: f64,
    pub created_at: String,
}

/// File magic for all .lqpatch files.
const PATCH_MAGIC: &[u8; 8] = b"LQPATCH\0";

// ── Writers ─────────────────────────────────────────────────────────

/// Write an `EditPatch` to disk. Dispatches to rank-one or dense layout
/// based on `patch.kind`.
pub fn write_patch(path: impl AsRef<Path>, patch: &EditPatch) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(PATCH_MAGIC)?;

    // Serialise metadata with the body fields empty.
    let meta = EditPatch {
        d: Vec::new(),
        k_norm: Vec::new(),
        delta_w: Vec::new(),
        ..patch.clone()
    };
    let meta_json = serde_json::to_vec(&meta)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    w.write_all(&(meta_json.len() as u32).to_le_bytes())?;
    w.write_all(&meta_json)?;

    match patch.kind.as_str() {
        "rank_one" => {
            w.write_all(&(patch.d.len() as u32).to_le_bytes())?;
            write_f32s(&mut w, &patch.d)?;
            w.write_all(&(patch.k_norm.len() as u32).to_le_bytes())?;
            write_f32s(&mut w, &patch.k_norm)?;
        }
        "dense" => {
            let expected = patch.hidden_size * patch.intermediate_size;
            if patch.delta_w.len() != expected {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "dense delta_w length {} != hidden*intermediate {}",
                        patch.delta_w.len(),
                        expected
                    ),
                ));
            }
            w.write_all(&(patch.delta_w.len() as u32).to_le_bytes())?;
            write_f32s(&mut w, &patch.delta_w)?;
        }
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown patch kind: {other}"),
            ));
        }
    }

    w.flush()?;
    Ok(())
}

// ── Readers ─────────────────────────────────────────────────────────

/// Read an `EditPatch` from disk. Dispatches on the stored kind.
pub fn read_patch(path: impl AsRef<Path>) -> std::io::Result<EditPatch> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != PATCH_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a LarQL patch file (bad magic)",
        ));
    }
    let meta_len = read_u32(&mut r)? as usize;
    let mut meta_buf = vec![0u8; meta_len];
    r.read_exact(&mut meta_buf)?;
    let mut patch: EditPatch = serde_json::from_slice(&meta_buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    match patch.kind.as_str() {
        "rank_one" => {
            let d_len = read_u32(&mut r)? as usize;
            patch.d = read_f32s(&mut r, d_len)?;
            let k_len = read_u32(&mut r)? as usize;
            patch.k_norm = read_f32s(&mut r, k_len)?;
            if patch.d.len() != patch.hidden_size
                || patch.k_norm.len() != patch.intermediate_size
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "rank_one shape mismatch: d={} (hidden={}), k={} (intermediate={})",
                        patch.d.len(), patch.hidden_size,
                        patch.k_norm.len(), patch.intermediate_size
                    ),
                ));
            }
        }
        "dense" => {
            let dw_len = read_u32(&mut r)? as usize;
            patch.delta_w = read_f32s(&mut r, dw_len)?;
            let expected = patch.hidden_size * patch.intermediate_size;
            if patch.delta_w.len() != expected {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("dense len {} != hidden*intermediate {}", patch.delta_w.len(), expected),
                ));
            }
        }
        other => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unknown patch kind: {other}"),
            ));
        }
    }
    Ok(patch)
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32s<R: Read>(r: &mut R, n: usize) -> std::io::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n);
    let mut buf = [0u8; 4];
    for _ in 0..n {
        r.read_exact(&mut buf)?;
        out.push(f32::from_le_bytes(buf));
    }
    Ok(out)
}

fn write_f32s<W: Write>(w: &mut W, xs: &[f32]) -> std::io::Result<()> {
    for &v in xs {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

// ── Construction helpers ────────────────────────────────────────────

/// Build a rank-1 patch from captured key and desired output delta.
/// (Phase B path — single-fact edit.)
pub fn compute_rank1(
    k: &[f32],
    d: &[f32],
    scale: f32,
    layer: usize,
    provenance: PatchProvenance,
) -> EditPatch {
    let kk = k.iter().map(|&v| v * v).sum::<f32>().max(1e-12);
    let k_norm: Vec<f32> = k.iter().map(|&v| v / kk).collect();
    let d_scaled: Vec<f32> = d.iter().map(|&v| v * scale).collect();
    EditPatch {
        version: 2,
        layer,
        module: "down_proj".to_string(),
        hidden_size: d.len(),
        intermediate_size: k.len(),
        kind: "rank_one".to_string(),
        scale,
        provenance,
        d: d_scaled,
        k_norm,
        delta_w: Vec::new(),
    }
}

/// Build a dense patch from a full ΔW matrix (hidden × intermediate, row-major).
/// (Phase C path — MEMIT output.)
pub fn compute_dense(
    delta_w: &Array2<f32>,
    layer: usize,
    provenance: PatchProvenance,
) -> EditPatch {
    let (hidden, intermediate) = (delta_w.shape()[0], delta_w.shape()[1]);
    // Row-major flatten.
    let mut flat = Vec::with_capacity(hidden * intermediate);
    for row in delta_w.rows() {
        for &v in row {
            flat.push(v);
        }
    }
    EditPatch {
        version: 2,
        layer,
        module: "down_proj".to_string(),
        hidden_size: hidden,
        intermediate_size: intermediate,
        kind: "dense".to_string(),
        scale: 1.0,
        provenance,
        d: Vec::new(),
        k_norm: Vec::new(),
        delta_w: flat,
    }
}

// ── Apply ───────────────────────────────────────────────────────────

/// Apply a patch to a model's `down_proj` weight at the target layer,
/// in-place. Handles both rank-1 and dense variants.
pub fn apply_patch(weights: &mut ModelWeights, patch: &EditPatch) -> Result<(), String> {
    let w_down_key = weights.arch.ffn_down_key(patch.layer);
    let existing = weights
        .tensors
        .get(&w_down_key)
        .ok_or_else(|| format!("apply_patch: W_down not found at {w_down_key}"))?;
    let (rows, cols) = (existing.shape()[0], existing.shape()[1]);
    let hidden = patch.hidden_size;
    let intermediate = patch.intermediate_size;

    // Detect storage layout.
    let transposed = if rows == hidden && cols == intermediate {
        false
    } else if rows == intermediate && cols == hidden {
        true
    } else {
        return Err(format!(
            "apply_patch: W_down shape {rows}x{cols} doesn't match patch ({hidden}x{intermediate})"
        ));
    };

    let mut updated = existing.as_standard_layout().to_owned();

    match patch.kind.as_str() {
        "rank_one" => {
            let d = Array1::from(patch.d.clone());
            let k = Array1::from(patch.k_norm.clone());
            let delta: Array2<f32> = if !transposed {
                outer(&d, &k) // (hidden, intermediate)
            } else {
                outer(&k, &d) // (intermediate, hidden)
            };
            updated = &updated + &delta;
        }
        "dense" => {
            // Reshape the flat row-major vector back into [hidden, intermediate].
            let delta = Array2::from_shape_vec((hidden, intermediate), patch.delta_w.clone())
                .map_err(|e| format!("dense reshape failed: {e}"))?;
            if !transposed {
                updated = &updated + &delta;
            } else {
                // Target storage is (intermediate, hidden); add the transpose.
                updated = &updated + &delta.t();
            }
        }
        other => return Err(format!("apply_patch: unknown kind {other}")),
    }

    let updated_weight: WeightArray = updated.into_shared();
    weights.tensors.insert(w_down_key, updated_weight);
    Ok(())
}

fn outer(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    let a_col = a.view().insert_axis(ndarray::Axis(1));
    let b_row = b.view().insert_axis(ndarray::Axis(0));
    a_col.dot(&b_row)
}
