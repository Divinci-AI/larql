//! Mechanistic fact-editing primitives.
//!
//! Implements the rank-1 ROME update on FFN `down_proj` and a tiny patch
//! file format, wrapping the algorithms validated in Python in
//! Divinci-AI/server notebooks/CHAPTER_20_HONEY.md (Phase 140) and
//! CHAPTER_23_PER_EDIT_CROWN.md (Phase 143).
//!
//! The rank-1 update:
//!
//!   ΔW = d ⊗ k_norm        where  k_norm = k / (k · k)
//!
//! satisfies exactly  `(W + ΔW) @ k = W @ k + d`  for any key `k`, and
//! for other keys `k'` the perturbation is `d * (k · k') / (k · k)` —
//! proportional to similarity with the edit's key, so orthogonal inputs
//! are unaffected.
//!
//! This module is designed to be thin — the hard numerical work already
//! lives in `forward::memit` for the multi-edit MEMIT path. `edit` is
//! the simpler single-edit primitive.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use larql_models::{ModelWeights, WeightArray};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// A single-layer, rank-1 FFN down_proj patch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditPatch {
    pub version: u32,
    pub layer: usize,
    pub module: String,
    /// Hidden size (= d.len()).
    pub hidden_size: usize,
    /// Intermediate size (= k_norm.len()).
    pub intermediate_size: usize,
    /// Binary-search scale that landed the edit (informational).
    pub scale: f32,
    /// Provenance — source prompt, old/new tokens, crown-delta. Optional.
    #[serde(default)]
    pub provenance: PatchProvenance,
    /// Output-delta direction. Shape: [hidden_size].
    #[serde(skip)]
    pub d: Vec<f32>,
    /// Pre-normalized key: k / (k · k). Shape: [intermediate_size].
    #[serde(skip)]
    pub k_norm: Vec<f32>,
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

/// Binary patch file magic. 8 bytes: "LQPATCH\0".
const PATCH_MAGIC: &[u8; 8] = b"LQPATCH\0";

/// Write an `EditPatch` to disk.
///
/// File layout:
///   8 bytes  : magic "LQPATCH\0"
///   4 bytes  : meta_json_len (u32, little-endian)
///   N bytes  : meta JSON (UTF-8)
///   4 bytes  : d_len  (u32 = hidden_size)
///   N*4 bytes: d   as f32 little-endian
///   4 bytes  : k_len (u32 = intermediate_size)
///   N*4 bytes: k_norm as f32 little-endian
pub fn write_patch(path: impl AsRef<Path>, patch: &EditPatch) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(PATCH_MAGIC)?;

    let meta = EditPatch {
        d: Vec::new(),
        k_norm: Vec::new(),
        ..patch.clone()
    };
    let meta_json = serde_json::to_vec(&meta)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    w.write_all(&(meta_json.len() as u32).to_le_bytes())?;
    w.write_all(&meta_json)?;

    w.write_all(&(patch.d.len() as u32).to_le_bytes())?;
    for &v in &patch.d {
        w.write_all(&v.to_le_bytes())?;
    }
    w.write_all(&(patch.k_norm.len() as u32).to_le_bytes())?;
    for &v in &patch.k_norm {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

/// Read an `EditPatch` from disk.
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

    let d_len = read_u32(&mut r)? as usize;
    patch.d = read_f32s(&mut r, d_len)?;
    let k_len = read_u32(&mut r)? as usize;
    patch.k_norm = read_f32s(&mut r, k_len)?;

    if patch.d.len() != patch.hidden_size || patch.k_norm.len() != patch.intermediate_size {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "shape mismatch: d={} (hidden={}), k={} (intermediate={})",
                patch.d.len(), patch.hidden_size, patch.k_norm.len(), patch.intermediate_size
            ),
        ));
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

/// Compute the rank-1 edit patch from a captured key and desired output delta.
///
/// `k`     : the FFN intermediate activation vector (last-token position)
///           at the crown layer for the SOURCE prompt. Length = intermediate_size.
/// `d`     : the desired additional contribution to the FFN output at that
///           position. Length = hidden_size.
/// `scale` : how much of `d` to actually use (the caller decides after
///           a binary/linear scale search).
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
        version: 1,
        layer,
        module: "down_proj".to_string(),
        hidden_size: d.len(),
        intermediate_size: k.len(),
        scale,
        provenance,
        d: d_scaled,
        k_norm,
    }
}

/// Apply a patch to a model's `down_proj` weight at the target layer,
/// in-place. `ΔW = d ⊗ k_norm`; the existing down_proj gets this outer
/// product added. Reversible by calling with negated `d`.
pub fn apply_patch(weights: &mut ModelWeights, patch: &EditPatch) -> Result<(), String> {
    let w_down_key = weights.arch.ffn_down_key(patch.layer);
    let existing = weights
        .tensors
        .get(&w_down_key)
        .ok_or_else(|| format!("apply_patch: W_down not found at {w_down_key}"))?;
    let (rows, cols) = (existing.shape()[0], existing.shape()[1]);

    // down_proj shape is either [hidden, intermediate] or [intermediate, hidden]
    // depending on how the model stores it. We detect by size matching.
    let hidden = patch.hidden_size;
    let intermediate = patch.intermediate_size;
    let (mut updated, transposed) = if rows == hidden && cols == intermediate {
        (existing.as_standard_layout().to_owned(), false)
    } else if rows == intermediate && cols == hidden {
        (existing.as_standard_layout().to_owned(), true)
    } else {
        return Err(format!(
            "apply_patch: W_down shape {rows}x{cols} doesn't match patch ({hidden}x{intermediate})"
        ));
    };

    // Build the rank-1 outer product: delta[i,j] = d[i] * k_norm[j] (canonical),
    // or d[j] * k_norm[i] if transposed layout.
    let d = Array1::from(patch.d.clone());
    let k = Array1::from(patch.k_norm.clone());
    if !transposed {
        // delta = d * k^T  →  shape (hidden, intermediate)
        let delta: Array2<f32> = outer(&d, &k);
        updated = &updated + &delta;
    } else {
        // delta = k * d^T  →  shape (intermediate, hidden)
        let delta: Array2<f32> = outer(&k, &d);
        updated = &updated + &delta;
    }

    // Install back into the tensor map as a new ArcArray2.
    let updated_weight: WeightArray = updated.into_shared();
    weights.tensors.insert(w_down_key, updated_weight);
    Ok(())
}

fn outer(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    let a_col = a.view().insert_axis(ndarray::Axis(1)); // (n, 1)
    let b_row = b.view().insert_axis(ndarray::Axis(0)); // (1, m)
    a_col.dot(&b_row)
}
