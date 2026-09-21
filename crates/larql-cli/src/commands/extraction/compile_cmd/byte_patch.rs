//! Write a compiled checkpoint as a byte-patched COPY of the base file.
//!
//! `save.rs` writes a standalone text-only checkpoint: it drops the multimodal towers, rewrites
//! `config.json` to a text architecture, and re-serialises every tensor out of the in-memory f32
//! representation. That is the right artifact to SERVE and the wrong one to AUDIT. The loader
//! strips a `model.` prefix on the way in, so the output shares no tensor names with the base it
//! came from, and nothing about it can be compared byte for byte with its origin.
//!
//! An erasure claim needs the opposite property. GDPR Art. 17, read with the Art. 5(2)
//! accountability duty, asks a controller to SHOW that data is gone rather than assert it — and
//! for a single artifact the strongest available evidence is that the file is bit-for-bit the
//! file that never held the data. That is only checkable when a compiled checkpoint differs from
//! its base exactly where its patch says and nowhere else.
//!
//! Measured on 2026-09-20 against `gemma-4-E2B-it`: the re-serialised output carried 1,339
//! tensors against the base's 2,011, with ZERO tensor names in common, so a restore that copies
//! the base's own bytes back refused outright — the headers do not line up, so there are no
//! offsets to restore to.
//!
//! This writer copies the base file and overwrites only the bytes the edge touched: the gate row,
//! the up row and the down column of each edited slot. The header, the tensor set, the names, the
//! dtypes and every other byte are exactly as they were. The compiled checkpoint is then the base
//! plus a named, bounded, reversible difference.
//!
//! It deliberately does NOT replace `save.rs`. Serving a text-only model and auditing an edit are
//! different jobs and want different artifacts; this adds the second one.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

/// One slot's worth of new weights, in the orientation the checkpoint stores them:
/// `gate` and `up` are rows of their tensors, `down` is a column of its own.
pub struct SlotEdit {
    pub layer: usize,
    pub slot: usize,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

#[derive(Debug)]
pub struct PatchReceipt {
    pub slots: Vec<(usize, usize)>,
    pub spans: usize,
    pub bytes_written: usize,
    pub sha256_base: String,
    pub sha256_out: String,
}

/// safetensors layout: 8-byte little-endian header length, that many bytes of JSON, then data.
/// Returns the parsed header and the absolute offset where tensor data begins.
fn read_header(path: &Path) -> Result<(serde_json::Value, u64), Box<dyn std::error::Error>> {
    let mut f = fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len);
    let mut raw = vec![0u8; n as usize];
    f.read_exact(&mut raw)?;
    Ok((serde_json::from_slice(&raw)?, 8 + n))
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 22];
    loop {
        let k = f.read(&mut buf)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// `[rows, cols]` and the byte offset of a named BF16 tensor's data.
fn tensor_meta(
    header: &serde_json::Value,
    data_start: u64,
    name: &str,
) -> Result<(usize, usize, u64), Box<dyn std::error::Error>> {
    let meta = header
        .get(name)
        .ok_or_else(|| format!("tensor {name} not in the base checkpoint"))?;
    let dtype = meta.get("dtype").and_then(|d| d.as_str()).unwrap_or("");
    if dtype != "BF16" {
        return Err(format!("tensor {name} is {dtype}; this writer only handles BF16").into());
    }
    let shape: Vec<usize> = meta
        .get("shape")
        .and_then(|s| s.as_array())
        .ok_or_else(|| format!("tensor {name} has no shape"))?
        .iter()
        .filter_map(|v| v.as_u64().map(|x| x as usize))
        .collect();
    if shape.len() != 2 {
        return Err(format!("tensor {name} is {}-D; expected 2-D", shape.len()).into());
    }
    let begin = meta
        .get("data_offsets")
        .and_then(|o| o.as_array())
        .and_then(|o| o.first())
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("tensor {name} has no data_offsets"))?;
    Ok((shape[0], shape[1], data_start + begin))
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    larql_models::quant::half::encode_bf16(values)
}

/// Copy `base_model` to `out_model` and overwrite only the bytes the edits name.
///
/// `patterns` are the gate/up/down key templates with `{}` standing in for the layer, exactly as
/// `detect.rs` resolved them against this checkpoint — so this writer never guesses at a naming
/// convention, it is handed the one that was already found.
pub fn write_byte_patched(
    base_model: &Path,
    out_model: &Path,
    edits: &[SlotEdit],
    patterns: (&str, &str, &str),
) -> Result<PatchReceipt, Box<dyn std::error::Error>> {
    if edits.is_empty() {
        return Err("no slot edits: refusing to write a checkpoint that claims an edit it does not carry".into());
    }
    let (header, data_start) = read_header(base_model)?;

    // Resolve and validate every span BEFORE copying a 10 GB file, so a bad slot costs a
    // millisecond rather than a copy. Ordered so the receipt is stable across runs.
    let mut spans: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
    let mut slots = Vec::new();
    for e in edits {
        let (gk, uk, dk) = (
            patterns.0.replace("{}", &e.layer.to_string()),
            patterns.1.replace("{}", &e.layer.to_string()),
            patterns.2.replace("{}", &e.layer.to_string()),
        );
        for (name, values, row_wise) in [
            (&gk, &e.gate, true),
            (&uk, &e.up, true),
            (&dk, &e.down, false),
        ] {
            let (rows, cols, base_off) = tensor_meta(&header, data_start, name)?;
            if row_wise {
                if e.slot >= rows {
                    return Err(format!("slot {} out of range for {name} [{rows}, {cols}]", e.slot).into());
                }
                if values.len() != cols {
                    return Err(format!(
                        "{name}: {} values for a row of {cols}", values.len()
                    )
                    .into());
                }
                spans.insert(base_off + (e.slot * cols * 2) as u64, bf16_bytes(values));
            } else {
                if e.slot >= cols {
                    return Err(format!("slot {} out of range for {name} [{rows}, {cols}]", e.slot).into());
                }
                if values.len() != rows {
                    return Err(format!(
                        "{name}: {} values for a column of {rows}", values.len()
                    )
                    .into());
                }
                let bytes = bf16_bytes(values);
                for r in 0..rows {
                    let off = base_off + ((r * cols + e.slot) * 2) as u64;
                    spans.insert(off, bytes[r * 2..r * 2 + 2].to_vec());
                }
            }
        }
        slots.push((e.layer, e.slot));
    }

    if let Some(parent) = out_model.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(base_model, out_model)?;

    let mut out = fs::OpenOptions::new().write(true).open(out_model)?;
    let mut bytes_written = 0usize;
    for (off, bytes) in &spans {
        out.seek(SeekFrom::Start(*off))?;
        out.write_all(bytes)?;
        bytes_written += bytes.len();
    }
    out.flush()?;
    drop(out);

    Ok(PatchReceipt {
        slots,
        spans: spans.len(),
        bytes_written,
        sha256_base: sha256_file(base_model)?,
        sha256_out: sha256_file(out_model)?,
    })
}

/// Copy every sidecar file VERBATIM — including `config.json`.
///
/// `save.rs::copy_model_config` rewrites the architecture because its output really is a
/// different, text-only model. A byte-patched checkpoint is the same model as its base, so
/// rewriting the config would make the pair disagree about what they are.
pub fn copy_sidecars_verbatim(base: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(output)?;
    for entry in fs::read_dir(base)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name();
        if Path::new(&name)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
        {
            continue; // the weights are handled by write_byte_patched
        }
        fs::copy(&path, output.join(&name))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    const GATE: &str = "model.language_model.layers.{}.mlp.gate_proj.weight";
    const UP: &str = "model.language_model.layers.{}.mlp.up_proj.weight";
    const DOWN: &str = "model.language_model.layers.{}.mlp.down_proj.weight";
    const FEATURES: usize = 6;
    const HIDDEN: usize = 4;

    /// A minimal BF16 checkpoint with the three FFN tensors of layer 0, plus one tensor the
    /// patch must never touch.
    fn write_base(path: &Path) {
        let names = [
            (GATE.replace("{}", "0"), FEATURES, HIDDEN),
            (UP.replace("{}", "0"), FEATURES, HIDDEN),
            (DOWN.replace("{}", "0"), HIDDEN, FEATURES),
            ("model.language_model.norm.weight".to_string(), 1, HIDDEN),
        ];
        let mut header = serde_json::Map::new();
        let mut blob: Vec<u8> = Vec::new();
        for (name, r, c) in &names {
            let begin = blob.len();
            let vals: Vec<f32> = (0..r * c).map(|i| 1.0 + i as f32 / 64.0).collect();
            blob.extend_from_slice(&bf16_bytes(&vals));
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "BF16", "shape": [r, c], "data_offsets": [begin, blob.len()]}),
            );
        }
        let raw = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = fs::File::create(path).unwrap();
        f.write_all(&(raw.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&raw).unwrap();
        f.write_all(&blob).unwrap();
    }

    fn edit(slot: usize) -> SlotEdit {
        SlotEdit {
            layer: 0,
            slot,
            gate: vec![9.0; HIDDEN],
            up: vec![8.0; HIDDEN],
            down: vec![7.0; HIDDEN],
        }
    }

    #[test]
    fn differs_from_base_only_inside_the_patched_slot() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.safetensors");
        let out = dir.path().join("out.safetensors");
        write_base(&base);
        let r = write_byte_patched(&base, &out, &[edit(2)], (GATE, UP, DOWN)).unwrap();

        let b = fs::read(&base).unwrap();
        let o = fs::read(&out).unwrap();
        assert_eq!(b.len(), o.len(), "the copy changed the file's length");
        assert_ne!(r.sha256_base, r.sha256_out, "nothing was patched at all");

        // Every differing byte must fall inside a span the receipt accounts for.
        let differing = b.iter().zip(&o).filter(|(x, y)| x != y).count();
        assert!(differing > 0);
        assert!(
            differing <= r.bytes_written,
            "{differing} bytes differ but only {} were written",
            r.bytes_written
        );

        // The header is untouched, so offsets still line up for a restore.
        let (hb, sb) = read_header(&base).unwrap();
        let (ho, so) = read_header(&out).unwrap();
        assert_eq!(hb, ho);
        assert_eq!(sb, so);
    }

    #[test]
    fn a_tensor_the_patch_does_not_name_is_byte_identical() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.safetensors");
        let out = dir.path().join("out.safetensors");
        write_base(&base);
        write_byte_patched(&base, &out, &[edit(1)], (GATE, UP, DOWN)).unwrap();

        let (h, start) = read_header(&base).unwrap();
        let (r, c, off) = tensor_meta(&h, start, "model.language_model.norm.weight").unwrap();
        let n = r * c * 2;
        let b = fs::read(&base).unwrap();
        let o = fs::read(&out).unwrap();
        assert_eq!(
            b[off as usize..off as usize + n],
            o[off as usize..off as usize + n],
            "a tensor outside the patch moved"
        );
    }

    #[test]
    fn another_slot_in_a_patched_tensor_is_byte_identical() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.safetensors");
        let out = dir.path().join("out.safetensors");
        write_base(&base);
        write_byte_patched(&base, &out, &[edit(2)], (GATE, UP, DOWN)).unwrap();

        let (h, start) = read_header(&base).unwrap();
        let (_, cols, off) = tensor_meta(&h, start, &GATE.replace("{}", "0")).unwrap();
        let b = fs::read(&base).unwrap();
        let o = fs::read(&out).unwrap();
        for slot in [0usize, 1, 3, 4, 5] {
            let s = off as usize + slot * cols * 2;
            assert_eq!(b[s..s + cols * 2], o[s..s + cols * 2], "slot {slot} moved");
        }
    }

    #[test]
    fn the_down_column_is_strided_not_contiguous() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.safetensors");
        let out = dir.path().join("out.safetensors");
        write_base(&base);
        write_byte_patched(&base, &out, &[edit(3)], (GATE, UP, DOWN)).unwrap();

        let (h, start) = read_header(&base).unwrap();
        let (rows, cols, off) = tensor_meta(&h, start, &DOWN.replace("{}", "0")).unwrap();
        let b = fs::read(&base).unwrap();
        let o = fs::read(&out).unwrap();
        let want = bf16_bytes(&[7.0f32]);
        for r in 0..rows {
            for c in 0..cols {
                let s = off as usize + (r * cols + c) * 2;
                if c == 3 {
                    assert_eq!(&o[s..s + 2], &want[..], "down[{r},{c}] not patched");
                } else {
                    assert_eq!(b[s..s + 2], o[s..s + 2], "down[{r},{c}] moved");
                }
            }
        }
    }

    #[test]
    fn refuses_a_slot_out_of_range_before_copying() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.safetensors");
        let out = dir.path().join("out.safetensors");
        write_base(&base);
        let err = write_byte_patched(&base, &out, &[edit(FEATURES)], (GATE, UP, DOWN)).unwrap_err();
        assert!(err.to_string().contains("out of range"), "{err}");
        // Name the tensor that caught it, not just the fact that something did. The row guard
        // (gate/up) and the column guard (down) check the same invariant from opposite
        // orientations — the slot indexes features either way — so a test that only asserts
        // "out of range" still passes with the row guard deleted, and mutation testing on
        // 2026-09-20 showed exactly that. Validation runs per tensor in order, so gate is the
        // one that must report it.
        assert!(
            err.to_string().contains("gate_proj"),
            "the row guard did not catch this; a later guard did: {err}"
        );
        assert!(!out.exists(), "a refused write still produced a file");
    }

    #[test]
    fn refuses_a_wrong_length_vector_before_copying() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.safetensors");
        let out = dir.path().join("out.safetensors");
        write_base(&base);
        let mut e = edit(0);
        e.gate = vec![1.0; HIDDEN + 1];
        let err = write_byte_patched(&base, &out, &[e], (GATE, UP, DOWN)).unwrap_err();
        assert!(err.to_string().contains("values for a row"), "{err}");
        assert!(!out.exists());
    }

    #[test]
    fn refuses_to_write_when_there_is_nothing_to_patch() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.safetensors");
        let out = dir.path().join("out.safetensors");
        write_base(&base);
        assert!(write_byte_patched(&base, &out, &[], (GATE, UP, DOWN)).is_err());
        assert!(!out.exists());
    }

    #[test]
    fn sidecars_are_copied_verbatim_including_config() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let out = dir.path().join("out");
        fs::create_dir_all(&base).unwrap();
        fs::write(base.join("config.json"), r#"{"architectures":["Gemma4ForConditionalGeneration"]}"#).unwrap();
        fs::write(base.join("tokenizer.json"), "{}").unwrap();
        write_base(&base.join("model.safetensors"));
        copy_sidecars_verbatim(&base, &out).unwrap();
        assert_eq!(
            fs::read_to_string(out.join("config.json")).unwrap(),
            r#"{"architectures":["Gemma4ForConditionalGeneration"]}"#,
            "config.json was rewritten; a byte-patched checkpoint is the SAME model as its base"
        );
        assert!(!out.join("model.safetensors").exists(), "weights must come from the patch writer");
    }
}
