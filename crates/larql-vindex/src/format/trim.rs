//! Write a vocabulary-trimmed vindex — the focused-slice compiler.
//!
//! `larql slice` carves by tensor ROLE and keeps every feature and every
//! vocabulary row. This module carves the other way: it keeps every role
//! and drops the per-layer-embedding rows a domain never uses.
//!
//! # Why this file and not FFN features
//!
//! On the Gemma 4 e2b vindex the three FFN matrices are 3.11 GB (27% of
//! resident bytes) while `ple_weights.bin` alone is 4.70 GB (42%). Two
//! measurements (2026-09-05/06, recorded as PA-61..PA-63 in the
//! focused-pruning task) set the direction:
//!
//!   * dropping decoder layers destroys the model at every band tried, so
//!     a focused tier keeps all layers;
//!   * gating PLE vocabulary rows leaves a narrow domain intact even with
//!     98.5% of the table withheld, while out-of-domain probes degrade in
//!     a readable order.
//!
//! The gate used for that study was an env knob that skipped rows at
//! read time; the bytes stayed on disk. This is the compiler that
//! actually removes them.
//!
//! # What it writes
//!
//! `ple_weights.bin` is a concatenation of f16 tensors described by
//! `weight_manifest.json`: a global projection, the big
//! `[vocab, ple_dim·layers]` embedding table, then two small tensors per
//! layer. Only the big table is rewritten — its kept rows are copied as
//! raw bytes, so a trimmed row is bit-identical to its source and no f16
//! round-trip happens. Everything before and after it is copied verbatim
//! and the manifest offsets are rewritten to match.
//!
//! Alongside it the trimmer writes a dense `vocab → row` map as an
//! ordinary `kind::VECTOR` manifest entry ([`PLE_VOCAB_MAP_KEY`]), one
//! f32 per vocabulary id holding either the new row index or
//! [`PLE_VOCAB_DROPPED`] for a row that is gone. A vector entry needs no
//! loader changes: it arrives in `ModelWeights::vectors` like any norm
//! weight. Row indices are integers below 2^24, so f32 holds them
//! exactly.
//!
//! The forward path (`larql_compute::forward::ple`) consults the map when
//! it is present: a dropped token keeps stream 1, the projection of its
//! main embedding, and loses only its per-layer row — which is exactly
//! what the env-knob study measured.
//!
//! Every other file is hard-linked into the destination when the
//! filesystem allows it and copied otherwise, so a trimmed vindex beside
//! its source costs only the bytes it actually rewrites.

use std::collections::BTreeSet;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::VindexError;
use crate::format::filenames::*;
use crate::format::weights::write_f32::{kind, WeightEntry};

/// Manifest key of the dense `vocab → row` map written beside a trimmed
/// per-layer embedding table. Absent from an untrimmed vindex, and the
/// forward path treats absence as "every row is present".
pub const PLE_VOCAB_MAP_KEY: &str = "per_layer_embed_vocab_map";

/// Map value for a vocabulary id whose per-layer row was dropped.
/// Negative so it can never be mistaken for a row index.
pub const PLE_VOCAB_DROPPED: f32 = -1.0;

/// How a corpus-derived keep-set was built, for the caller to report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeepSetOrigin {
    /// Distinct ids the tokenizer actually emitted for the corpus.
    pub from_corpus: usize,
    /// Special/added tokens folded in regardless of the corpus.
    pub specials: usize,
    /// Ids added by `--keep-bytes`, the single-byte fallback pieces.
    pub byte_fallback: usize,
    pub total: usize,
    pub corpus_lines: usize,
}

/// Build a keep-set by tokenizing `corpus` with the vindex's own
/// tokenizer — the honest form of "which rows does this domain use".
///
/// Substring matching against `tokenizer.json` over-approximates badly:
/// it keeps every vocabulary entry that happens to occur anywhere in the
/// text, including pieces the tokenizer would never emit for it. Running
/// the real tokenizer keeps exactly the rows the domain's prompts will
/// look up.
///
/// Each line is encoded separately so a corpus reads as a list of
/// prompts rather than one long document; BPE merges do not cross the
/// line break that way, which matches how the prompts actually arrive.
///
/// `keep_specials` folds in every added/special token — the chat markers
/// among them. Dropping `<|turn>` from a chat model's table is not a
/// focused slice, it is a broken one, so this defaults on at the call
/// sites. `keep_byte_fallback` additionally keeps single-character
/// pieces, which bounds how badly unseen text degrades at the cost of a
/// larger table.
pub fn keep_set_from_corpus(
    tokenizer: &tokenizers::Tokenizer,
    corpus: &str,
    keep_specials: bool,
    keep_byte_fallback: bool,
) -> Result<(KeepSet, KeepSetOrigin), VindexError> {
    let mut ids: BTreeSet<u32> = BTreeSet::new();
    let mut lines = 0usize;
    let encode = |text: &str, ids: &mut BTreeSet<u32>| -> Result<(), VindexError> {
        // add_special_tokens: the encoder adds whatever the tokenizer's
        // post-processor would add in real use, so a BOS the runtime
        // sends is not missing from the table.
        let enc = tokenizer
            .encode(text, true)
            .map_err(|e| VindexError::Parse(format!("tokenizing corpus: {e}")))?;
        ids.extend(enc.get_ids().iter().copied());
        Ok(())
    };

    for line in corpus.lines() {
        if line.trim().is_empty() {
            continue;
        }
        lines += 1;
        encode(line, &mut ids)?;
    }

    // Then the whole corpus in one pass. Splitting on lines throws the
    // newline characters away, and a chat template is mostly newlines —
    // the first version of this function dropped token 107 ("\n") from a
    // Gemma 4 keep-set while faithfully keeping `<|turn>`, which would
    // have shipped a slice whose own prompt format had no rows. The
    // whole-corpus pass also recovers merges that span a line break.
    encode(corpus, &mut ids)?;
    let from_corpus = ids.len();

    let mut specials = 0usize;
    if keep_specials {
        // `get_added_tokens_decoder` is keyed BY id: (id, AddedToken).
        for (id, _token) in tokenizer.get_added_tokens_decoder() {
            if ids.insert(id) {
                specials += 1;
            }
        }
    }

    let mut byte_fallback = 0usize;
    if keep_byte_fallback {
        for (piece, id) in tokenizer.get_vocab(true) {
            if piece.chars().count() == 1 && ids.insert(id) {
                byte_fallback += 1;
            }
        }
    }

    if ids.is_empty() {
        return Err(VindexError::Parse(
            "corpus produced no token ids — nothing to keep".into(),
        ));
    }
    let origin = KeepSetOrigin {
        from_corpus,
        specials,
        byte_fallback,
        total: ids.len(),
        corpus_lines: lines,
    };
    Ok((KeepSet::Ids(ids), origin))
}

/// Which vocabulary rows a trim keeps.
#[derive(Debug, Clone)]
pub enum KeepSet {
    /// Keep token ids strictly below `n` — a frequency proxy, useful for
    /// staircase studies but not a domain vocabulary.
    TopN(usize),
    /// Keep exactly these ids. Ids at or beyond the table's row count are
    /// ignored rather than treated as an error, so a keep-set built
    /// against a different tokenizer revision still produces a usable
    /// artifact.
    Ids(BTreeSet<u32>),
}

impl KeepSet {
    fn resolve(&self, vocab: usize) -> Vec<u32> {
        match self {
            Self::TopN(n) => (0..vocab.min(*n) as u32).collect(),
            Self::Ids(ids) => ids
                .iter()
                .copied()
                .filter(|i| (*i as usize) < vocab)
                .collect(),
        }
    }
}

/// What a trim did, for the caller to print or record as a receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrimReport {
    pub vocab_total: usize,
    pub vocab_kept: usize,
    pub ple_bytes_before: u64,
    pub ple_bytes_after: u64,
    pub walk_index_dropped: bool,
    pub walk_index_bytes: u64,
    /// Files hard-linked rather than copied.
    pub linked_files: usize,
    pub copied_files: usize,
}

impl TrimReport {
    /// Total bytes removed from the source vindex.
    pub fn bytes_saved(&self) -> u64 {
        self.ple_bytes_before.saturating_sub(self.ple_bytes_after) + self.walk_index_bytes
    }
}

/// Options for [`trim_vindex`].
#[derive(Debug, Clone)]
pub struct TrimOptions {
    pub keep: KeepSet,
    /// Also drop `down_features.bin` / `down_meta.bin`. Measured as
    /// answer-preserving (PA-62), at the cost of the FFN planner falling
    /// to its dense last-resort rung.
    pub drop_walk_index: bool,
}

/// Write a vocabulary-trimmed copy of `src` into `dst`.
///
/// `dst` must not already exist. Returns what was kept and what it cost.
pub fn trim_vindex(src: &Path, dst: &Path, opts: &TrimOptions) -> Result<TrimReport, VindexError> {
    if dst.exists() {
        return Err(VindexError::Parse(format!(
            "destination already exists: {}",
            dst.display()
        )));
    }
    let manifest_path = src.join(WEIGHT_MANIFEST_JSON);
    let manifest_bytes = std::fs::read(&manifest_path)?;
    let mut entries: Vec<WeightEntry> = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| VindexError::Parse(format!("{WEIGHT_MANIFEST_JSON}: {e}")))?;

    // The big table is the only PLE entry with vocab-many rows; find it by
    // shape rather than by key so this does not hard-code one arch's
    // tensor name.
    let ple_entries: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.file == PLE_WEIGHTS_BIN)
        .map(|(i, _)| i)
        .collect();
    if ple_entries.is_empty() {
        return Err(VindexError::Parse(format!(
            "{} has no {PLE_WEIGHTS_BIN} entries — nothing to trim",
            src.display()
        )));
    }
    let table_idx = *ple_entries
        .iter()
        .max_by_key(|i| entries[**i].length)
        .expect("ple_entries is non-empty");
    let (vocab, cols) = {
        let e = &entries[table_idx];
        if e.shape.len() != 2 {
            return Err(VindexError::Parse(format!(
                "{} is not 2D — cannot trim",
                e.key
            )));
        }
        (e.shape[0], e.shape[1])
    };
    let row_bytes = entries[table_idx].length as usize / vocab.max(1);
    if row_bytes == 0 || row_bytes * vocab != entries[table_idx].length as usize {
        return Err(VindexError::Parse(format!(
            "{} length {} is not a whole number of {vocab} rows",
            entries[table_idx].key, entries[table_idx].length
        )));
    }

    let kept = opts.keep.resolve(vocab);
    if kept.is_empty() {
        return Err(VindexError::Parse(
            "keep-set is empty — a trim must keep at least one vocabulary row".into(),
        ));
    }

    std::fs::create_dir_all(dst)?;

    // ── 1. every file except the one we rewrite ──
    let mut report = TrimReport {
        vocab_total: vocab,
        vocab_kept: kept.len(),
        ple_bytes_before: std::fs::metadata(src.join(PLE_WEIGHTS_BIN))?.len(),
        ple_bytes_after: 0,
        walk_index_dropped: opts.drop_walk_index,
        walk_index_bytes: 0,
        linked_files: 0,
        copied_files: 0,
    };
    let dropped_files: &[&str] = if opts.drop_walk_index {
        &[PLE_WEIGHTS_BIN, DOWN_FEATURES_BIN, DOWN_META_BIN]
    } else {
        &[PLE_WEIGHTS_BIN]
    };
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy().to_string();
        if dropped_files.contains(&name.as_str()) {
            if name != PLE_WEIGHTS_BIN {
                report.walk_index_bytes += entry.metadata()?.len();
            }
            continue;
        }
        if name == WEIGHT_MANIFEST_JSON || name == INDEX_JSON {
            continue; // rewritten below
        }
        let from = entry.path();
        let to = dst.join(&name);
        if entry.file_type()?.is_dir() {
            copy_dir(&from, &to, &mut report)?;
        } else {
            link_or_copy(&from, &to, &mut report)?;
        }
    }

    // ── 2. the trimmed ple_weights.bin ──
    let src_file = std::fs::File::open(src.join(PLE_WEIGHTS_BIN))?;
    let src_map = unsafe { memmap2::Mmap::map(&src_file)? };
    let out_path = dst.join(PLE_WEIGHTS_BIN);
    let mut out = BufWriter::with_capacity(8 << 20, std::fs::File::create(&out_path)?);
    let mut offset: u64 = 0;

    // Entries are written in manifest order so offsets stay monotonic.
    let mut order: Vec<usize> = ple_entries.clone();
    order.sort_by_key(|i| entries[*i].offset);

    let mut vocab_map = vec![PLE_VOCAB_DROPPED; vocab];
    for (row, id) in kept.iter().enumerate() {
        vocab_map[*id as usize] = row as f32;
    }

    for idx in &order {
        let (start, len) = {
            let e = &entries[*idx];
            (e.offset as usize, e.length as usize)
        };
        if start + len > src_map.len() {
            return Err(VindexError::Parse(format!(
                "{} runs past the end of {PLE_WEIGHTS_BIN}",
                entries[*idx].key
            )));
        }
        if *idx == table_idx {
            for id in &kept {
                let row_start = start + (*id as usize) * row_bytes;
                out.write_all(&src_map[row_start..row_start + row_bytes])?;
            }
            let written = (kept.len() * row_bytes) as u64;
            let e = &mut entries[*idx];
            e.offset = offset;
            e.length = written;
            e.shape = vec![kept.len(), cols];
            offset += written;
        } else {
            out.write_all(&src_map[start..start + len])?;
            let e = &mut entries[*idx];
            e.offset = offset;
            offset += len as u64;
        }
    }

    // ── 3. the vocab → row map, as a plain vector entry ──
    let mut map_bytes = Vec::with_capacity(vocab * 4);
    for v in &vocab_map {
        map_bytes.extend_from_slice(&v.to_le_bytes());
    }
    out.write_all(&map_bytes)?;
    entries.push(WeightEntry {
        key: PLE_VOCAB_MAP_KEY.into(),
        kind: kind::VECTOR.into(),
        shape: vec![vocab],
        offset,
        length: map_bytes.len() as u64,
        file: PLE_WEIGHTS_BIN.into(),
    });
    offset += map_bytes.len() as u64;
    out.flush()?;
    drop(out);
    report.ple_bytes_after = offset;

    // ── 4. manifest + index.json ──
    std::fs::write(
        dst.join(WEIGHT_MANIFEST_JSON),
        serde_json::to_vec_pretty(&entries).map_err(|e| VindexError::Parse(e.to_string()))?,
    )?;

    let index_bytes = std::fs::read(src.join(INDEX_JSON))?;
    let mut index: serde_json::Value = serde_json::from_slice(&index_bytes)
        .map_err(|e| VindexError::Parse(format!("{INDEX_JSON}: {e}")))?;
    if let Some(obj) = index.as_object_mut() {
        // `vocab_size` deliberately stays at the model's true vocabulary:
        // the embedding table and lm_head are untouched, so the model can
        // still tokenize and emit every token. Only per-layer rows went.
        obj.insert(
            "ple_vocab_trim".into(),
            serde_json::json!({
                "vocab_total": vocab,
                "vocab_kept": kept.len(),
                "ple_bytes_before": report.ple_bytes_before,
                "ple_bytes_after": report.ple_bytes_after,
                "walk_index_dropped": report.walk_index_dropped,
                "map_key": PLE_VOCAB_MAP_KEY,
            }),
        );
        if opts.drop_walk_index {
            obj.insert("has_walk_index".into(), serde_json::json!(false));
        }
    }
    std::fs::write(
        dst.join(INDEX_JSON),
        serde_json::to_vec_pretty(&index).map_err(|e| VindexError::Parse(e.to_string()))?,
    )?;

    Ok(report)
}

fn copy_dir(from: &Path, to: &Path, report: &mut TrimReport) -> Result<(), VindexError> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let child_to = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &child_to, report)?;
        } else {
            link_or_copy(&entry.path(), &child_to, report)?;
        }
    }
    Ok(())
}

/// Hard-link when the filesystem allows it, copy otherwise. A trimmed
/// vindex beside its source should not cost another 6 GB of unchanged
/// weights.
fn link_or_copy(from: &Path, to: &Path, report: &mut TrimReport) -> Result<(), VindexError> {
    match std::fs::hard_link(from, to) {
        Ok(()) => {
            report.linked_files += 1;
            Ok(())
        }
        Err(_) => {
            std::fs::copy(from, to)?;
            report.copied_files += 1;
            Ok(())
        }
    }
}

/// Read a keep-set from a file of whitespace-separated token ids.
pub fn keep_set_from_file(path: &Path) -> Result<KeepSet, VindexError> {
    let text = std::fs::read_to_string(path)?;
    let ids: BTreeSet<u32> = text
        .split_whitespace()
        .filter_map(|t| t.parse::<u32>().ok())
        .collect();
    if ids.is_empty() {
        return Err(VindexError::Parse(format!(
            "{} contained no token ids",
            path.display()
        )));
    }
    Ok(KeepSet::Ids(ids))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_fixture(dir: &Path, vocab: usize, cols: usize) {
        std::fs::create_dir_all(dir).unwrap();
        let row_bytes = cols * 2;
        // proj tensor, then the table, then one per-layer tensor.
        let proj = vec![1u8; 8];
        let mut table = Vec::with_capacity(vocab * row_bytes);
        for id in 0..vocab {
            table.extend(std::iter::repeat_n(id as u8, row_bytes));
        }
        let tail = vec![7u8; 6];
        let mut blob = Vec::new();
        blob.extend_from_slice(&proj);
        blob.extend_from_slice(&table);
        blob.extend_from_slice(&tail);
        std::fs::write(dir.join(PLE_WEIGHTS_BIN), &blob).unwrap();

        let entries = vec![
            WeightEntry {
                key: "per_layer_model_projection.weight".into(),
                kind: kind::TENSOR_F16.into(),
                shape: vec![2, 2],
                offset: 0,
                length: proj.len() as u64,
                file: PLE_WEIGHTS_BIN.into(),
            },
            WeightEntry {
                key: "embed_tokens_per_layer.weight".into(),
                kind: kind::TENSOR_F16.into(),
                shape: vec![vocab, cols],
                offset: proj.len() as u64,
                length: table.len() as u64,
                file: PLE_WEIGHTS_BIN.into(),
            },
            WeightEntry {
                key: "layers.0.per_layer_projection.weight".into(),
                kind: kind::TENSOR_F16.into(),
                shape: vec![3, 1],
                offset: (proj.len() + table.len()) as u64,
                length: tail.len() as u64,
                file: PLE_WEIGHTS_BIN.into(),
            },
        ];
        std::fs::write(
            dir.join(WEIGHT_MANIFEST_JSON),
            serde_json::to_vec(&entries).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join(INDEX_JSON), br#"{"vocab_size":8}"#).unwrap();
        std::fs::write(dir.join(DOWN_FEATURES_BIN), vec![3u8; 32]).unwrap();
        std::fs::write(dir.join(DOWN_META_BIN), vec![4u8; 16]).unwrap();
        std::fs::write(dir.join(TOKENIZER_JSON), b"{}").unwrap();
    }

    fn read_entries(dir: &Path) -> Vec<WeightEntry> {
        serde_json::from_slice(&std::fs::read(dir.join(WEIGHT_MANIFEST_JSON)).unwrap()).unwrap()
    }

    #[test]
    fn kept_rows_are_copied_byte_for_byte_and_offsets_stay_consistent() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        let (vocab, cols) = (8usize, 4usize);
        write_fixture(&src, vocab, cols);

        let keep: BTreeSet<u32> = [1u32, 5, 6].into_iter().collect();
        let report = trim_vindex(
            &src,
            &dst,
            &TrimOptions {
                keep: KeepSet::Ids(keep),
                drop_walk_index: false,
            },
        )
        .unwrap();

        assert_eq!(report.vocab_total, 8);
        assert_eq!(report.vocab_kept, 3);

        let entries = read_entries(&dst);
        let table = entries
            .iter()
            .find(|e| e.key == "embed_tokens_per_layer.weight")
            .unwrap();
        assert_eq!(table.shape, vec![3, cols]);

        // Every entry's declared range must land inside the file, and the
        // kept rows must be the source rows unchanged.
        let blob = std::fs::read(dst.join(PLE_WEIGHTS_BIN)).unwrap();
        for e in entries.iter().filter(|e| e.file == PLE_WEIGHTS_BIN) {
            assert!(
                e.offset as usize + e.length as usize <= blob.len(),
                "{} runs past the file",
                e.key
            );
        }
        let row_bytes = cols * 2;
        for (row, id) in [1u8, 5, 6].iter().enumerate() {
            let start = table.offset as usize + row * row_bytes;
            assert!(blob[start..start + row_bytes].iter().all(|b| b == id));
        }
    }

    #[test]
    fn the_vocab_map_points_at_new_rows_and_marks_dropped_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        write_fixture(&src, 8, 4);

        trim_vindex(
            &src,
            &dst,
            &TrimOptions {
                keep: KeepSet::Ids([1u32, 5, 6].into_iter().collect()),
                drop_walk_index: false,
            },
        )
        .unwrap();

        let entries = read_entries(&dst);
        let map_entry = entries
            .iter()
            .find(|e| e.key == PLE_VOCAB_MAP_KEY)
            .expect("map entry written");
        assert_eq!(map_entry.kind, kind::VECTOR);
        assert_eq!(map_entry.shape, vec![8]);

        let blob = std::fs::read(dst.join(PLE_WEIGHTS_BIN)).unwrap();
        let start = map_entry.offset as usize;
        let map: Vec<f32> = blob[start..start + map_entry.length as usize]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(map[1], 0.0);
        assert_eq!(map[5], 1.0);
        assert_eq!(map[6], 2.0);
        for dropped in [0usize, 2, 3, 4, 7] {
            assert_eq!(map[dropped], PLE_VOCAB_DROPPED);
        }
    }

    #[test]
    fn dropping_the_walk_index_removes_both_files_and_records_the_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        write_fixture(&src, 8, 4);

        let report = trim_vindex(
            &src,
            &dst,
            &TrimOptions {
                keep: KeepSet::TopN(4),
                drop_walk_index: true,
            },
        )
        .unwrap();

        assert!(!dst.join(DOWN_FEATURES_BIN).exists());
        assert!(!dst.join(DOWN_META_BIN).exists());
        assert_eq!(report.walk_index_bytes, 48);
        assert!(dst.join(TOKENIZER_JSON).exists(), "other files survive");

        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dst.join(INDEX_JSON)).unwrap()).unwrap();
        assert_eq!(index["has_walk_index"], serde_json::json!(false));
        assert_eq!(index["ple_vocab_trim"]["vocab_kept"], serde_json::json!(4));
        // The true vocabulary is untouched: lm_head can still emit anything.
        assert_eq!(index["vocab_size"], serde_json::json!(8));
    }

    #[test]
    fn an_empty_keep_set_is_rejected_rather_than_writing_a_dead_artifact() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        write_fixture(&src, 8, 4);

        let err = trim_vindex(
            &src,
            &dst,
            &TrimOptions {
                keep: KeepSet::Ids(BTreeSet::new()),
                drop_walk_index: false,
            },
        );
        assert!(err.is_err());
    }

    /// A chat template is mostly newlines. Splitting the corpus on lines
    /// throws those characters away, and the first version of
    /// `keep_set_from_corpus` shipped a keep-set that held `<|turn>` but
    /// not `"\n"` — a slice whose own prompt format had no rows. The
    /// whole-corpus pass is what prevents that, so assert on it directly.
    #[test]
    fn a_corpus_keep_set_holds_the_newline_rows_its_chat_template_needs() {
        let tok = match tokenizers::Tokenizer::from_file(
            std::path::Path::new(&std::env::var("HOME").unwrap_or_default())
                .join("vindex/gemma4-full-b20ff753/tokenizer.json"),
        ) {
            Ok(t) => t,
            // The real tokenizer is a local asset, not a repo fixture.
            Err(_) => return,
        };
        let corpus = "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n";
        let (keep, origin) = keep_set_from_corpus(&tok, corpus, true, false).unwrap();
        let KeepSet::Ids(ids) = keep else {
            panic!("expected an id set")
        };
        let newline = tok
            .token_to_id("\n")
            .expect("tokenizer has a newline token");
        assert!(
            ids.contains(&newline),
            "newline row {newline} was dropped from a chat-templated corpus"
        );
        assert!(origin.total >= origin.from_corpus);
    }

    #[test]
    fn ids_beyond_the_table_are_ignored_not_fatal() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        write_fixture(&src, 8, 4);

        let report = trim_vindex(
            &src,
            &dst,
            &TrimOptions {
                keep: KeepSet::Ids([2u32, 99, 1000].into_iter().collect()),
                drop_walk_index: false,
            },
        )
        .unwrap();
        assert_eq!(report.vocab_kept, 1);
    }
}
