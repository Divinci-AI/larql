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

/// Filename of the receipt a trim writes beside the artifact it produced.
pub const PRUNING_RECEIPT_JSON: &str = "pruning_receipt.json";

/// Schema tag, so a reader can tell which shape it is holding.
pub const PRUNING_RECEIPT_VERSION: &str = "larql.pruning_receipt.v1";

/// One structural assertion about the keep-set.
///
/// Structural, not behavioural: a passing check says the rows a prompt
/// format needs are present, never that the model still answers well.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageCheck {
    pub name: String,
    /// `required` checks decide `coverage.passed`. `informational` ones
    /// are reported and never block: a chat template carries optional
    /// branches (tool calling and its JSON scaffolding) that a focused
    /// slice is entitled to drop, and failing a build over those would
    /// be crying wolf.
    pub severity: String,
    /// What the check is asserting, in words, so a receipt reads without
    /// this source file next to it.
    pub asserts: String,
    pub required: usize,
    pub present: usize,
    /// Ids the keep-set is missing. Empty when the check passes.
    pub missing: Vec<u32>,
    pub passed: bool,
}

/// How a keep-set was derived — enough for a third party to rebuild it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeepSetProvenance {
    /// `corpus`, `file`, or `top_n`.
    pub method: String,
    pub total_ids: usize,
    /// SHA-256 over the sorted ids, newline-separated: the keep-set's
    /// identity, independent of how it was written down.
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corpus_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_corpus: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specials: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_fallback: Option<usize>,
}

/// What a trim removed, what it kept, and what that is checkable against.
///
/// The receipt deliberately separates three claims a reader might
/// conflate. It states what was removed (bytes, rows), it asserts that
/// the keep-set covers the serving prompt format (structural coverage),
/// and it carries behavioural evidence only when someone actually
/// measured it. `behavior.status` is `not_verified` until probe results
/// are attached, because a compiler cannot know whether the model still
/// answers — and a receipt that implied otherwise would be the exact
/// thing this project criticises elsewhere.
///
/// Signing and hash-chaining are not done here. The digests below are
/// what make the artifact independently reproducible: rerun `larql trim`
/// with the same source and keep-set and the output hashes must match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningReceipt {
    pub receipt_version: String,
    pub created_at: String,
    pub tool: serde_json::Value,
    pub source: serde_json::Value,
    pub output: serde_json::Value,
    pub keep_set: KeepSetProvenance,
    pub removed: serde_json::Value,
    pub coverage: serde_json::Value,
    pub behavior: serde_json::Value,
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
    /// Whether every coverage check passed. `None` when no receipt was
    /// written, so "not checked" never reads as "passed".
    pub coverage_passed: Option<bool>,
    /// The trimmed artifact's own vindex id. `None` when the source had
    /// no publish manifest to re-identify.
    pub new_vindex_sha256: Option<String>,
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
    /// Write `pruning_receipt.json` beside the artifact.
    pub write_receipt: bool,
    /// How the keep-set was derived, when the caller knows. Recorded in
    /// the receipt so a third party can rebuild the same set.
    pub keep_origin: Option<KeepSetOrigin>,
    /// Digest of the corpus the keep-set came from, when there was one.
    pub corpus_sha256: Option<String>,
    /// Probe results measured against the trimmed artifact by whoever
    /// ran them. Absent means the receipt says so rather than implying
    /// the model was checked.
    pub probe_results: Option<serde_json::Value>,
}

impl TrimOptions {
    /// Options with no receipt and no provenance — the minimum a trim
    /// needs. Callers that want a receipt fill the rest in.
    pub fn bare() -> Self {
        Self {
            keep: KeepSet::TopN(0),
            drop_walk_index: false,
            write_receipt: false,
            keep_origin: None,
            corpus_sha256: None,
            probe_results: None,
        }
    }
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
        coverage_passed: None,
        new_vindex_sha256: None,
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

    // ── 5. identity ──
    // Before the receipt, so the receipt records the artifact's own id.
    let kept_set_for_id: BTreeSet<u32> = kept.iter().copied().collect();
    let keep_sha = keep_set_digest(&kept_set_for_id);
    report.new_vindex_sha256 = rewrite_manifest(src, dst, &keep_sha, opts)?.map(|(sha, _)| sha);

    // ── 6. the receipt ──
    if opts.write_receipt {
        let kept_set: BTreeSet<u32> = kept.iter().copied().collect();
        let receipt = build_receipt(src, dst, &kept_set, &report, opts)?;
        report.coverage_passed = Some(
            receipt
                .coverage
                .get("passed")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        );
        std::fs::write(
            dst.join(PRUNING_RECEIPT_JSON),
            serde_json::to_vec_pretty(&receipt).map_err(|e| VindexError::Parse(e.to_string()))?,
        )?;
    }

    Ok(report)
}

/// Chat templates are built out of newlines and turn markers; this is
/// the file they live in.
const CHAT_TEMPLATE_JINJA: &str = "chat_template.jinja";

/// The publish manifest — vindex identity, not the weight manifest.
/// `filenames.rs` has no constant for it (it is written by the publish
/// path rather than the format writers), so it is named here.
const MANIFEST_JSON: &str = "manifest.json";

/// Remove jinja control and expression blocks, leaving the literal text
/// a template emits. Deliberately crude: it is a coverage heuristic, and
/// keeping slightly too much text only makes the assertion stricter.
fn strip_jinja(template: &str) -> String {
    let mut out = String::with_capacity(template.len());
    let bytes: Vec<char> = template.chars().collect();
    let mut i = 0usize;
    while i < bytes.len() {
        let two: String = bytes[i..(i + 2).min(bytes.len())].iter().collect();
        if two == "{{" || two == "{%" || two == "{#" {
            let close = match two.as_str() {
                "{{" => "}}",
                "{%" => "%}",
                _ => "#}",
            };
            let mut j = i + 2;
            while j + 1 < bytes.len() {
                let pair: String = bytes[j..j + 2].iter().collect();
                if pair == close {
                    break;
                }
                j += 1;
            }
            i = (j + 2).min(bytes.len());
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }

    // Collapse the indentation the jinja left behind. A template source
    // is indented for humans; the prompt it renders is not, and the
    // tokenizer has distinct pieces for a run of 11 newlines or 29
    // spaces. Asserting on those demands rows a rendered prompt never
    // uses — the second version of this check failed 40 of 43 on
    // exactly that. Runs collapse to at most a blank line and a single
    // space, which is what a rendered turn actually contains.
    let mut collapsed = String::with_capacity(out.len());
    let mut newlines = 0usize;
    let mut spaces = 0usize;
    for ch in out.chars() {
        match ch {
            '\n' => {
                spaces = 0;
                newlines += 1;
                if newlines <= 2 {
                    collapsed.push(ch);
                }
            }
            ' ' | '\t' => {
                newlines = 0;
                spaces += 1;
                if spaces == 1 {
                    collapsed.push(' ');
                }
            }
            _ => {
                newlines = 0;
                spaces = 0;
                collapsed.push(ch);
            }
        }
    }
    collapsed
}

/// Assert that the rows a serving prompt format needs are present.
///
/// This encodes a bug rather than a theory. The first corpus keep-set
/// built for this compiler tokenized line by line, which throws newline
/// characters away — it kept `<|turn>` and `<turn|>` faithfully and
/// dropped token 107, `"\n"`. A chat template is mostly newlines, so
/// that slice's own prompt format had no rows, and nothing about its
/// size would have revealed it. A keep-set therefore ships with a
/// coverage assertion, not just a count.
fn coverage_checks(dir: &Path, kept: &BTreeSet<u32>) -> Result<Vec<CoverageCheck>, VindexError> {
    let tokenizer = match crate::format::load::load_vindex_tokenizer(dir) {
        Ok(t) => t,
        // No tokenizer to check against: say so rather than claiming a pass.
        Err(_) => {
            return Ok(vec![CoverageCheck {
                name: "tokenizer".into(),
                severity: "required".into(),
                asserts: "the vindex carries a tokenizer to check the keep-set against".into(),
                required: 1,
                present: 0,
                missing: Vec::new(),
                passed: false,
            }])
        }
    };
    let mut checks = Vec::new();

    let special_ids: BTreeSet<u32> = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    let missing: Vec<u32> = special_ids.difference(kept).copied().collect();
    checks.push(CoverageCheck {
        name: "added_and_special_tokens".into(),
        severity: "required".into(),
        asserts: "every added/special token — the turn and tool markers among them —                   still has a per-layer row"
            .into(),
        required: special_ids.len(),
        present: special_ids.len() - missing.len(),
        passed: missing.is_empty(),
        missing,
    });

    // What the chat template is literally made of, with its jinja
    // stripped. Tokenizing the raw template instead asserts on `{%`,
    // `endfor` and variable names a prompt never carries — the first
    // version of this check did exactly that and reported 24/389 on a
    // slice that served correctly, which is a broken check, not a broken
    // artifact. The literal text is the part that reaches the model:
    // the turn markers, the role words, and the newlines between them.
    let template_path = dir.join(CHAT_TEMPLATE_JINJA);
    if let Ok(template) = std::fs::read_to_string(&template_path) {
        let literals = strip_jinja(&template);
        if let Ok(enc) = tokenizer.encode(literals.as_str(), false) {
            let ids: BTreeSet<u32> = enc.get_ids().iter().copied().collect();

            // Required: the whitespace the template is assembled from.
            // This is the bug that motivated the whole check — a
            // keep-set held `<|turn>` and `<turn|>` but not `"\n"`, so
            // the slice's own prompt format had no rows.
            let ws: BTreeSet<u32> = ids
                .iter()
                .copied()
                .filter(|id| {
                    tokenizer
                        .id_to_token(*id)
                        .is_some_and(|t| !t.is_empty() && t.chars().all(|c| c.is_whitespace()))
                })
                .collect();
            let missing: Vec<u32> = ws.difference(kept).copied().collect();
            checks.push(CoverageCheck {
                name: "chat_template_whitespace".into(),
                severity: "required".into(),
                asserts: "every whitespace piece the vindex's own chat_template.jinja \
                          renders — the newlines a turn is built from — still has a row"
                    .into(),
                required: ws.len(),
                present: ws.len() - missing.len(),
                passed: missing.is_empty(),
                missing,
            });

            // Informational: the rest of the template's literal text.
            // A template's tool-calling branch carries JSON scaffolding
            // (`properties`, `required`, `enum`) that a slice which
            // never calls tools is entitled to drop, so a miss here is
            // something to read, not something to fail on.
            let missing: Vec<u32> = ids.difference(kept).copied().collect();
            let present = ids.len() - missing.len();
            checks.push(CoverageCheck {
                name: "chat_template_literals".into(),
                severity: "informational".into(),
                asserts: "how much of the literal text of chat_template.jinja still has \
                          rows. Optional branches — tool calling and its JSON \
                          scaffolding — are legitimately absent from a focused slice, \
                          so read the missing list rather than treating this as a gate"
                    .into(),
                required: ids.len(),
                present,
                passed: missing.is_empty(),
                missing,
            });
        }
    }

    Ok(checks)
}

/// Build the receipt for a completed trim.
#[allow(clippy::too_many_arguments)]
fn build_receipt(
    src: &Path,
    dst: &Path,
    kept: &BTreeSet<u32>,
    report: &TrimReport,
    opts: &TrimOptions,
) -> Result<PruningReceipt, VindexError> {
    let keep_sha = keep_set_digest(kept);

    let sha = |dir: &Path, name: &str| -> Option<String> {
        crate::format::checksums::sha256_file(&dir.join(name)).ok()
    };

    let method = match (&opts.keep, &opts.keep_origin) {
        (_, Some(_)) => "corpus",
        (KeepSet::TopN(_), None) => "top_n",
        (KeepSet::Ids(_), None) => "file",
    };

    let checks = coverage_checks(src, kept)?;
    let all_passed = checks
        .iter()
        .filter(|c| c.severity == "required")
        .all(|c| c.passed);

    let source_index: serde_json::Value = std::fs::read(src.join(INDEX_JSON))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(serde_json::Value::Null);

    Ok(PruningReceipt {
        receipt_version: PRUNING_RECEIPT_VERSION.into(),
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_default(),
        tool: serde_json::json!({
            "name": "larql trim",
            "crate_version": env!("CARGO_PKG_VERSION"),
        }),
        source: serde_json::json!({
            "directory": src.file_name().map(|n| n.to_string_lossy().to_string()),
            "model": source_index.get("model"),
            "family": source_index.get("family"),
            "num_layers": source_index.get("num_layers"),
            "vocab_size": source_index.get("vocab_size"),
            "ple_weights_sha256": sha(src, PLE_WEIGHTS_BIN),
            "tokenizer_sha256": sha(src, TOKENIZER_JSON),
        }),
        output: serde_json::json!({
            "directory": dst.file_name().map(|n| n.to_string_lossy().to_string()),
            "ple_weights_sha256": sha(dst, PLE_WEIGHTS_BIN),
            "weight_manifest_sha256": sha(dst, WEIGHT_MANIFEST_JSON),
            "reproduce": "rerun `larql trim` with the same source and keep-set;                           these digests must match",
        }),
        keep_set: KeepSetProvenance {
            method: method.into(),
            total_ids: kept.len(),
            sha256: keep_sha,
            corpus_sha256: opts.corpus_sha256.clone(),
            tokenizer_sha256: sha(src, TOKENIZER_JSON),
            from_corpus: opts.keep_origin.as_ref().map(|o| o.from_corpus),
            specials: opts.keep_origin.as_ref().map(|o| o.specials),
            byte_fallback: opts.keep_origin.as_ref().map(|o| o.byte_fallback),
        },
        removed: serde_json::json!({
            "vocab_rows_total": report.vocab_total,
            "vocab_rows_kept": report.vocab_kept,
            "vocab_rows_dropped": report.vocab_total.saturating_sub(report.vocab_kept),
            "ple_bytes_before": report.ple_bytes_before,
            "ple_bytes_after": report.ple_bytes_after,
            "walk_index_dropped": report.walk_index_dropped,
            "walk_index_bytes": report.walk_index_bytes,
            "total_bytes_saved": report.bytes_saved(),
        }),
        coverage: serde_json::json!({
            "passed": all_passed,
            "scope": "structural only — these checks say the rows a prompt format needs                       are present, never that the model still answers correctly",
            "checks": checks,
        }),
        behavior: match &opts.probe_results {
            Some(results) => serde_json::json!({
                "status": "attached",
                "note": "probe results supplied by the caller; this compiler did not                          run them and does not vouch for them",
                "results": results,
            }),
            None => serde_json::json!({
                "status": "not_verified",
                "note": "no probes were run against this artifact. Removal and coverage                          are recorded above; whether the model still answers is a                          separate measurement, and this receipt does not claim it.",
            }),
        },
    })
}

/// SHA-256 over the sorted keep-set, newline-separated: the keep-set's
/// identity, independent of how it was written down. Shared by the
/// receipt and the artifact id so the two can never disagree.
fn keep_set_digest(kept: &BTreeSet<u32>) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write as _;
    let mut text = String::new();
    for id in kept {
        let _ = writeln!(text, "{id}");
    }
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

/// Give the trimmed artifact its OWN identity.
///
/// `manifest.json` carries `vindexSha256` / `shortId`, and Divinci gates
/// patch compatibility on them (the served-model == vindex eligibility
/// rule). A trim that hard-links the source manifest therefore ships an
/// artifact claiming to BE the vindex it was cut from — so a patch built
/// against the full vindex reads as compatible with a slice that no
/// longer has the rows it addresses, and Model Edits would apply it
/// silently. Trimming changes what the model answers, so it has to
/// change the identity too.
///
/// The new sha is derived, not random: sha256 over the source sha, the
/// keep-set digest and the walk-index flag. Two trims with the same
/// inputs produce the same id, which is what makes the receipt's
/// "rerun and compare" claim hold for the manifest as well as the blob.
fn rewrite_manifest(
    src: &Path,
    dst: &Path,
    keep_sha: &str,
    opts: &TrimOptions,
) -> Result<Option<(String, String)>, VindexError> {
    use sha2::{Digest, Sha256};

    let raw = match std::fs::read(src.join(MANIFEST_JSON)) {
        Ok(r) => r,
        // Not every vindex carries one; nothing to re-identify.
        Err(_) => return Ok(None),
    };
    let mut manifest: serde_json::Value = serde_json::from_slice(&raw)
        .map_err(|e| VindexError::Parse(format!("{MANIFEST_JSON}: {e}")))?;
    let Some(obj) = manifest.as_object_mut() else {
        return Ok(None);
    };

    let source_sha = obj
        .get("vindexSha256")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let source_short = obj
        .get("shortId")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let mut hasher = Sha256::new();
    hasher.update(source_sha.as_bytes());
    hasher.update(b"\0keep=");
    hasher.update(keep_sha.as_bytes());
    hasher.update(b"\0walk_index_dropped=");
    hasher.update(if opts.drop_walk_index { b"1" } else { b"0" });
    let new_sha = format!("{:x}", hasher.finalize());
    let new_short = new_sha[..8].to_string();

    // Re-checksum what actually landed: ple_weights.bin was rewritten and
    // the walk index may be gone, so the inherited map is wrong on both.
    let mut total: u64 = 0;
    if let Some(files) = obj.get_mut("files").and_then(|f| f.as_object_mut()) {
        let names: Vec<String> = files.keys().cloned().collect();
        for name in names {
            let path = dst.join(&name);
            if !path.exists() {
                files.remove(&name);
                continue;
            }
            if let Ok(sum) = crate::format::checksums::sha256_file(&path) {
                files.insert(name.clone(), serde_json::json!(sum));
            }
            total += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        }
    }

    obj.insert("vindexSha256".into(), serde_json::json!(new_sha));
    obj.insert("shortId".into(), serde_json::json!(new_short));
    if total > 0 {
        obj.insert("totalBytes".into(), serde_json::json!(total));
    }
    obj.insert(
        "derivedFrom".into(),
        serde_json::json!({
            "vindexSha256": source_sha,
            "shortId": source_short,
            "how": "larql trim — per-layer embedding vocabulary rows dropped",
            "keep_set_sha256": keep_sha,
            "walk_index_dropped": opts.drop_walk_index,
        }),
    );
    // The copy loop hard-linked this file, so it shares an inode with the
    // SOURCE manifest — writing through it would rewrite the identity of
    // the vindex we were cut from. Break the link first.
    let out = dst.join(MANIFEST_JSON);
    let _ = std::fs::remove_file(&out);
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&manifest).map_err(|e| VindexError::Parse(e.to_string()))?,
    )?;
    Ok(Some((new_sha, new_short)))
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
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec(&serde_json::json!({
                "vindexSha256": "a".repeat(64),
                "shortId": "aaaaaaaa",
                "totalBytes": 999,
                "files": {
                    PLE_WEIGHTS_BIN: "stale",
                    DOWN_FEATURES_BIN: "stale",
                    TOKENIZER_JSON: "stale",
                },
            }))
            .unwrap(),
        )
        .unwrap();
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
                ..TrimOptions::bare()
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
                ..TrimOptions::bare()
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
                ..TrimOptions::bare()
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
                ..TrimOptions::bare()
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
    fn a_receipt_records_what_went_and_refuses_to_claim_the_model_still_works() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        write_fixture(&src, 8, 4);

        let report = trim_vindex(
            &src,
            &dst,
            &TrimOptions {
                keep: KeepSet::Ids([1u32, 5, 6].into_iter().collect()),
                drop_walk_index: true,
                write_receipt: true,
                ..TrimOptions::bare()
            },
        )
        .unwrap();

        let receipt: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dst.join(PRUNING_RECEIPT_JSON)).unwrap())
                .unwrap();
        assert_eq!(receipt["receipt_version"], PRUNING_RECEIPT_VERSION);
        assert_eq!(receipt["removed"]["vocab_rows_kept"], serde_json::json!(3));
        assert_eq!(
            receipt["removed"]["vocab_rows_dropped"],
            serde_json::json!(5)
        );
        assert_eq!(
            receipt["removed"]["walk_index_dropped"],
            serde_json::json!(true)
        );
        assert_eq!(receipt["keep_set"]["total_ids"], serde_json::json!(3));
        assert!(receipt["keep_set"]["sha256"].as_str().unwrap().len() == 64);

        // The claim this receipt must never make on its own.
        assert_eq!(
            receipt["behavior"]["status"],
            serde_json::json!("not_verified")
        );
        assert!(report.coverage_passed.is_some());
    }

    #[test]
    fn attached_probe_results_are_marked_as_the_callers_not_the_compilers() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        write_fixture(&src, 8, 4);

        trim_vindex(
            &src,
            &dst,
            &TrimOptions {
                keep: KeepSet::TopN(4),
                write_receipt: true,
                probe_results: Some(serde_json::json!({"France": "Paris"})),
                ..TrimOptions::bare()
            },
        )
        .unwrap();

        let receipt: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dst.join(PRUNING_RECEIPT_JSON)).unwrap())
                .unwrap();
        assert_eq!(receipt["behavior"]["status"], serde_json::json!("attached"));
        assert_eq!(
            receipt["behavior"]["results"]["France"],
            serde_json::json!("Paris")
        );
        assert!(receipt["behavior"]["note"]
            .as_str()
            .unwrap()
            .contains("did not"));
    }

    #[test]
    fn a_vindex_without_a_tokenizer_fails_coverage_rather_than_passing_it() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        write_fixture(&src, 8, 4);
        // The fixture's tokenizer.json is "{}" — not loadable as a tokenizer.
        let report = trim_vindex(
            &src,
            &dst,
            &TrimOptions {
                keep: KeepSet::TopN(4),
                write_receipt: true,
                ..TrimOptions::bare()
            },
        )
        .unwrap();
        assert_eq!(report.coverage_passed, Some(false));
        let receipt: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dst.join(PRUNING_RECEIPT_JSON)).unwrap())
                .unwrap();
        assert_eq!(receipt["coverage"]["passed"], serde_json::json!(false));
    }

    #[test]
    fn no_receipt_means_coverage_is_unknown_not_passed() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        write_fixture(&src, 8, 4);
        let report = trim_vindex(
            &src,
            &dst,
            &TrimOptions {
                keep: KeepSet::TopN(4),
                ..TrimOptions::bare()
            },
        )
        .unwrap();
        assert_eq!(report.coverage_passed, None);
        assert!(!dst.join(PRUNING_RECEIPT_JSON).exists());
    }

    /// Divinci gates patch compatibility on the manifest's vindex id. A
    /// trim that inherited it would ship a slice claiming to BE the
    /// vindex it was cut from, so a patch addressing rows the slice no
    /// longer has would read as compatible and be applied silently.
    #[test]
    fn a_trimmed_artifact_gets_its_own_identity_and_fresh_checksums() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        write_fixture(&src, 8, 4);

        let report = trim_vindex(
            &src,
            &dst,
            &TrimOptions {
                keep: KeepSet::Ids([1u32, 5, 6].into_iter().collect()),
                drop_walk_index: true,
                ..TrimOptions::bare()
            },
        )
        .unwrap();

        let m: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dst.join("manifest.json")).unwrap()).unwrap();
        let new_sha = m["vindexSha256"].as_str().unwrap();
        assert_ne!(new_sha, "a".repeat(64), "identity must not be inherited");
        assert_eq!(report.new_vindex_sha256.as_deref(), Some(new_sha));
        assert_eq!(m["shortId"].as_str().unwrap(), &new_sha[..8]);
        assert_eq!(m["derivedFrom"]["shortId"], serde_json::json!("aaaaaaaa"));
        assert_eq!(
            m["derivedFrom"]["walk_index_dropped"],
            serde_json::json!(true)
        );

        // Checksums are re-derived, and a dropped file leaves the map.
        let files = m["files"].as_object().unwrap();
        assert!(
            !files.contains_key(DOWN_FEATURES_BIN),
            "dropped file still listed"
        );
        assert_ne!(files[PLE_WEIGHTS_BIN], serde_json::json!("stale"));
        assert_ne!(files[TOKENIZER_JSON], serde_json::json!("stale"));
    }

    /// The id is derived, not random: the receipt promises that rerunning
    /// a trim reproduces the artifact, and that has to cover the manifest.
    #[test]
    fn the_same_inputs_produce_the_same_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        write_fixture(&src, 8, 4);
        let run = |dst: &Path| {
            trim_vindex(
                src.as_path(),
                dst,
                &TrimOptions {
                    keep: KeepSet::TopN(4),
                    ..TrimOptions::bare()
                },
            )
            .unwrap()
            .new_vindex_sha256
        };
        let first = run(&tmp.path().join("a"));
        let second = run(&tmp.path().join("b"));
        assert_eq!(first, second);
        // The source manifest must be untouched: the copy loop hard-links
        // it, so a careless write would rewrite the identity of the vindex
        // being trimmed FROM.
        let src_manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(src.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(
            src_manifest["vindexSha256"],
            serde_json::json!("a".repeat(64))
        );
        assert!(src_manifest.get("derivedFrom").is_none());
    }

    /// ...and a different keep-set is a different artifact.
    #[test]
    fn a_different_keep_set_is_a_different_artifact() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        write_fixture(&src, 8, 4);
        let run = |dst: &Path, keep: KeepSet| {
            trim_vindex(
                src.as_path(),
                dst,
                &TrimOptions {
                    keep,
                    ..TrimOptions::bare()
                },
            )
            .unwrap()
            .new_vindex_sha256
        };
        assert_ne!(
            run(&tmp.path().join("a"), KeepSet::TopN(4)),
            run(&tmp.path().join("b"), KeepSet::TopN(5))
        );
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
                ..TrimOptions::bare()
            },
        )
        .unwrap();
        assert_eq!(report.vocab_kept, 1);
    }
}
