//! `larql trim <SRC> -o <DST>` — write a vocabulary-trimmed vindex.
//!
//! Where `larql slice` carves by tensor ROLE and keeps every vocabulary
//! row, `trim` keeps every role and drops the per-layer-embedding rows a
//! domain never uses. On Gemma 4 e2b that table is 4.70 GB, 42% of the
//! vindex — the largest single lever there is.
//!
//! The keep-set is either a file of token ids (the honest form: tokenize
//! your domain corpus and pass the ids it uses) or `--keep-topn N`, a
//! frequency proxy useful for staircase studies. `--drop-walk-index`
//! additionally removes `down_features.bin`, which is answer-preserving
//! but pushes the FFN planner to its dense last-resort rung.
//!
//! Unchanged files are hard-linked when the filesystem allows it, so a
//! trimmed vindex beside its source costs only the bytes it rewrites.

use std::path::PathBuf;

use clap::Args;
use larql_vindex::format::trim::{
    keep_set_from_corpus, keep_set_from_file, trim_vindex, KeepSet, TrimOptions,
};

use crate::commands::primary::cache;

#[derive(Args)]
pub struct TrimArgs {
    /// Source vindex: directory, `hf://owner/name`, `owner/name`, or cache shorthand.
    pub source: String,

    /// Destination directory. Must not already exist.
    #[arg(short = 'o', long)]
    pub output: PathBuf,

    /// Domain corpus, one prompt per line. Tokenized with the vindex's
    /// OWN tokenizer, and the rows those ids name are what the slice
    /// keeps — the honest way to say "which rows does this domain use".
    #[arg(long, conflicts_with_all = ["keep_file", "keep_topn"])]
    pub keep_corpus: Option<PathBuf>,

    /// File of token ids to keep, whitespace- or newline-separated.
    #[arg(long, conflicts_with_all = ["keep_topn", "keep_corpus"])]
    pub keep_file: Option<PathBuf>,

    /// Keep token ids below N. A frequency proxy, not a domain
    /// vocabulary — prefer `--keep-corpus` for a real focused slice.
    #[arg(long, conflicts_with_all = ["keep_file", "keep_corpus"])]
    pub keep_topn: Option<usize>,

    /// With `--keep-corpus`: also keep every single-character piece, so
    /// unseen text degrades gently instead of losing its rows entirely.
    /// Costs a larger table.
    #[arg(long)]
    pub keep_byte_fallback: bool,

    /// With `--keep-corpus`: drop special/added tokens the corpus never
    /// used. Off by default because a chat model whose `<|turn>` rows are
    /// gone is broken, not focused.
    #[arg(long)]
    pub drop_unused_specials: bool,

    /// Write the resolved keep-set to this file, one token id per line,
    /// so it can be inspected or reused with `--keep-file`.
    #[arg(long)]
    pub emit_keep_set: Option<PathBuf>,

    /// Also drop the f32 walk index (`down_features.bin` + `down_meta.bin`).
    #[arg(long)]
    pub drop_walk_index: bool,
}

pub fn run(args: TrimArgs) -> Result<(), Box<dyn std::error::Error>> {
    let src = cache::resolve_model(&args.source)?;

    if let Ok(found) = larql_vindex::format::generation::detect_generation(&src) {
        if found != larql_vindex::format::generation::ContainerGeneration::V2 {
            return Err(larql_vindex::format::generation::unsupported_generation(
                "trim", &src, found,
            )
            .into());
        }
    }

    let keep = if let Some(corpus_path) = &args.keep_corpus {
        let corpus = std::fs::read_to_string(corpus_path)?;
        let tokenizer = larql_vindex::format::load::load_vindex_tokenizer(&src)?;
        let (keep, origin) = keep_set_from_corpus(
            &tokenizer,
            &corpus,
            !args.drop_unused_specials,
            args.keep_byte_fallback,
        )?;
        println!(
            "Keep-set from {} ({} lines, tokenized with the vindex's own tokenizer):",
            corpus_path.display(),
            origin.corpus_lines
        );
        println!("  {} ids emitted by the corpus", origin.from_corpus);
        if origin.specials > 0 {
            println!("  {} special/added tokens folded in", origin.specials);
        }
        if origin.byte_fallback > 0 {
            println!(
                "  {} single-character fallback pieces",
                origin.byte_fallback
            );
        }
        println!("  {} ids total", origin.total);
        keep
    } else if let Some(path) = &args.keep_file {
        keep_set_from_file(path)?
    } else if let Some(n) = args.keep_topn {
        KeepSet::TopN(n)
    } else {
        return Err(
            "pass --keep-corpus <corpus.txt> (preferred), --keep-file <ids.txt>, \
             or --keep-topn <N> to say which vocabulary rows to keep"
                .into(),
        );
    };

    if let Some(path) = &args.emit_keep_set {
        if let KeepSet::Ids(ids) = &keep {
            let mut out = String::new();
            for id in ids {
                out.push_str(&id.to_string());
                out.push('\n');
            }
            std::fs::write(path, out)?;
            println!("  keep-set written to {}", path.display());
        }
    }

    let report = trim_vindex(
        &src,
        &args.output,
        &TrimOptions {
            keep,
            drop_walk_index: args.drop_walk_index,
        },
    )?;

    let gb = |b: u64| b as f64 / 1e9;
    println!("Trimmed vindex: {}", args.output.display());
    println!(
        "  vocabulary rows: {} of {} kept ({:.1}%)",
        report.vocab_kept,
        report.vocab_total,
        100.0 * report.vocab_kept as f64 / report.vocab_total.max(1) as f64
    );
    println!(
        "  per-layer embeddings: {:.2} GB -> {:.2} GB",
        gb(report.ple_bytes_before),
        gb(report.ple_bytes_after)
    );
    if report.walk_index_dropped {
        println!(
            "  walk index dropped: {:.2} GB",
            gb(report.walk_index_bytes)
        );
    }
    println!("  total saved: {:.2} GB", gb(report.bytes_saved()));
    println!(
        "  unchanged files: {} hard-linked, {} copied",
        report.linked_files, report.copied_files
    );
    Ok(())
}
