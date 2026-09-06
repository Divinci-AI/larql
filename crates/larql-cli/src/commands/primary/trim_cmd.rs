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
use larql_vindex::format::trim::{keep_set_from_file, trim_vindex, KeepSet, TrimOptions};

use crate::commands::primary::cache;

#[derive(Args)]
pub struct TrimArgs {
    /// Source vindex: directory, `hf://owner/name`, `owner/name`, or cache shorthand.
    pub source: String,

    /// Destination directory. Must not already exist.
    #[arg(short = 'o', long)]
    pub output: PathBuf,

    /// File of token ids to keep, whitespace- or newline-separated.
    /// Mutually exclusive with `--keep-topn`.
    #[arg(long, conflicts_with = "keep_topn")]
    pub keep_file: Option<PathBuf>,

    /// Keep token ids below N. A frequency proxy, not a domain
    /// vocabulary — prefer `--keep-file` for a real focused slice.
    #[arg(long, conflicts_with = "keep_file")]
    pub keep_topn: Option<usize>,

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

    let keep = match (&args.keep_file, args.keep_topn) {
        (Some(path), _) => keep_set_from_file(path)?,
        (None, Some(n)) => KeepSet::TopN(n),
        (None, None) => {
            return Err(
                "pass --keep-file <ids.txt> or --keep-topn <N> to say which \
                 vocabulary rows to keep"
                    .into(),
            )
        }
    };

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
