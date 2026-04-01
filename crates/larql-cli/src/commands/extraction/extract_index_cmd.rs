use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use larql_vindex::IndexBuildCallbacks;
use larql_vindex::write_model_weights;
use larql_inference::{ InferenceModel};

#[derive(Args)]
pub struct ExtractIndexArgs {
    /// Model path or HuggingFace model ID (extracts directly from weights).
    /// Not needed if --from-vectors is used.
    model: Option<String>,

    /// Output path for the .vindex directory.
    #[arg(short, long)]
    output: PathBuf,

    /// Build from already-extracted NDJSON vector files instead of model weights.
    /// Point to the directory containing ffn_gate.vectors.jsonl, etc.
    #[arg(long)]
    from_vectors: Option<PathBuf>,

    /// Top-K tokens to store per feature in down metadata (only for model extraction).
    #[arg(long, default_value = "10")]
    down_top_k: usize,

    /// Extract level: browse (gate+embed+down_meta), inference (+attention+norms),
    /// all (+up+down+lm_head for COMPILE).
    #[arg(long, default_value = "browse", value_parser = parse_extract_level)]
    level: larql_vindex::ExtractLevel,

    /// Include full model weights. Alias for --level all (deprecated, use --level instead).
    #[arg(long)]
    include_weights: bool,

    /// Store weights in f16 (half precision). Halves file sizes with negligible accuracy loss.
    #[arg(long)]
    f16: bool,

    /// Skip stages that already have output files (resume interrupted builds).
    #[arg(long)]
    resume: bool,
}

fn parse_extract_level(s: &str) -> Result<larql_vindex::ExtractLevel, String> {
    match s.to_lowercase().as_str() {
        "browse" => Ok(larql_vindex::ExtractLevel::Browse),
        "inference" => Ok(larql_vindex::ExtractLevel::Inference),
        "all" => Ok(larql_vindex::ExtractLevel::All),
        _ => Err(format!("unknown extract level: {s} (expected: browse, inference, all)")),
    }
}

struct CliBuildCallbacks {
    stage_start: Option<Instant>,
    feature_bar: ProgressBar,
}

impl CliBuildCallbacks {
    fn new() -> Self {
        let feature_bar = ProgressBar::new(0);
        feature_bar.set_style(
            ProgressStyle::default_bar()
                .template("  {spinner} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        feature_bar.set_draw_target(indicatif::ProgressDrawTarget::stderr());

        Self {
            stage_start: None,
            feature_bar,
        }
    }
}

impl IndexBuildCallbacks for CliBuildCallbacks {
    fn on_stage(&mut self, stage: &str) {
        self.feature_bar.finish_and_clear();
        eprintln!("\n── {stage} ──");
        self.stage_start = Some(Instant::now());
    }

    fn on_layer_start(&mut self, component: &str, layer: usize, total: usize) {
        self.feature_bar.reset();
        self.feature_bar
            .set_message(format!("{component} L{layer} ({}/{})", layer + 1, total));
    }

    fn on_feature_progress(
        &mut self,
        component: &str,
        _layer: usize,
        done: usize,
        total: usize,
    ) {
        if total > 0 {
            self.feature_bar.set_length(total as u64);
        }
        self.feature_bar.set_position(done as u64);
        if total == 0 {
            self.feature_bar
                .set_message(format!("{component} {done} records"));
        }
    }

    fn on_layer_done(&mut self, component: &str, layer: usize, elapsed_ms: f64) {
        self.feature_bar.finish_and_clear();
        eprintln!("  {component} L{layer:2}: {:.1}s", elapsed_ms / 1000.0);
    }

    fn on_stage_done(&mut self, stage: &str, _elapsed_ms: f64) {
        self.feature_bar.finish_and_clear();
        if let Some(start) = self.stage_start.take() {
            eprintln!("  {stage}: {:.1}s", start.elapsed().as_secs_f64());
        }
    }
}

pub fn run(args: ExtractIndexArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut callbacks = CliBuildCallbacks::new();
    let build_start = Instant::now();

    // Resolve extract level: --include-weights upgrades to All (backwards compat)
    let level = if args.include_weights {
        larql_vindex::ExtractLevel::All
    } else {
        args.level
    };

    let dtype = if args.f16 {
        larql_vindex::StorageDtype::F16
    } else {
        larql_vindex::StorageDtype::F32
    };

    if let Some(ref vectors_dir) = args.from_vectors {
        // Build from existing NDJSON files
        eprintln!("Building vindex from vectors: {}", vectors_dir.display());
        eprintln!("Output: {}", args.output.display());

        larql_vindex::build_vindex_from_vectors(vectors_dir, &args.output, &mut callbacks)?;

        if matches!(level, larql_vindex::ExtractLevel::Inference | larql_vindex::ExtractLevel::All) {
            let model_name = args.model.as_deref().ok_or(
                "--model required with --level inference/all (need model to extract weights)",
            )?;
            eprintln!("\nLoading model for weights: {}", model_name);
            let model = InferenceModel::load(model_name)?;
            write_model_weights(model.weights(), &args.output, &mut callbacks)?;
        }
    } else {
        // Build from model weights
        let model_name = args
            .model
            .as_deref()
            .ok_or("Either provide a model name or use --from-vectors")?;

        eprintln!("Loading model: {}", model_name);
        let start = Instant::now();
        let model = InferenceModel::load(model_name)?;
        eprintln!(
            "  {} layers, hidden_size={}, intermediate_size={}, vocab_size={} ({:.1}s)",
            model.num_layers(),
            model.hidden_size(),
            model.weights().intermediate_size,
            model.weights().vocab_size,
            start.elapsed().as_secs_f64()
        );

        let level_str = match level {
            larql_vindex::ExtractLevel::Browse => "browse",
            larql_vindex::ExtractLevel::Inference => "inference",
            larql_vindex::ExtractLevel::All => "all",
        };
        let dtype_str = match dtype {
            larql_vindex::StorageDtype::F32 => "f32",
            larql_vindex::StorageDtype::F16 => "f16",
        };
        eprintln!("\nBuilding index: {} (level={}, dtype={})", args.output.display(), level_str, dtype_str);

        let output = &args.output;

        if args.resume {
            let has_gate = output.join("gate_vectors.bin").exists();
            let has_embed = output.join("embeddings.bin").exists();
            let has_down = output.join("down_meta.bin").exists()
                || (output.join("down_meta.jsonl").exists()
                    && std::fs::metadata(output.join("down_meta.jsonl"))
                        .map(|m| m.len() > 1000)
                        .unwrap_or(false));

            if has_gate && has_embed && has_down {
                eprintln!("  Resuming: core files exist — skipping extraction");
                larql_vindex::build_vindex_resume(
                    model.weights(),
                    model.tokenizer(),
                    model_name,
                    output,
                    &mut callbacks,
                )?;
            } else {
                eprintln!("  Resume: missing core files — full rebuild");
                larql_vindex::build_vindex(
                    model.weights(), model.tokenizer(), model_name,
                    output, args.down_top_k, level, dtype, &mut callbacks,
                )?;
            }
        } else {
            larql_vindex::build_vindex(
                model.weights(), model.tokenizer(), model_name,
                output, args.down_top_k, level, dtype, &mut callbacks,
            )?;
        }
    }

    callbacks.feature_bar.finish_and_clear();
    let build_elapsed = build_start.elapsed();

    // Print summary
    eprintln!("\n── Summary ──");
    eprintln!("  Output: {}", args.output.display());

    if build_elapsed.as_secs() >= 60 {
        eprintln!(
            "  Build time: {:.1}min",
            build_elapsed.as_secs_f64() / 60.0
        );
    } else {
        eprintln!("  Build time: {:.1}s", build_elapsed.as_secs_f64());
    }

    for name in &[
        "index.json",
        "gate_vectors.bin",
        "embeddings.bin",
        "down_meta.jsonl",
        "down_meta.bin",
        "tokenizer.json",
        "attn_weights.bin",
        "up_weights.bin",
        "down_weights.bin",
        "norms.bin",
        "lm_head.bin",
        "weight_manifest.json",
    ] {
        let path = args.output.join(name);
        if let Ok(meta) = std::fs::metadata(&path) {
            let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
            if size_mb > 1024.0 {
                eprintln!("  {name}: {:.2} GB", size_mb / 1024.0);
            } else if size_mb > 0.1 {
                eprintln!("  {name}: {:.1} MB", size_mb);
            } else {
                let size_kb = meta.len() as f64 / 1024.0;
                eprintln!("  {name}: {:.1} KB", size_kb);
            }
        } else {
            eprintln!("  {name}: (not found)");
        }
    }

    // Total: sum all files in the directory
    let total_size: u64 = std::fs::read_dir(&args.output)
        .ok()
        .map(|entries| {
            entries.filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0);
    eprintln!(
        "  Total: {:.2} GB",
        total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    eprintln!("\nUsage:");
    eprintln!(
        "  larql walk --index {} -p \"The capital of France is\"",
        args.output.display()
    );

    Ok(())
}
