//! `larql memit` — batch fact editing via joint covariance-MEMIT.
//!
//! Reads a JSON list of edits, optionally auto-discovers each edit's crown
//! layer (Phase A), groups edits by layer, and invokes the covariance-based
//! MEMIT solver already shipping in `larql_inference::forward::memit::run_memit`.
//! Writes one dense `.lqpatch` per affected layer.
//!
//! Phase C of RFC-0001. The joint least-squares MEMIT in run_memit implements
//! the closed-form from Meng et al. 2022–2023 with a null-space covariance
//! projection that keeps specificity high — complementary to the Python
//! simple-MEMIT variant validated in CHAPTER_21_STACK.md / CHAPTER_22_DISTRIBUTED_STACK.md.

use std::fs::{self};
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::{
    edit::{compute_dense, write_patch, PatchProvenance},
    forward::memit::{run_memit, MemitFact},
    forward::predict_with_ffn,
    InferenceModel, LastPositionAblatingFfn, WeightFfn,
};
use serde::{Deserialize, Serialize};

#[derive(Args)]
pub struct MemitArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// JSON file listing edits to apply. Format:
    ///   [
    ///     {"label":"france-to-tokyo","src":"Capital of France? A:",
    ///      "new_token":" Tokyo","layer":27},
    ///     ...
    ///   ]
    /// If "layer" is omitted, crown discovery runs for that edit.
    #[arg(short, long)]
    edits: PathBuf,

    /// Output directory for per-layer patch files.
    #[arg(short, long)]
    output: PathBuf,

    /// Ridge regularisation for the MEMIT matrix solve.
    #[arg(long, default_value = "0.01")]
    ridge: f64,

    /// Target-direction alpha: how hard to push toward the new-token's
    /// embedding. Chapter 21 used 2× for France→Tokyo; a small value here
    /// works well for well-conditioned edits.
    #[arg(long, default_value = "1.0")]
    target_alpha: f32,

    /// Predict top-k window used by the crown scan.
    #[arg(long, default_value = "100")]
    top_k: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EditSpec {
    /// Human-readable label — used in the patch filename.
    label: String,
    /// Source prompt the model currently answers incorrectly.
    src: String,
    /// Token string the edit should make the model produce.
    new_token: String,
    /// Optional explicit crown layer; auto-discovered when omitted.
    #[serde(default)]
    layer: Option<usize>,
}

pub fn run(args: MemitArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let t0 = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    eprintln!(
        "  {} layers ({:.1}s)",
        model.num_layers(),
        t0.elapsed().as_secs_f64()
    );

    let edits_json = fs::read_to_string(&args.edits)
        .map_err(|e| format!("failed to read {}: {e}", args.edits.display()))?;
    let edits: Vec<EditSpec> = serde_json::from_str(&edits_json)
        .map_err(|e| format!("edits.json parse: {e}"))?;
    eprintln!("Loaded {} edit specs", edits.len());

    // Build MemitFacts. Each needs prompt_tokens, target_token_id, layer.
    let weights = model.weights();
    let mut facts: Vec<MemitFact> = Vec::with_capacity(edits.len());
    for edit in &edits {
        let prompt_tokens = tokenize(&model, &edit.src)?;
        let target_tokens = model
            .tokenizer()
            .encode(edit.new_token.as_str(), false)
            .map_err(|e| format!("tokenize target '{}': {e}", edit.new_token))?
            .get_ids()
            .to_vec();
        let target_token_id = *target_tokens.first().ok_or_else(|| {
            format!("new_token '{}' tokenized to empty id list", edit.new_token)
        })?;

        let layer = match edit.layer {
            Some(l) => l,
            None => {
                eprintln!("  [{}] discovering crown layer...", edit.label);
                let l = scan_crown_layer(&model, &prompt_tokens, edit.new_token.trim(), args.top_k)?;
                eprintln!("  [{}] crown = L{l}", edit.label);
                l
            }
        };

        facts.push(MemitFact {
            prompt_tokens,
            target_token_id,
            layer,
            label: edit.label.clone(),
        });
    }

    // Invoke the existing covariance-MEMIT solver.
    eprintln!(
        "\nRunning covariance-MEMIT (ridge={}, target_alpha={})...",
        args.ridge, args.target_alpha
    );
    let memit_start = Instant::now();
    let results = run_memit(
        weights,
        &facts,
        args.ridge,
        args.target_alpha,
        model.tokenizer(),
    )
    .map_err(|e| format!("run_memit: {e}"))?;
    eprintln!(
        "  MEMIT solve: {:.1}s  ({} layer(s) updated)",
        memit_start.elapsed().as_secs_f64(),
        results.len()
    );

    // Prepare output dir.
    fs::create_dir_all(&args.output)?;

    // Serialise each layer's dense ΔW into a `.lqpatch` file.
    for result in &results {
        let delta = &result.delta_w;
        let provenance = PatchProvenance {
            src_prompt: String::new(),
            tgt_prompt: String::new(),
            old_token: String::new(),
            new_token: format!(
                "MEMIT batch ({} facts @ L{})",
                result.fact_results.len(),
                result.layer
            ),
            crown_delta: 0.0,
            created_at: now_iso(),
        };
        let patch = compute_dense(delta, result.layer, provenance);
        let path = args
            .output
            .join(format!("memit_L{}.lqpatch", result.layer));
        write_patch(&path, &patch)?;
        let mb = (patch.delta_w.len() * 4) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "  wrote {}  ({:.1} MB, {} facts at this layer)",
            path.display(),
            mb,
            result.fact_results.len()
        );
    }

    // Manifest.
    let manifest = serde_json::json!({
        "model": args.model,
        "edits_file": args.edits.display().to_string(),
        "patches": results.iter().map(|r| {
            format!("memit_L{}.lqpatch", r.layer)
        }).collect::<Vec<_>>(),
        "ridge": args.ridge,
        "target_alpha": args.target_alpha,
        "num_edits": edits.len(),
        "num_layers": results.len(),
    });
    let manifest_path = args.output.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    eprintln!("  wrote {}", manifest_path.display());

    eprintln!("\nDone. Apply with:");
    eprintln!(
        "  larql apply-patch {} -p {}/memit_L*.lqpatch",
        args.model,
        args.output.display()
    );
    Ok(())
}

fn tokenize(
    model: &InferenceModel,
    text: &str,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let encoding = model
        .tokenizer()
        .encode(text, true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    Ok(encoding.get_ids().to_vec())
}

fn scan_crown_layer(
    model: &InferenceModel,
    tokens: &[u32],
    expect: &str,
    top_k: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let weights = model.weights();
    let num_layers = model.num_layers();
    let start_layer = (num_layers * 3) / 5;
    let end_layer = num_layers.saturating_sub(2);
    let weight_ffn = WeightFfn { weights };
    let baseline = larql_inference::forward::predict(weights, model.tokenizer(), tokens, top_k);
    let baseline_expect = prob_of(&baseline.predictions, expect);
    let mut best: Option<(usize, f64)> = None;
    let mut best_flipped: Option<(usize, f64)> = None;
    for layer in start_layer..=end_layer {
        let ffn = LastPositionAblatingFfn::new(&weight_ffn, layer);
        let r = predict_with_ffn(weights, model.tokenizer(), tokens, top_k, &ffn);
        let top = r
            .predictions
            .first()
            .map(|(t, _)| t.trim().to_string())
            .unwrap_or_default();
        let expect_prob = prob_of(&r.predictions, expect);
        let delta = expect_prob - baseline_expect;
        let flipped = !top.eq_ignore_ascii_case(expect);
        if flipped && best_flipped.map_or(true, |(_, d)| delta < d) {
            best_flipped = Some((layer, delta));
        }
        if best.map_or(true, |(_, d)| delta < d) {
            best = Some((layer, delta));
        }
    }
    Ok(best_flipped
        .map(|(l, _)| l)
        .or(best.map(|(l, _)| l))
        .unwrap_or(start_layer))
}

fn prob_of(predictions: &[(String, f64)], target: &str) -> f64 {
    for (tok, prob) in predictions {
        if tok.trim().eq_ignore_ascii_case(target.trim()) {
            return *prob;
        }
    }
    0.0
}

fn now_iso() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch-{now}")
}
