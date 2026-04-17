//! `larql crown` — discover the crown MLP layer for a fact-editing prompt.
//!
//! For each layer L in the configurable scan range, we run a forward pass
//! with that layer's FFN output zeroed at the last-token position (via
//! `LastPositionAblatingFfn`) and measure how much the expected token's
//! probability drops. The layer that most suppresses the expected token
//! (especially one where the top-1 prediction flips to something else) is
//! the crown — the load-bearing writer for that fact.
//!
//! This implements Phase 125c of the mechanistic-interpretability research
//! arc in Divinci-AI/server notebooks/CHAPTER_17_CORONATION.md. It is the
//! first of three commands proposed in RFC-0001 (crown, edit, memit).
//!
//! Example:
//!
//!   larql crown <model> \
//!       --prompt "Capital of France? A:" \
//!       --expect " Paris"
//!
//! Output (JSON with `--json`):
//!   { "crown_layer": 27, "delta_expect": -14.19,
//!     "top_after_ablation": "France",
//!     "scan": [{"layer": 23, "delta": -6.87, "top": "Paris", "expect_prob": ...}, ...] }

use std::time::Instant;

use clap::Args;
use larql_inference::{
    InferenceModel, LastPositionAblatingFfn, WeightFfn, predict, predict_with_ffn,
};

#[derive(Args)]
pub struct CrownArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Prompt text whose final token prediction we will audit.
    #[arg(short, long)]
    prompt: String,

    /// Expected next-token string (e.g., " Paris"). We measure how much
    /// each layer's ablation suppresses this token's logit / probability.
    #[arg(short, long)]
    expect: String,

    /// First layer to scan (inclusive). Default: 60% of model depth
    /// (entity zone typically starts around this depth per Chapter 15).
    #[arg(long)]
    start_layer: Option<usize>,

    /// Last layer to scan (inclusive). Default: `num_layers - 2`
    /// (final layer excluded — ablating it trivially breaks everything).
    #[arg(long)]
    end_layer: Option<usize>,

    /// How many top predictions to look up per forward pass. Larger =
    /// better chance of finding the expected token in the top-k window
    /// after ablation, but slower. Default 100.
    #[arg(short = 'k', long, default_value = "100")]
    top_k: usize,

    /// Emit machine-readable JSON to stdout (in addition to stderr diagnostics).
    #[arg(long)]
    json: bool,
}

#[derive(serde::Serialize)]
struct LayerResult {
    layer: usize,
    delta_expect_prob: f64,
    top_token: String,
    top_prob: f64,
    expect_prob: f64,
    flipped: bool,
}

#[derive(serde::Serialize)]
struct CrownReport {
    model: String,
    prompt: String,
    expect: String,
    baseline_top: String,
    baseline_expect_prob: f64,
    crown_layer: Option<usize>,
    crown_delta: Option<f64>,
    crown_top_after_ablation: Option<String>,
    scan: Vec<LayerResult>,
}

pub fn run(args: CrownArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let num_layers = model.num_layers();
    eprintln!(
        "  {num_layers} layers, hidden_size={} ({:.1}s)",
        model.hidden_size(),
        start.elapsed().as_secs_f64()
    );

    let start_layer = args.start_layer.unwrap_or((num_layers * 3) / 5);
    let end_layer = args.end_layer.unwrap_or(num_layers.saturating_sub(2));
    if start_layer > end_layer {
        return Err(format!(
            "start_layer ({start_layer}) must be <= end_layer ({end_layer})"
        )
        .into());
    }

    // Tokenize the prompt.
    let encoding = model
        .tokenizer()
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("Prompt: {:?}  ({} tokens)", args.prompt, token_ids.len());
    eprintln!("Expect: {:?}", args.expect);

    // Baseline forward pass.
    let weights = model.weights();
    eprintln!("Running baseline forward pass...");
    let base_start = Instant::now();
    let baseline = predict(weights, model.tokenizer(), &token_ids, args.top_k);
    eprintln!("  Baseline: {:.2}s", base_start.elapsed().as_secs_f64());

    let expect_norm = args.expect.trim();
    let (baseline_top, _baseline_top_prob) = baseline
        .predictions
        .first()
        .map(|(t, p)| (t.clone(), *p))
        .unwrap_or_else(|| ("?".to_string(), 0.0));
    let baseline_expect_prob = prob_of(&baseline.predictions, expect_norm);
    eprintln!(
        "  Baseline top: {:?}, expect prob: {:.4}",
        baseline_top, baseline_expect_prob
    );

    // Per-layer ablation scan.
    eprintln!(
        "\nScanning L{}..=L{} with last-position MLP ablation...",
        start_layer, end_layer
    );
    let weight_ffn = WeightFfn { weights };
    let mut scan = Vec::with_capacity(end_layer + 1 - start_layer);
    for layer in start_layer..=end_layer {
        let ffn = LastPositionAblatingFfn::new(&weight_ffn, layer);
        let t = Instant::now();
        let result =
            predict_with_ffn(weights, model.tokenizer(), &token_ids, args.top_k, &ffn);
        let elapsed = t.elapsed().as_secs_f64();
        let (top_token, top_prob) = result
            .predictions
            .first()
            .map(|(t, p)| (t.clone(), *p))
            .unwrap_or_else(|| ("?".to_string(), 0.0));
        let expect_prob = prob_of(&result.predictions, expect_norm);
        let flipped = !top_token.eq_ignore_ascii_case(expect_norm);

        eprintln!(
            "  L{layer:>3}  top={top_token:<12} Δprob={:+.4} top_prob={:.3}  ({elapsed:.1}s){}",
            expect_prob - baseline_expect_prob,
            top_prob,
            if flipped { "  ← flipped" } else { "" }
        );

        scan.push(LayerResult {
            layer,
            delta_expect_prob: expect_prob - baseline_expect_prob,
            top_token,
            top_prob,
            expect_prob,
            flipped,
        });
    }

    // Pick the crown: among layers where top flipped, the one with the
    // most-negative delta_expect_prob. If none flipped, the layer with the
    // largest suppression magnitude.
    let (crown_layer, crown_delta, crown_top) = {
        let pick = scan
            .iter()
            .filter(|r| r.flipped)
            .min_by(|a, b| a.delta_expect_prob.partial_cmp(&b.delta_expect_prob).unwrap())
            .or_else(|| {
                scan.iter().min_by(|a, b| {
                    a.delta_expect_prob.partial_cmp(&b.delta_expect_prob).unwrap()
                })
            });
        (
            pick.map(|c| c.layer),
            pick.map(|c| c.delta_expect_prob),
            pick.map(|c| c.top_token.clone()),
        )
    };

    eprintln!();
    if let (Some(layer), Some(delta), Some(top)) =
        (crown_layer, crown_delta, crown_top.as_ref())
    {
        eprintln!(
            "Crown layer: L{layer}  (Δexpect_prob = {delta:+.4}, top after = {top:?})"
        );
    } else {
        eprintln!("No crown layer found in scan range (all deltas were zero).");
    }

    let report = CrownReport {
        model: args.model.clone(),
        prompt: args.prompt.clone(),
        expect: args.expect.clone(),
        baseline_top,
        baseline_expect_prob,
        crown_layer,
        crown_delta,
        crown_top_after_ablation: crown_top,
        scan,
    };

    if args.json {
        let json = serde_json::to_string_pretty(&report)?;
        println!("{json}");
    }

    Ok(())
}

/// Return the probability of a token by exact-match (trim / case-insensitive).
fn prob_of(predictions: &[(String, f64)], target: &str) -> f64 {
    for (tok, prob) in predictions {
        if tok.trim().eq_ignore_ascii_case(target) {
            return *prob;
        }
    }
    0.0
}
