//! `larql edit` — single-fact rank-1 editor.
//!
//! Given a source prompt the model currently answers one way, and a target
//! prompt showing the desired behaviour, computes a rank-1 ΔW on the crown
//! layer's down_proj and writes it as a portable patch file (see
//! `larql_inference::edit::EditPatch`).
//!
//! Implements Phase B of RFC-0001 using the Phase A `larql crown` for
//! automatic crown-layer discovery and a linear scale search (Chapter 18
//! Phase 130 — the simpler variant; a binary search can replace this later).

use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::{
    edit::{compute_rank1, write_patch, PatchProvenance},
    forward::{capture_ffn_activation_matrix, predict_with_ffn},
    InferenceModel, LastPositionAblatingFfn, LastPositionInjectingFfn, WeightFfn,
};
use larql_inference::ndarray::Array1;

#[derive(Args)]
pub struct EditArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Source prompt — the model's current (to-be-overwritten) answer prompt.
    #[arg(long)]
    src: String,

    /// Target prompt — a prompt where the model already produces the desired answer.
    /// The edit transports the relation Source→Source_answer into Source→Target_answer
    /// by capturing how the crown layer behaves on the target and imprinting that
    /// direction conditional on the source's key.
    #[arg(long)]
    tgt: String,

    /// The token string we want the SOURCE prompt to produce after the edit.
    /// Must be reachable within `--top-k` predictions during the scale search.
    #[arg(long)]
    new_token: String,

    /// Explicit crown layer. If omitted, runs ablation scan (same as `larql crown`)
    /// to discover the source prompt's load-bearing MLP.
    #[arg(long)]
    layer: Option<usize>,

    /// Scale grid for the linear search. Default: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0.
    #[arg(long, value_delimiter = ',')]
    scales: Option<Vec<f32>>,

    /// Predict top-k window used by the scale search to detect the new-token flip.
    #[arg(long, default_value = "100")]
    top_k: usize,

    /// Output patch file path (binary .lqpatch).
    #[arg(short, long)]
    output: PathBuf,

    /// Skip the scale search and use this exact scale. Useful for batch pipelines.
    #[arg(long)]
    fixed_scale: Option<f32>,

    /// Optional label recorded in patch provenance (e.g., "France-to-Tokyo").
    #[arg(long)]
    label: Option<String>,
}

pub fn run(args: EditArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let t0 = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let num_layers = model.num_layers();
    eprintln!(
        "  {num_layers} layers, hidden={}, intermediate={} ({:.1}s)",
        model.hidden_size(),
        model.weights().intermediate_size,
        t0.elapsed().as_secs_f64()
    );

    let weights = model.weights();
    let hidden = weights.hidden_size;
    let intermediate = weights.intermediate_size;

    let src_tokens = tokenize(&model, &args.src)?;
    let tgt_tokens = tokenize(&model, &args.tgt)?;
    eprintln!(
        "Source ({} tokens): {:?}\nTarget ({} tokens): {:?}",
        src_tokens.len(),
        args.src,
        tgt_tokens.len(),
        args.tgt
    );

    // 1. Determine crown layer.
    let layer = match args.layer {
        Some(l) => {
            eprintln!("Using explicit crown layer: L{l}");
            l
        }
        None => {
            eprintln!("Discovering crown layer via ablation scan...");
            let crown = scan_crown_layer(&model, &src_tokens, &args.new_token, args.top_k)?;
            eprintln!("  Crown layer discovered: L{crown}");
            crown
        }
    };

    // 2. Capture k_src and k_tgt at crown layer.
    eprintln!("\nCapturing FFN intermediate activations at L{layer}...");
    let act_src = capture_ffn_activation_matrix(weights, &src_tokens, layer)
        .ok_or_else(|| format!("failed to capture activations for src prompt at L{layer}"))?;
    let act_tgt = capture_ffn_activation_matrix(weights, &tgt_tokens, layer)
        .ok_or_else(|| format!("failed to capture activations for tgt prompt at L{layer}"))?;

    let k_src_row = act_src.row(act_src.shape()[0] - 1).to_owned();
    let k_tgt_row = act_tgt.row(act_tgt.shape()[0] - 1).to_owned();
    if k_src_row.len() != intermediate || k_tgt_row.len() != intermediate {
        return Err(format!(
            "intermediate size mismatch: got {}/{}, expected {intermediate}",
            k_src_row.len(),
            k_tgt_row.len()
        )
        .into());
    }

    // 3. Compute d_base = W_down @ (k_tgt - k_src).
    //    W_down is stored under arch.ffn_down_key(layer); may be stored as
    //    [hidden, intermediate] or [intermediate, hidden]. Handle both.
    let w_down_key = weights.arch.ffn_down_key(layer);
    let w_down = weights
        .tensors
        .get(&w_down_key)
        .ok_or_else(|| format!("W_down missing at {w_down_key}"))?;
    let k_diff: Array1<f32> = &k_tgt_row - &k_src_row;

    let w_view = w_down.view();
    let d_base: Array1<f32> = if w_down.shape() == [hidden, intermediate] {
        // canonical: out = W @ k → shape (hidden,)
        w_view.dot(&k_diff)
    } else if w_down.shape() == [intermediate, hidden] {
        // transposed: out = k^T @ W → shape (hidden,)
        k_diff.view().dot(&w_view)
    } else {
        return Err(format!(
            "unexpected W_down shape {:?} at layer {layer}",
            w_down.shape()
        )
        .into());
    };
    eprintln!(
        "  ||k_src|| = {:.2},  ||k_tgt|| = {:.2},  ||d_base|| = {:.2}",
        k_src_row.iter().map(|v| v * v).sum::<f32>().sqrt(),
        k_tgt_row.iter().map(|v| v * v).sum::<f32>().sqrt(),
        d_base.iter().map(|v| v * v).sum::<f32>().sqrt()
    );

    // 4. Scale search — find minimum scale that flips top-1 of source prompt to new_token.
    let d_base_vec = d_base.to_vec();
    let new_token_norm = args.new_token.trim();
    let weight_ffn = WeightFfn { weights };

    let chosen_scale = if let Some(s) = args.fixed_scale {
        eprintln!("\nUsing fixed scale = {s}");
        s
    } else {
        let scale_grid = args
            .scales
            .unwrap_or_else(|| vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]);
        eprintln!("\nLinear scale search (grid: {:?}):", scale_grid);
        let mut chosen: Option<f32> = None;
        for &s in &scale_grid {
            let scaled: Vec<f32> = d_base_vec.iter().map(|&v| v * s).collect();
            let ffn = LastPositionInjectingFfn::new(&weight_ffn, layer, scaled);
            let result = predict_with_ffn(weights, model.tokenizer(), &src_tokens, 5, &ffn);
            let top = result
                .predictions
                .first()
                .map(|(t, _)| t.trim().to_string())
                .unwrap_or_default();
            eprintln!("  scale={s:>4}  top = {top}");
            if top.eq_ignore_ascii_case(new_token_norm) {
                chosen = Some(s);
                break;
            }
        }
        chosen.ok_or("scale search exhausted without flipping to new_token — try a larger --scales range")?
    };
    eprintln!("  → chosen scale: {chosen_scale}");

    // 5. Construct + write patch.
    let provenance = PatchProvenance {
        src_prompt: args.src.clone(),
        tgt_prompt: args.tgt.clone(),
        old_token: String::new(), // not needed — captured by src
        new_token: args.new_token.clone(),
        crown_delta: 0.0,
        created_at: now_iso(),
    };

    // Note: we record d_base (unscaled) and bake the scale into d below
    // so apply_patch can be reconstructed without knowing d_base.
    let patch = compute_rank1(
        &k_src_row.to_vec(),
        &d_base_vec,
        chosen_scale,
        layer,
        provenance,
    );

    write_patch(&args.output, &patch)?;
    let meta_rel = (patch.d.iter().map(|v| v * v).sum::<f32>().sqrt())
        / (k_src_row.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-9);
    eprintln!(
        "\nWrote patch: {}  (layer=L{}, scale={}, Δ-rel~{:.4})",
        args.output.display(),
        patch.layer,
        patch.scale,
        meta_rel
    );
    if let Some(lbl) = args.label {
        eprintln!("  label: {lbl}");
    }
    Ok(())
}

fn tokenize(model: &InferenceModel, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
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
    let mut best: Option<(usize, f64, String)> = None;
    let mut best_flipped: Option<(usize, f64)> = None;
    for layer in start_layer..=end_layer {
        let ffn = LastPositionAblatingFfn::new(&weight_ffn, layer);
        let r = predict_with_ffn(weights, model.tokenizer(), tokens, top_k, &ffn);
        let top = r.predictions.first().map(|(t, _)| t.trim().to_string()).unwrap_or_default();
        let expect_prob = prob_of(&r.predictions, expect);
        let delta = expect_prob - baseline_expect;
        let flipped = !top.eq_ignore_ascii_case(expect.trim());
        eprintln!(
            "  L{layer:>3}  top={top:<12} Δprob={:+.4}{}",
            delta,
            if flipped { "  ← flipped" } else { "" }
        );
        if flipped {
            if best_flipped.map_or(true, |(_, d)| delta < d) {
                best_flipped = Some((layer, delta));
            }
        }
        if best.as_ref().map_or(true, |(_, d, _)| delta < *d) {
            best = Some((layer, delta, top));
        }
    }
    Ok(best_flipped.map(|(l, _)| l).or(best.map(|(l, _, _)| l)).unwrap_or(start_layer))
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
    // Simple timestamp — avoid chrono dep for a single ISO string.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch-{now}")
}
