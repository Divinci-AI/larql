//! Single-edge compilation: one prompt + one answer → one compiled edge.
//!
//! Captures the residual at the target layer for the prompt, looks up the
//! answer token's embedding, installs an edge that fires only on this prompt
//! and pushes the answer token through the LM head. CLI-driven; contrasts
//! with patch mode (vindex-driven, many edges).

use larql_vindex::format::filenames::*;
use std::collections::HashMap;

use ndarray::ArcArray2;

use super::constraints::{
    collateral_provenance, fluency_provenance, fluency_ratio, judge, judge_baselines, judge_fluency,
    parse_control, BalancerOutcome, ControlBaseline, ControlResult, Fluency, Verdict,
};
use super::detect::detect_ffn_pattern;
use super::edge::install_edge;
use super::save::{copy_model_config, merge_for_save, write_safetensors};
use super::CompileArgs;

/// Top-1 continuation for a prompt, wrapped exactly as the trigger prompt was.
///
/// A control measured through a different chat wrap than the install is measuring a different
/// input, so the wrap is threaded through rather than re-decided here.
/// Total NLL of `text` under `weights`, over every predicted token. One forward pass: the
/// pre-norm residual at each position goes through `hidden_to_raw_logits`, the same final norm,
/// lm_head, scaling and softcap `predict` applies at the last position, then log-softmax.
fn text_fluency(
    weights: &larql_models::ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
) -> Result<Fluency, Box<dyn std::error::Error>> {
    let enc = tokenizer
        .encode(text, true)
        .map_err(|e| format!("tokenize fluency text: {}", e))?;
    ids_fluency(weights, enc.get_ids())
}

/// `text_fluency` on token ids. Split out so it is testable against `predict` without a tokenizer
/// round trip.
fn ids_fluency(
    weights: &larql_models::ModelWeights,
    ids: &[u32],
) -> Result<Fluency, Box<dyn std::error::Error>> {
    if ids.len() < 2 {
        return Err("the fluency text tokenizes to fewer than 2 tokens; there is nothing to score".into());
    }
    let rf = larql_inference::forward::predict::forward_raw_logits(
        larql_models::WeightsView::dense(weights),
        ids,
        None,
    );
    let mut nll = 0f64;
    for i in 0..ids.len() - 1 {
        let row = rf.h_pre_norm.slice(ndarray::s![i..i + 1, ..]).to_owned();
        let logits = larql_inference::forward::predict::hidden_to_raw_logits(weights, &row);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = max + logits.iter().map(|&l| (l as f64 - max).exp()).sum::<f64>().ln();
        nll += lse - logits[ids[i + 1] as usize] as f64;
    }
    Ok(Fluency { nll, tokens: ids.len() - 1 })
}

fn top1(
    weights: &larql_models::ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    base: &std::path::Path,
    prompt: &str,
    no_chat_template: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let wrapped = if no_chat_template {
        prompt.to_string()
    } else {
        super::chat::render_user_prompt(base, prompt)?
    };
    let enc = tokenizer
        .encode(wrapped.as_str(), true)
        .map_err(|e| format!("tokenize control: {}", e))?;
    let pred = larql_inference::forward::predict(weights, tokenizer, enc.get_ids(), 1);
    Ok(pred
        .predictions
        .first()
        .map(|(t, _)| t.trim().to_string())
        .unwrap_or_default())
}

pub fn run(args: CompileArgs) -> Result<(), Box<dyn std::error::Error>> {
    let prompt = args.prompt.as_ref().unwrap();
    let answer = args.answer.as_ref().unwrap();

    // Parse before loading a 10 GB checkpoint: a typo in --control should cost a second, not
    // the model load plus a forward pass.
    let control_specs = args
        .controls
        .iter()
        .map(|c| parse_control(c))
        .collect::<Result<Vec<_>, _>>()?;

    eprintln!("LARQL AOT Compiler — single mode");
    eprintln!("  base:   {}", args.base.display());
    eprintln!("  prompt: {}...", &prompt[..prompt.len().min(60)]);
    eprintln!("  answer: {}", answer);
    eprintln!("  layer:  {}", args.layer);
    eprintln!("  slot:   {}", args.slot);
    eprintln!("  output: {}", args.output.display());

    eprintln!("\nLoading model...");
    let mut weights = larql_models::loading::load_model_dir(&args.base)?;
    let config = weights.arch.config();
    eprintln!("  {} layers, dim={}", config.num_layers, config.hidden_size);

    let tokenizer_path = args.base.join(TOKENIZER_JSON);
    if !tokenizer_path.exists() {
        return Err(format!("tokenizer.json not found in {}", args.base.display()).into());
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("tokenizer: {}", e))?;

    let (wrapped_prompt, template_source) = if args.no_chat_template {
        (prompt.clone(), "raw (--no-chat-template)".to_string())
    } else {
        let rendered = super::chat::render_user_prompt(&args.base, prompt)?;
        (rendered, "tokenizer_config.chat_template".to_string())
    };
    // Match HF's default tokenisation: add_special_tokens=True adds a BOS
    // on top of whatever the chat template already contains. Served models
    // (Ollama, HF generate) tokenise this way, so our trigger residual
    // must come from the same sequence. See verify_compiled.py.
    let encoding = tokenizer
        .encode(wrapped_prompt.as_str(), true)
        .map_err(|e| format!("tokenize: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("  chat wrap:    {}", template_source);
    eprintln!("  prompt tokens: {}", token_ids.len());

    eprintln!("\nCapturing L{} residual...", args.layer);
    let residuals =
        larql_inference::forward::capture_residuals(&weights, &token_ids, &[args.layer]);
    let (_, residual) = residuals
        .into_iter()
        .find(|(l, _)| *l == args.layer)
        .ok_or("failed to capture residual")?;

    let trigger_norm: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("  trigger norm: {:.2}", trigger_norm);

    let ans_encoding = tokenizer
        .encode(answer.as_str(), false)
        .map_err(|e| format!("tokenize answer: {}", e))?;
    let ans_ids = ans_encoding.get_ids();
    if ans_ids.is_empty() {
        return Err("answer tokenizes to empty".into());
    }
    let ans_token = ans_ids[0];
    eprintln!(
        "  answer token: {} → {:?}",
        ans_token,
        tokenizer.decode(&[ans_token], false).unwrap_or_default()
    );

    let hidden = config.hidden_size;
    let write: Vec<f32> = (0..hidden)
        .map(|j| weights.embed[[ans_token as usize, j]])
        .collect();

    let gate_pattern = detect_ffn_pattern(&weights.tensors, "gate");
    let up_pattern = detect_ffn_pattern(&weights.tensors, "up");
    let down_pattern = detect_ffn_pattern(&weights.tensors, "down");

    let gate_key = gate_pattern.replace("{}", &args.layer.to_string());
    let up_key = up_pattern.replace("{}", &args.layer.to_string());
    let down_key = down_pattern.replace("{}", &args.layer.to_string());

    let mut modified: HashMap<String, ArcArray2<f32>> = HashMap::new();
    for key in [&gate_key, &up_key, &down_key] {
        let original = weights
            .tensors
            .get(key)
            .ok_or_else(|| format!("tensor not found: {}", key))?;
        modified.insert(key.clone(), original.to_owned().into());
    }

    // ── Control baselines, BEFORE the edge exists ──────────────
    if !control_specs.is_empty() {
        eprintln!("\nMeasuring {} control(s) against the base...", control_specs.len());
    }
    let mut baselines: Vec<ControlBaseline> = Vec::new();
    for c in &control_specs {
        let got = top1(&weights, &tokenizer, &args.base, &c.prompt, args.no_chat_template)?;
        eprintln!("  {:?} -> {:?}", c.prompt, got);
        baselines.push(ControlBaseline {
            prompt: c.prompt.clone(),
            expected: c.expected.clone(),
            base_got: got,
        });
    }
    if let Verdict::Refuse(why) = judge_baselines(control_specs.len(), &baselines) {
        return Err(why.into());
    }

    // ── Fluency baseline, BEFORE the edge exists ───────────────
    let fluency_text = match &args.fluency_text {
        Some(p) => Some(std::fs::read_to_string(p).map_err(|e| format!("read --fluency-text {}: {}", p.display(), e))?),
        None => None,
    };
    let fluency_before = match &fluency_text {
        Some(t) => {
            let f = text_fluency(&weights, &tokenizer, t)?;
            eprintln!("  fluency (base): ppl {:.3} over {} tokens", f.ppl(), f.tokens);
            Some(f)
        }
        None => None,
    };

    eprintln!("\nInstalling edge...");
    let stats = install_edge(
        &mut modified,
        &gate_key,
        &up_key,
        &down_key,
        args.slot,
        &residual,
        &write,
        args.gate_scale,
        args.alpha,
    )?;
    eprintln!("  gate_scale={}, alpha={:.3}", args.gate_scale, stats.alpha);
    eprintln!("  installed at L{} slot {}", args.layer, args.slot);

    // ── Balancer: scale the down vector up/down until the target token's
    //    probability lands in [floor, ceiling]. Matches the LQL REBALANCE
    //    convention (larql-lql/src/executor/mutation.rs:948). Each iteration
    //    runs one forward pass so this is the main cost of compile.
    eprintln!(
        "\nBalancing (target '{}' in [{:.2}, {:.2}], max {} iters)...",
        answer, args.floor, args.ceiling, args.max_iters,
    );
    const DOWN_SCALE: f32 = 0.85;
    const UP_SCALE: f32 = 1.15;
    // Disabled and never-converged both fall out of this loop, and used to fall into the same
    // unconditional write. They are different states: --max-iters 0 makes no claim about the
    // install, whereas running out of iterations is a claim that failed.
    let mut balancer = BalancerOutcome::Disabled;
    for iter in 0..args.max_iters {
        // Swap the modified slot tensors into weights for the forward pass
        for key in [&gate_key, &up_key, &down_key] {
            weights.tensors.insert(key.clone(), modified[key].clone());
        }
        let pred = larql_inference::forward::predict(&weights, &tokenizer, &token_ids, 20);
        let prob: f64 = pred
            .predictions
            .iter()
            .find(|(tok, _)| tok.trim() == answer.as_str())
            .map(|(_, p)| *p)
            .unwrap_or(0.0);
        eprintln!("  iter {}: prob('{}') = {:.3}", iter, answer, prob);

        balancer = BalancerOutcome::NotConverged {
            iters: iter + 1,
            last_prob: prob,
            floor: args.floor,
            ceiling: args.ceiling,
        };
        let scale = if prob > args.ceiling {
            DOWN_SCALE
        } else if prob < args.floor {
            UP_SCALE
        } else {
            eprintln!("  converged");
            balancer = BalancerOutcome::Converged { prob };
            break;
        };
        let dt = modified.get_mut(&down_key).unwrap();
        let h = hidden.min(dt.shape()[0]);
        for j in 0..h {
            dt[[j, args.slot]] *= scale;
        }
    }

    // Final swap so weights.tensors carries the final-iteration modified slot.
    for key in [&gate_key, &up_key, &down_key] {
        weights.tensors.insert(key.clone(), modified[key].clone());
    }

    // ── In memory: ADVISORY only. ─────────────────────────────
    // §48 (erasure programme, 2026-09-23): this measurement and the written file disagreed — the
    // in-memory model blanked a control that the written bytes answer at p 0.9999 — so it no
    // longer decides anything. It is printed because a disagreement is itself worth seeing.
    if !control_specs.is_empty() {
        eprintln!("\nIn-memory control check (advisory; the gate judges the WRITTEN file)...");
    }
    for b in &baselines {
        let got = top1(&weights, &tokenizer, &args.base, &b.prompt, args.no_chat_template)?;
        let mark = if got == b.expected { "" } else { "   ← differs in memory" };
        eprintln!("  {:?} -> {:?}{}", b.prompt, got, mark);
    }
    if std::env::var_os("LARQL_COMPILE_DIAG").is_some() && !control_specs.is_empty() {
        // Same in-memory model, with ONLY the edited slot rounded through bf16 as the file stores
        // it. If this agrees with the written file where the unrounded one did not, the
        // disagreement is the rounding of the edge.
        let mut rounded = modified.clone();
        {
            let g = rounded.get_mut(&gate_key).unwrap();
            let r = super::staging::round_through_bf16(&g.row(args.slot).to_vec());
            g.row_mut(args.slot).iter_mut().zip(r).for_each(|(x, v)| *x = v);
        }
        {
            let u = rounded.get_mut(&up_key).unwrap();
            let r = super::staging::round_through_bf16(&u.row(args.slot).to_vec());
            u.row_mut(args.slot).iter_mut().zip(r).for_each(|(x, v)| *x = v);
        }
        {
            let d = rounded.get_mut(&down_key).unwrap();
            let r = super::staging::round_through_bf16(&d.column(args.slot).to_vec());
            d.column_mut(args.slot).iter_mut().zip(r).for_each(|(x, v)| *x = v);
        }
        for key in [&gate_key, &up_key, &down_key] {
            weights.tensors.insert(key.clone(), rounded[key].clone());
        }
        eprintln!("  diag: in memory with the edited slot rounded through bf16:");
        for b in &baselines {
            let got = top1(&weights, &tokenizer, &args.base, &b.prompt, args.no_chat_template)?;
            eprintln!("    {:?} -> {:?}", b.prompt, got);
        }
    }

    // ── Write into STAGING. Nothing reaches the output until the written file is judged. ──
    super::staging::check_output_free(&args.output)?;
    let stage = super::staging::staging_dir(&args.output);
    super::staging::create(&stage)?;
    eprintln!("\nWriting compiled model to staging {}...", stage.display());
    let written = write_checkpoint(&args, &stage, &weights, modified, &gate_key, &up_key, &down_key,
                                   (&gate_pattern, &up_pattern, &down_pattern));
    if let Err(e) = written {
        super::staging::discard(&stage)?;
        return Err(e);
    }
    // free the in-memory model before loading the written one: two f32 copies of a 10 GB
    // checkpoint do not fit on the machines this runs on
    drop(weights);

    // ── The gate: judged on the file as written, reloaded from disk. ──
    let verdict = judge_written(
        &stage,
        &tokenizer,
        &args,
        &baselines,
        control_specs.len(),
        &balancer,
        fluency_before,
        fluency_text.as_deref(),
    );
    match verdict {
        Ok((collateral, fluency)) => {
            super::staging::promote(&stage, &args.output)?;
            eprintln!("\n  {}", collateral);
            eprintln!("  {}", fluency);
            eprintln!("  judged on the written file; promoted to {}", args.output.display());
            Ok(())
        }
        Err(why) => {
            // Refuse with NO output directory: a caller who re-runs after a refusal must not find
            // a half-built output that looks like a previous success.
            super::staging::discard(&stage)?;
            Err(why.into())
        }
    }
}

/// Write the checkpoint into `dir` (the staging directory): a byte-patched copy of the base, or a
/// re-serialised text-only model. Consumes `modified`.
#[allow(clippy::too_many_arguments)]
fn write_checkpoint(
    args: &CompileArgs,
    dir: &std::path::Path,
    weights: &larql_models::ModelWeights,
    modified: HashMap<String, ArcArray2<f32>>,
    gate_key: &str,
    up_key: &str,
    down_key: &str,
    patterns: (&str, &str, &str),
) -> Result<(), Box<dyn std::error::Error>> {
    let output_file = dir.join("model.safetensors");
    if args.byte_patch {
        let base_file = args.base.join("model.safetensors");
        if !base_file.exists() {
            return Err(format!(
                "--byte-patch needs a single-file base checkpoint; {} has no model.safetensors. \
                 A sharded base is not handled yet, and patching one shard while re-serialising \
                 the rest would be worse than refusing.",
                args.base.display()
            )
            .into());
        }
        let edit = super::byte_patch::SlotEdit {
            layer: args.layer,
            slot: args.slot,
            gate: modified[gate_key].row(args.slot).to_vec(),
            up: modified[up_key].row(args.slot).to_vec(),
            down: modified[down_key].column(args.slot).to_vec(),
        };
        let receipt = super::byte_patch::write_byte_patched(&base_file, &output_file, &[edit], patterns)?;
        super::byte_patch::copy_sidecars_verbatim(&args.base, dir)?;
        eprintln!("  byte-patched: {} span(s), {} byte(s) rewritten", receipt.spans, receipt.bytes_written);
        eprintln!("  base sha256:     {}", receipt.sha256_base);
        eprintln!("  compiled sha256: {}", receipt.sha256_out);
        // report the slots the WRITER patched, not the ones the caller asked for: if those
        // ever diverge, the provenance line should say what actually happened to the file
        eprintln!(
            "  differs from its base only inside {}",
            receipt.slots.iter().map(|(l, s)| format!("L{l} slot {s}")).collect::<Vec<_>>().join(", ")
        );
        return Ok(());
    }
    let merged = merge_for_save(weights, modified);
    write_safetensors(&merged.tensors, &merged.vectors, &output_file)?;
    let file_size = std::fs::metadata(&output_file)?.len();
    eprintln!(
        "  saved: {} ({:.1} GB, {} tensors, {} vectors)",
        output_file.display(),
        file_size as f64 / 1e9,
        merged.tensors.len(),
        merged.vectors.len(),
    );
    copy_model_config(&args.base, dir);
    Ok(())
}

/// Reload the written checkpoint and apply the collateral gate and the fluency bound to IT.
/// Returns the two provenance lines on a pass, the refusal reason otherwise.
#[allow(clippy::too_many_arguments)]
fn judge_written(
    dir: &std::path::Path,
    tokenizer: &tokenizers::Tokenizer,
    args: &CompileArgs,
    baselines: &[ControlBaseline],
    requested: usize,
    balancer: &BalancerOutcome,
    fluency_before: Option<Fluency>,
    fluency_text: Option<&str>,
) -> Result<(String, String), String> {
    eprintln!("\nReloading the written file to judge it...");
    let w = larql_models::loading::load_model_dir(dir).map_err(|e| format!("reload written checkpoint: {e}"))?;
    let mut results: Vec<ControlResult> = Vec::new();
    if requested > 0 {
        eprintln!("Re-measuring {} control(s) on the written file...", requested);
    }
    for b in baselines {
        let got = top1(&w, tokenizer, &args.base, &b.prompt, args.no_chat_template).map_err(|e| e.to_string())?;
        let intact = got == b.expected;
        eprintln!("  {:?} -> {:?}{}", b.prompt, got, if intact { "" } else { "   ← CHANGED" });
        results.push(ControlResult { prompt: b.prompt.clone(), expected: b.expected.clone(), got, intact });
    }
    if let Verdict::Refuse(why) = judge(requested, &results, balancer) {
        return Err(why);
    }
    let mut measured_ratio = None;
    if let Some(t) = fluency_text {
        let after = text_fluency(&w, tokenizer, t).ok();
        if let Verdict::Refuse(why) = judge_fluency(fluency_before, after, args.max_fluency_ratio) {
            return Err(why);
        }
        measured_ratio = match (fluency_before, after) {
            (Some(b), Some(a)) => Some(fluency_ratio(b, a)),
            _ => None,
        };
    }
    Ok((
        collateral_provenance(requested, balancer),
        fluency_provenance(fluency_text.is_some(), measured_ratio, args.max_fluency_ratio),
    ))
}

#[cfg(test)]
mod fluency_tests {
    use super::ids_fluency;
    use larql_inference::test_utils::make_test_tokenizer;
    use larql_models::test_fixtures::make_synthetic_e2b_like_weights;

    /// The sequence NLL must agree with `predict`, the path every control is judged by, on the
    /// Gemma-4-E2B-like architecture (per-layer embeddings, shared KV): the NLL `ids_fluency`
    /// charges the LAST token equals -ln p(last | prefix) from `predict_with_temperature`, and by
    /// causality that is exactly what appending the token adds to the prefix's NLL.
    /// E2B-like synthetic weights, with the final norm given VARIED non-unit weights as a trained
    /// model has. With the fixture's default, RMS-norm is idempotent, so applying the final norm
    /// twice changes nothing and a double-norm bug is invisible (mutation-tested 2026-09-23).
    fn e2b_like_with_real_final_norm() -> larql_models::ModelWeights {
        let mut w = make_synthetic_e2b_like_weights();
        let key = w.arch.final_norm_key().to_string();
        let v: Vec<f32> = (0..w.hidden_size).map(|i| 0.3 + 0.17 * (i % 11) as f32).collect();
        w.vectors.insert(key, v);
        w
    }

    #[test]
    fn last_token_nll_matches_predict_on_an_e2b_like_model() {
        let weights = e2b_like_with_real_final_norm();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let prefix: Vec<u32> = vec![3, 7, 1, 12, 5];
        let last: u32 = 9;
        let mut full = prefix.clone();
        full.push(last);

        let with = ids_fluency(&weights, &full).unwrap();
        let without = ids_fluency(&weights, &prefix).unwrap();
        assert_eq!(with.tokens, prefix.len());
        assert_eq!(without.tokens, prefix.len() - 1);
        let appended = with.nll - without.nll;

        let pred = larql_inference::forward::predict::predict_with_temperature(
            &weights, &tokenizer, &prefix, weights.vocab_size, 1.0,
        );
        let idx = pred.token_ids.iter().position(|&t| t == last).expect("last token decodes");
        let p = pred.predictions[idx].1;
        let expected = -p.ln();
        assert!(
            (appended - expected).abs() < 1e-3,
            "appended NLL {appended} disagrees with predict's -ln p = {expected}"
        );
    }

    #[test]
    fn fewer_than_two_tokens_is_an_error_not_a_zero() {
        let weights = make_synthetic_e2b_like_weights();
        assert!(ids_fluency(&weights, &[4]).is_err());
    }
}

