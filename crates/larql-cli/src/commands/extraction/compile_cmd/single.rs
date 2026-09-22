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
    collateral_provenance, judge, judge_baselines, parse_control, BalancerOutcome,
    ControlBaseline, ControlResult, Verdict,
};
use super::detect::detect_ffn_pattern;
use super::edge::install_edge;
use super::save::{copy_model_config, merge_for_save, write_safetensors};
use super::CompileArgs;

/// Top-1 continuation for a prompt, wrapped exactly as the trigger prompt was.
///
/// A control measured through a different chat wrap than the install is measuring a different
/// input, so the wrap is threaded through rather than re-decided here.
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

    // ── The gate. Everything above measured; this decides. ────
    let mut results: Vec<ControlResult> = Vec::new();
    if !control_specs.is_empty() {
        eprintln!("\nRe-measuring {} control(s) after the install...", control_specs.len());
    }
    for b in &baselines {
        let got = top1(&weights, &tokenizer, &args.base, &b.prompt, args.no_chat_template)?;
        let intact = got == b.expected;
        eprintln!("  {:?} -> {:?}{}", b.prompt, got, if intact { "" } else { "   ← CHANGED" });
        results.push(ControlResult {
            prompt: b.prompt.clone(),
            expected: b.expected.clone(),
            got,
            intact,
        });
    }
    if let Verdict::Refuse(why) = judge(control_specs.len(), &results, &balancer) {
        // Refuse BEFORE create_dir_all: a caller who re-runs after a refusal should not find a
        // half-built output directory that looks like a previous success.
        return Err(why.into());
    }
    eprintln!("\n  {}", collateral_provenance(control_specs.len(), &balancer));

    eprintln!("\nSaving compiled model...");
    std::fs::create_dir_all(&args.output)?;
    let output_file = args.output.join("model.safetensors");

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
            gate: modified[&gate_key].row(args.slot).to_vec(),
            up: modified[&up_key].row(args.slot).to_vec(),
            down: modified[&down_key].column(args.slot).to_vec(),
        };
        let receipt = super::byte_patch::write_byte_patched(
            &base_file,
            &output_file,
            &[edit],
            (&gate_pattern, &up_pattern, &down_pattern),
        )?;
        super::byte_patch::copy_sidecars_verbatim(&args.base, &args.output)?;
        eprintln!(
            "  byte-patched: {} span(s), {} byte(s) rewritten",
            receipt.spans, receipt.bytes_written
        );
        eprintln!("  base sha256:     {}", receipt.sha256_base);
        eprintln!("  compiled sha256: {}", receipt.sha256_out);
        // report the slots the WRITER patched, not the ones the caller asked for: if those
        // ever diverge, the provenance line should say what actually happened to the file
        eprintln!(
            "  differs from its base only inside {}",
            receipt
                .slots
                .iter()
                .map(|(l, s)| format!("L{l} slot {s}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        return Ok(());
    }

    let merged = merge_for_save(&weights, modified);
    write_safetensors(&merged.tensors, &merged.vectors, &output_file)?;

    let file_size = std::fs::metadata(&output_file)?.len();
    eprintln!(
        "  saved: {} ({:.1} GB, {} tensors, {} vectors)",
        output_file.display(),
        file_size as f64 / 1e9,
        merged.tensors.len(),
        merged.vectors.len(),
    );

    copy_model_config(&args.base, &args.output);

    eprintln!("\nDone.");
    eprintln!(
        "  larql compile --base {} --prompt \"...\" --answer \"{}\" → {}",
        args.base.display(),
        answer,
        args.output.display()
    );
    Ok(())
}
