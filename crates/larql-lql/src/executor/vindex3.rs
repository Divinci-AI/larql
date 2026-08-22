//! VINDEX3 statement execution (LQL-1: USE / SHOW / INFER).
//!
//! The invariant this module exists to keep: **LQL binds a model once,
//! then operates on the runtime's declared facts and capabilities.**
//! No statement here reconstructs architecture from weights or family
//! metadata — every fact printed by `STATS` / `SHOW LAYERS` is read
//! from the container's executable plan, and `INFER` runs the same
//! runtime seam `larql-server` serves
//! (`prefill_into → session_with_kv → continue_session`), so a fourth
//! entry point joins the equality chain the SERVE-1 gates established.

use larql_inference::layer_graph::generate::detok::Detokenizer;
use larql_inference::vindex3::{continue_session, plan_kv_geometry, Vindex3Runtime};
use larql_inference::{EosConfig, SamplingConfig};
use larql_kv::CanonicalKvState;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::tokenizers::Tokenizer;

use crate::error::LqlError;
use crate::executor::{Backend, Session};

/// The statements a VINDEX3 binding serves today. Everything else gets
/// [`unsupported`] — a capability refusal, not a format apology.
pub(crate) const SUPPORTED: &str = "SELECT, DESCRIBE, WALK, EXPLAIN WALK, \
     SHOW RELATIONS/LAYERS/FEATURES/ENTITIES/PATCHES, INFER [TOP n] [GENERATE n], \
     EXPLAIN INFER, TRACE, STATS, USE, INSERT [MODE KNN|COMPOSE], DELETE, UPDATE, MERGE, \
     BEGIN/SAVE/APPLY/REMOVE PATCH, COMPILE [CURRENT] INTO VINDEX, DIFF [PHYSICAL], COMPACT INTO VINDEX";

/// Component id a container's text stack is bound under.
pub(crate) const V3_COMPONENT: &str = "target";

pub(crate) type V3Runtime = Vindex3Runtime<ProductionBackend>;

/// The capability refusal for statements a V3 binding does not serve.
pub(crate) fn unsupported(what: &str) -> LqlError {
    LqlError::Execution(format!(
        "{what} is not supported on a VINDEX3 container yet. \
         Supported: {SUPPORTED}."
    ))
}

impl Session {
    /// `INFER` on a V3 binding. Without `GENERATE`: classic single-step
    /// top-k next-token prediction, priced from the batch-prefill
    /// logits. With `GENERATE n`: greedy autoregressive continuation
    /// through the proven runtime stack.
    pub(crate) fn exec_v3_infer(
        &self,
        prompt: &str,
        top_k: usize,
        generate: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let Backend::Vindex3 {
            runtime,
            tokenizer,
            overlay,
            ..
        } = &self.backend
        else {
            unreachable!("caller matched the backend");
        };
        let tokenizer = tokenizer.as_ref().ok_or_else(|| {
            LqlError::Execution(
                "INFER needs a tokenizer and this container carries no tokenizer.json — \
                 token-id capability only"
                    .into(),
            )
        })?;
        let prompt_ids = encode_v3_prompt(tokenizer, prompt)?;

        let start = std::time::Instant::now();

        match generate {
            None => {
                // One observed pass over the same traversal generation
                // runs: logits for the display, per-layer residual taps
                // for the KnnStore override (only captured while the
                // overlay holds entries).
                let capture_residuals = !overlay.knn_store.is_empty();
                let mut residuals: Vec<(usize, Vec<f32>)> = Vec::new();
                let mut sink = |event: larql_inference::vindex3::PlaneEvent| {
                    if capture_residuals {
                        if let larql_inference::vindex3::PlaneEvent::Layer { index, trace } = event
                        {
                            // Normed FFN inputs — the same tap the
                            // stored keys were captured from.
                            if let Some(last) = trace.ffn_input.last() {
                                residuals.push((index, last.clone()));
                            }
                        }
                    }
                    Ok(())
                };
                // Compose edits reach execution through the operand-
                // source seam; without them this is the plain pass,
                // bit for bit.
                let output = match compose_overrides(runtime, overlay)? {
                    Some(overrides) => runtime
                        .execute_streaming_overlaid(&prompt_ids, &overrides, &mut sink)
                        .map_err(|e| LqlError::exec("v3 prefill failed", e))?,
                    None => runtime
                        .execute_streaming(&prompt_ids, &mut sink)
                        .map_err(|e| LqlError::exec("v3 prefill failed", e))?,
                };
                let logits = output.logits.ok_or_else(|| {
                    LqlError::Execution("the component carries no output head".into())
                })?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                let scored = top_k_probs(&logits, top_k);
                // The shared post-logits resolution rule — same
                // first-stored-layer, fixed-threshold gate V2 runs.
                let raw: Vec<(String, f64)> = scored
                    .iter()
                    .map(|(id, prob)| {
                        (
                            tokenizer.decode(&[*id], false).unwrap_or_default(),
                            *prob as f64,
                        )
                    })
                    .collect();
                let (_, knn_override) = larql_inference::apply_knn_override(
                    raw,
                    &residuals,
                    Some(&overlay.knn_store),
                    top_k,
                );

                let mut out = Vec::new();
                out.push("Predictions (VINDEX3 program):".into());
                match &knn_override {
                    Some(ovr) => {
                        let model_top1 = scored.first().map(|(id, prob)| {
                            (
                                tokenizer.decode(&[*id], false).unwrap_or_default(),
                                *prob as f64,
                            )
                        });
                        out.push(format!(
                            "   1. {:20} (100.00%, {})",
                            ovr.token,
                            crate::executor::helpers::format_knn_override_summary(
                                ovr,
                                model_top1.as_ref(),
                            ),
                        ));
                        for (i, (id, prob)) in
                            scored.iter().take(top_k.saturating_sub(1)).enumerate()
                        {
                            let token = tokenizer.decode(&[*id], false).unwrap_or_default();
                            out.push(format!(
                                "  {:2}. {:20} ({:.2}%)  [id {}]",
                                i + 2,
                                token,
                                prob * 100.0,
                                id,
                            ));
                        }
                    }
                    None => {
                        for (i, (id, prob)) in scored.iter().enumerate() {
                            let token = tokenizer.decode(&[*id], false).unwrap_or_default();
                            out.push(format!(
                                "  {:2}. {:20} ({:.2}%)  [id {}]",
                                i + 1,
                                token,
                                prob * 100.0,
                                id,
                            ));
                        }
                    }
                }
                out.push(format!("  {:.0}ms", elapsed_ms));
                if knn_override.is_some() {
                    out.push(
                        "  note: KNN override is a post-logits retrieval sidecar, not an \
                         FFN/residual edit."
                            .into(),
                    );
                }
                Ok(out)
            }
            Some(n) => {
                let mut kv = CanonicalKvState::new();
                let overrides = compose_overrides(runtime, overlay)?;
                let prefill_logits = match &overrides {
                    Some(ov) => runtime.prefill_into_overlaid(&prompt_ids, ov, &mut kv),
                    None => runtime.prefill_into(&prompt_ids, &mut kv),
                }
                .map_err(|e| LqlError::exec("v3 prefill failed", e))?;
                let mut session = match &overrides {
                    Some(ov) => runtime.session_with_kv_overlaid(&mut kv, ov),
                    None => runtime.session_with_kv(&mut kv),
                }
                .map_err(|e| LqlError::exec("v3 session failed", e))?;
                let mut detok = Detokenizer::new(tokenizer);
                detok.seed(&prompt_ids);
                let mut text = String::new();
                let result = continue_session(
                    &mut session,
                    prefill_logits,
                    n as usize,
                    SamplingConfig::greedy(),
                    &EosConfig::builtin(),
                    |id| text.push_str(&detok.push(id)),
                )
                .map_err(|e| LqlError::exec("v3 decode failed", e))?;
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

                let ids: Vec<String> = result.tokens.iter().map(u32::to_string).collect();
                Ok(vec![
                    format!("Generated ({} tokens, greedy):", result.tokens.len()),
                    format!("  ids:  {}", ids.join(",")),
                    format!("  text: {}", text),
                    format!(
                        "  prompt {} tokens, continued from position {}",
                        result.prompt_len, result.prompt_len,
                    ),
                    format!("  {:.0}ms", elapsed_ms),
                ])
            }
        }
    }

    /// `STATS` on a V3 binding: the container's own authority, not a
    /// reconstruction — generation, component, closure, and geometry as
    /// the executable plan declares them.
    pub(crate) fn exec_v3_stats(&self) -> Result<Vec<String>, LqlError> {
        let Backend::Vindex3 {
            path,
            runtime,
            tokenizer,
            ..
        } = &self.backend
        else {
            unreachable!("caller matched the backend");
        };
        let plan = runtime.plan();
        let geometry = plan_kv_geometry(plan);
        let sliding = geometry.iter().filter(|g| g.window.is_some()).count();
        let full = geometry.len() - sliding;
        let kv_dims: std::collections::BTreeSet<usize> =
            geometry.iter().map(|g| g.kv_dim).collect();
        let kv_dims: Vec<String> = kv_dims.iter().map(usize::to_string).collect();

        Ok(vec![
            format!("Model:           {} (VINDEX3)", runtime.model_name()),
            format!("Path:            {}", path.display()),
            "Generation:      3".into(),
            format!("Component:       {V3_COMPONENT}"),
            "Execution:       closed (operand-verified executable plan)".into(),
            format!("Layers:          {}", plan.layers.len()),
            format!("Attention:       {sliding} sliding / {full} full (windows from the plan)"),
            format!(
                "KV geometry:     plan-derived; kv_dim {}",
                kv_dims.join(", ")
            ),
            format!(
                "Output head:     {}",
                if plan.output.is_some() {
                    "present"
                } else {
                    "absent"
                }
            ),
            format!(
                "Tokenizer:       {}",
                if tokenizer.is_some() {
                    "present"
                } else {
                    "absent (token-id capability only)"
                }
            ),
            format!("Capabilities:    {SUPPORTED}"),
        ])
    }

    /// `SHOW LAYERS` on a V3 binding: per-layer attention facts, read
    /// off the executable plan.
    pub(crate) fn exec_v3_show_layers(&self) -> Result<Vec<String>, LqlError> {
        let Backend::Vindex3 { runtime, .. } = &self.backend else {
            unreachable!("caller matched the backend");
        };
        let plan = runtime.plan();
        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<10} {:>8} {:>10} {:>10} {:>8}",
            "Layer", "Attention", "Window", "Q heads", "KV heads", "Head dim"
        ));
        out.push("-".repeat(60));
        for (index, layer) in plan.layers.iter().enumerate() {
            let attention = &layer.attention;
            let (kind, window) = match attention.window {
                Some(w) => ("sliding", w.to_string()),
                None => ("full", "-".to_string()),
            };
            out.push(format!(
                "{:<8} {:<10} {:>8} {:>10} {:>10} {:>8}",
                index,
                kind,
                window,
                attention.num_q_heads,
                attention.num_kv_heads,
                attention.head_dim,
            ));
        }
        Ok(out)
    }
}

impl Session {
    /// `EXPLAIN INFER` on a V3 binding (LQL-2): render the structured
    /// [`ExplainPlan`] — the executable authority that will run, not a
    /// reconstruction. Static: no tokens execute.
    pub(crate) fn exec_v3_explain(&self) -> Result<Vec<String>, LqlError> {
        let Backend::Vindex3 { runtime, .. } = &self.backend else {
            unreachable!("caller matched the backend");
        };
        let explain = larql_inference::vindex3::ExplainPlan::from_runtime(runtime);

        let mut out = Vec::new();
        out.push("MODEL".into());
        out.push(format!("  name: {}", explain.model));
        out.push(format!("  generation: {}", explain.generation));
        out.push(format!("  component: {}", explain.component));
        out.push("  execution: closed".into());
        out.push(String::new());
        out.push("PLAN".into());
        out.push(format!(
            "  embedding  vocab {} scaled {} normed {}",
            explain.embedding.vocab_size, explain.embedding.scaled, explain.embedding.normed,
        ));
        for layer in &explain.layers {
            out.push(format!("  layer {}", layer.layer));
            for op in &layer.ops {
                match op.as_str() {
                    "attention" => {
                        let a = &layer.attention;
                        out.push(format!(
                            "    attention  mode {} window {}  q/kv {}/{}  head_dim {}{}{}",
                            a.mode,
                            a.window.map_or("-".into(), |w| w.to_string()),
                            a.q_heads,
                            a.kv_heads,
                            a.head_dim,
                            if a.gated { "  gated" } else { "" },
                            if a.qk_norm { "  qk_norm" } else { "" },
                        ));
                        if a.sinks || a.biased {
                            out.push(format!(
                                "      extras      {}{}",
                                if a.sinks { " sinks" } else { "" },
                                if a.biased { " qkvo_bias" } else { "" },
                            ));
                        }
                        for operand in &a.operands {
                            out.push(format!(
                                "      {:12} {}::{} @{}",
                                operand.role, operand.object, operand.tensor, operand.dtype,
                            ));
                        }
                    }
                    "ffn" => {
                        let f = &layer.ffn;
                        out.push(format!(
                            "    ffn        kind {}{}",
                            f.kind,
                            f.experts
                                .map_or(String::new(), |(e, k)| format!("  experts {e} top_k {k}")),
                        ));
                        for operand in &f.operands {
                            out.push(format!(
                                "      {:12} {}::{} @{}",
                                operand.role, operand.object, operand.tensor, operand.dtype,
                            ));
                        }
                    }
                    other => out.push(format!("    {other}")),
                }
            }
        }
        out.push(String::new());
        out.push("CONTINUATION".into());
        out.push("  provider: caller-owned KvState (CanonicalKvState default)".into());
        for (layer, g) in explain.continuation.iter().enumerate() {
            out.push(format!(
                "  layer {layer}: kv_dim {} window {}",
                g.kv_dim,
                g.window.map_or("-".into(), |w| w.to_string()),
            ));
        }
        out.push(String::new());
        out.push("OUTPUT".into());
        match &explain.output {
            Some(head) => {
                out.push(format!(
                    "  output_head: present  vocab {}{}{}",
                    head.vocab,
                    if head.multiplied { "  multiplied" } else { "" },
                    if head.softcapped { "  softcapped" } else { "" },
                ));
            }
            None => out.push("  output_head: absent".into()),
        }
        out.push(format!(
            "  final_norm: {}",
            if explain.final_norm {
                "present"
            } else {
                "absent"
            }
        ));
        Ok(out)
    }

    /// `TRACE "prompt"` on a V3 binding (LQL-2): observe the canonical
    /// executor while it ingests the prompt, then report the greedy
    /// next token. Observation is subscription — the executor's parity
    /// gate pins that tracing never changes arithmetic, and the LQL
    /// gate pins that the reported token equals INFER's.
    pub(crate) fn exec_v3_trace(&self, prompt: &str) -> Result<Vec<String>, LqlError> {
        use larql_inference::vindex3::{RecordingObserver, StepEvent};
        let Backend::Vindex3 {
            runtime,
            tokenizer,
            overlay,
            ..
        } = &self.backend
        else {
            unreachable!("caller matched the backend");
        };
        let tokenizer = tokenizer.as_ref().ok_or_else(|| {
            LqlError::Execution("TRACE needs a tokenizer — this container carries none".into())
        })?;
        let prompt_ids = encode_v3_prompt(tokenizer, prompt)?;

        // TRACE observes the same effective program INFER runs — a
        // compose edit must not fork the two.
        let mut session = match compose_overrides(runtime, overlay)? {
            Some(overrides) => runtime.session_overlaid(&overrides),
            None => runtime.session(),
        }
        .map_err(|e| LqlError::exec("v3 session failed", e))?;
        let mut out = vec!["Trace (VINDEX3 program, observed execution):".into()];
        let mut logits = Vec::new();
        for (offset, &token) in prompt_ids.iter().enumerate() {
            let mut recorder = RecordingObserver::default();
            logits = session
                .step_observed(token, &mut recorder)
                .map_err(|e| LqlError::exec("v3 step failed", e))?;
            out.push(format!("position {offset} (prompt token {token})"));
            for event in &recorder.events {
                match event {
                    StepEvent::Embedded { .. } => out.push("  embed".into()),
                    StepEvent::AttentionDone { layer } => {
                        out.push(format!("  layer {layer}: attention"))
                    }
                    StepEvent::FfnDone { layer } => out.push(format!("  layer {layer}: ffn")),
                    StepEvent::Logits { vocab } => {
                        out.push(format!("  output_head (vocab {vocab})"))
                    }
                }
            }
        }
        let mut sampler = larql_inference::Sampler::new(SamplingConfig::greedy());
        let next = sampler
            .sample(&logits)
            .ok_or_else(|| LqlError::Execution("no next token from the logits".into()))?;
        let text = tokenizer.decode(&[next], false).unwrap_or_default();
        out.push(format!("next token {next} {text:?} (greedy)"));
        Ok(out)
    }
}

/// Open a container as a V3 binding: runtime (refusing closure
/// defects) plus the optional tokenizer capability.
pub(crate) type V3Knowledge = larql_vindex::format::vindex3::knowledge::KnowledgeView;

pub(crate) fn bind(
    path: &std::path::Path,
) -> Result<(V3Runtime, Option<Tokenizer>, Option<V3Knowledge>), LqlError> {
    let runtime = Vindex3Runtime::open(path, V3_COMPONENT, ProductionBackend::new())
        .map_err(|e| LqlError::exec("failed to open VINDEX3 container", e))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(path).ok();
    // The browse view needs the tokenizer (feature annotations decode
    // token ids); a tokenizer-less container binds without it.
    let knowledge = match &tokenizer {
        Some(tok) => Some(
            runtime
                .knowledge_view(tok)
                .map_err(|e| LqlError::exec("failed to bind the V3 query surface", e))?,
        ),
        None => None,
    };
    Ok((runtime, tokenizer, knowledge))
}

/// One observed pass over the runtime's plan, returning the last
/// position's **normed FFN input** at `layer` — V2's exact install
/// statistic (its walk-FFN trace captures the post-norm vector the
/// gates multiply), so a gate built from it fires on the prompt that
/// produced it (V3-LQL-3B capture: the KNN key and the compose gate
/// direction). The pass is the same canonical traversal INFER runs;
/// the tap is a subscription, never a second executor.
/// The overlay's compose edits as executor operand overrides — `None`
/// while no compose state exists, so callers keep the plain (bit-for-
/// bit identical) execution path.
pub(crate) fn compose_overrides(
    runtime: &V3Runtime,
    overlay: &larql_vindex::format::vindex3::knowledge::KnowledgeOverlay,
) -> Result<Option<larql_inference::vindex3::OperandOverrides>, LqlError> {
    if !overlay.has_vector_state() {
        return Ok(None);
    }
    overlay
        .operand_overrides(runtime.plan())
        .map(Some)
        .map_err(|e| LqlError::exec("failed to derive operand overrides", e))
}

pub(crate) fn capture_layer_residual(
    runtime: &V3Runtime,
    tokenizer: &Tokenizer,
    prompt: &str,
    layer: usize,
    overrides: Option<&larql_inference::vindex3::OperandOverrides>,
) -> Result<Vec<f32>, LqlError> {
    let prompt_ids = encode_v3_prompt(tokenizer, prompt)?;
    let mut captured: Option<Vec<f32>> = None;
    let mut sink = |event: larql_inference::vindex3::PlaneEvent| {
        if let larql_inference::vindex3::PlaneEvent::Layer { index, trace } = event {
            if index == layer {
                captured = trace.ffn_input.last().cloned();
            }
        }
        Ok(())
    };
    // The capture runs over the same effective program INFER runs —
    // V2's contract (its capture forward observes the patch overlay).
    match overrides {
        Some(overrides) => runtime.execute_streaming_overlaid(&prompt_ids, overrides, &mut sink),
        None => runtime.execute_streaming(&prompt_ids, &mut sink),
    }
    .map_err(|e| LqlError::exec("v3 capture pass failed", e))?;
    captured.ok_or_else(|| LqlError::Execution(format!("no residual captured at layer {layer}")))
}

pub(crate) fn encode_v3_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>, LqlError> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| LqlError::Execution(format!("tokenize: {e}")))?;
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    if ids.is_empty() {
        return Err(LqlError::Execution("prompt tokenises to empty".into()));
    }
    Ok(ids)
}

/// Softmax over the logits, then the top-k `(token_id, probability)`
/// pairs, ties keeping the lower id (the greedy sampler's rule).
pub(crate) fn top_k_probs(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut scored: Vec<(u32, f32)> = exps
        .iter()
        .enumerate()
        .map(|(id, &e)| (id as u32, e / sum))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}
