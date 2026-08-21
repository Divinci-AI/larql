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
const SUPPORTED: &str = "INFER [TOP n] [GENERATE n], STATS, SHOW LAYERS, USE";

/// Component id a container's text stack is bound under.
pub(crate) const V3_COMPONENT: &str = "target";

type V3Runtime = Vindex3Runtime<ProductionBackend>;

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
            runtime, tokenizer, ..
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
        let mut kv = CanonicalKvState::new();
        let prefill_logits = runtime
            .prefill_into(&prompt_ids, &mut kv)
            .map_err(|e| LqlError::exec("v3 prefill failed", e))?;

        match generate {
            None => {
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                let mut out = Vec::new();
                out.push("Predictions (VINDEX3 program):".into());
                for (i, (id, prob)) in top_k_probs(&prefill_logits, top_k).iter().enumerate() {
                    let token = tokenizer.decode(&[*id], false).unwrap_or_default();
                    out.push(format!(
                        "  {:2}. {:20} ({:.2}%)  [id {}]",
                        i + 1,
                        token,
                        prob * 100.0,
                        id,
                    ));
                }
                out.push(format!("  {:.0}ms", elapsed_ms));
                Ok(out)
            }
            Some(n) => {
                let mut session = runtime
                    .session_with_kv(&mut kv)
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

/// Open a container as a V3 binding: runtime (refusing closure
/// defects) plus the optional tokenizer capability.
pub(crate) fn bind(path: &std::path::Path) -> Result<(V3Runtime, Option<Tokenizer>), LqlError> {
    let runtime = Vindex3Runtime::open(path, V3_COMPONENT, ProductionBackend::new())
        .map_err(|e| LqlError::exec("failed to open VINDEX3 container", e))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(path).ok();
    Ok((runtime, tokenizer))
}

fn encode_v3_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>, LqlError> {
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
fn top_k_probs(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
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
