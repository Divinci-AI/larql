//! VI3-SERVE-1: a VINDEX3 container served over the normal API.
//!
//! One vertical slice, deliberately boring:
//!
//! ```text
//! VINDEX3 container → Vindex3Runtime → CanonicalKvState
//!     → prefill_into() → session_with_kv() → continue_session()
//!     → existing SSE/JSON shaping
//! ```
//!
//! What deliberately does **not** happen here: no `load_vindex3() ->
//! ModelWeights`, no `VectorIndex`, no `ModelArchitecture` — a V3
//! container binds as an executable program and is served through the
//! runtime stack INF-0..3 and KV-1 gated bit-for-bit. The V2/V3
//! distinction is decided once, at model binding
//! ([`crate::bootstrap::load_artifact`] /
//! [`crate::state::AppState::served`]); generation code below the
//! binding never asks which format it is running.
//!
//! Per-request cost note: every request opens a fresh session, which
//! loads the plan's operands (`DecodeSession::new` keeps weights
//! resident per session, not per server). Fine for the semantic gate
//! this rung is; a shared resident session/operand pool is later,
//! perf-shaped work.

use std::path::{Path, PathBuf};

use larql_inference::layer_graph::generate::detok::Detokenizer;
use larql_inference::vindex3::{
    continue_session, continue_session_masked, LogitsMask, Vindex3Runtime,
};
use larql_inference::{EosConfig, SamplingConfig};
use larql_kv::CanonicalKvState;
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::tokenizers;

use crate::error::ServerError;
use crate::state::model_id_from_name;

/// Component id a container's text stack is served under.
const SERVED_COMPONENT: &str = "target";

/// One bound VINDEX3 container: the opened runtime plus the serving
/// glue (tokenizer, id). Holds no `ModelWeights` and no `VectorIndex`
/// — structurally, the old inference path is unreachable from here.
pub struct V3Model {
    /// Model ID (derived from the container directory name).
    pub id: String,
    /// Container directory on disk.
    pub path: PathBuf,
    /// The opened executable program + operand store + backend.
    pub runtime: Vindex3Runtime<ProductionBackend>,
    /// Tokenizer for the text-facing API (`tokenizer.json` in the
    /// container directory).
    pub tokenizer: tokenizers::Tokenizer,
}

/// Bind a VINDEX3 container for serving: open the component's plan
/// and operand store (refusing closure defects), and load the
/// container's tokenizer — the text API cannot serve ids-only.
pub fn load_v3_model(path: &Path) -> Result<V3Model, Box<dyn std::error::Error + Send + Sync>> {
    let runtime = Vindex3Runtime::open(path, SERVED_COMPONENT, ProductionBackend::new())
        .map_err(|e| format!("open VINDEX3 container: {e}"))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(path)
        .map_err(|e| format!("VINDEX3 container has no servable tokenizer.json: {e}"))?;
    // The container names itself (`index.model`); the directory name is
    // only the last-resort fallback for a container encoded nameless.
    let name = match runtime.model_name() {
        "" => path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "vindex3".to_string()),
        named => named.to_string(),
    };
    Ok(V3Model {
        id: model_id_from_name(&name),
        path: path.to_path_buf(),
        runtime,
        tokenizer,
    })
}

/// What one V3 generation produced, shaped for the OpenAI routes:
/// per-token surface text in emission order plus the ids behind them.
pub struct V3Generation {
    pub ids: Vec<u32>,
    pub texts: Vec<String>,
    pub prompt_tokens: usize,
    /// True when generation ended before the token budget — the EOS
    /// signal the routes fold into `finish_reason`.
    pub stopped_early: bool,
    /// How many of the run's prompt tokens were served from a resumed
    /// KV state instead of being re-prefilled (0 on a fresh run).
    pub reused_prompt_tokens: usize,
}

/// A generation's continuation state, detached from any session so it
/// can outlive the request (N1): the KV plus exactly the token ids the
/// KV has absorbed. `absorbed_ids` can be one short of prompt+emitted —
/// the driver never steps the final emitted token on a budget stop —
/// which is why the ids travel with the state instead of being
/// re-derived by callers.
pub struct V3KvHandoff {
    pub kv: CanonicalKvState,
    pub absorbed_ids: Vec<u32>,
}

/// The SERVE-1 stack for one request: fresh caller-owned continuation
/// state, batch prefill, resume, drive the sampler — streaming each
/// token's `(id, text)` through `on_token` as it is emitted.
pub fn generate_v3(
    model: &V3Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
    on_token: impl FnMut(u32, &str),
) -> Result<V3Generation, ServerError> {
    generate_v3_resumable(model, prompt_ids, None, max_tokens, sampling, eos, on_token)
        .map(|(generation, _)| generation)
}

/// [`generate_v3`] under a logits mask (N0.6 — tools / structured
/// output on the V3 runtime). `mask_fn` is the V2 constrained driver's
/// contract verbatim — generated-so-far ids plus mutable logits, FSM
/// state in the closure — so one schema-to-mask pipeline serves both
/// runtimes. Constrained runs never resume from a KV handoff (the
/// callers gate that), so this is the fresh-prefill path only.
pub fn generate_v3_constrained(
    model: &V3Model,
    prompt_ids: &[u32],
    max_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
    mask_fn: LogitsMask<'_>,
    on_token: impl FnMut(u32, &str),
) -> Result<V3Generation, ServerError> {
    generate_v3_request(
        model,
        prompt_ids,
        None,
        max_tokens,
        sampling,
        eos,
        Some(mask_fn),
        on_token,
    )
    .map(|(generation, _)| generation)
}

/// [`generate_v3`] with KV continuation (N1). When `resume` carries a
/// prior turn's [`V3KvHandoff`] whose `absorbed_ids` are a strict
/// prefix of `prompt_ids`, only the unseen suffix is prefilled — the
/// resumed positions cost nothing. Any mismatch (different rendering,
/// tokenizer seam effects, an exhausted prompt) falls back to a full
/// fresh prefill, so reuse is purely an optimisation: the produced
/// tokens are identical either way, which the V3 serve tests pin.
///
/// The returned handoff holds the state through this generation for
/// the next chain link.
pub fn generate_v3_resumable(
    model: &V3Model,
    prompt_ids: &[u32],
    resume: Option<V3KvHandoff>,
    max_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
    on_token: impl FnMut(u32, &str),
) -> Result<(V3Generation, V3KvHandoff), ServerError> {
    generate_v3_request(
        model, prompt_ids, resume, max_tokens, sampling, eos, None, on_token,
    )
}

/// The one V3 generation body behind the free, resumable, and
/// constrained entry points — and the direct entry for callers that
/// need BOTH knobs (a chained Responses request under a
/// `response_format` constraint resumes AND masks; the two are
/// orthogonal: resume shapes prefill, the mask shapes sampling).
#[allow(clippy::too_many_arguments)]
pub fn generate_v3_request(
    model: &V3Model,
    prompt_ids: &[u32],
    resume: Option<V3KvHandoff>,
    max_tokens: usize,
    sampling: SamplingConfig,
    eos: &EosConfig,
    mask_fn: Option<LogitsMask<'_>>,
    mut on_token: impl FnMut(u32, &str),
) -> Result<(V3Generation, V3KvHandoff), ServerError> {
    // A handoff is resumable only when the new prompt extends exactly
    // what the KV already absorbed.
    let resumed = resume.filter(|h| {
        !h.absorbed_ids.is_empty()
            && h.absorbed_ids.len() < prompt_ids.len()
            && prompt_ids.starts_with(&h.absorbed_ids)
    });
    let (mut kv, reused_prompt_tokens) = match resumed {
        Some(h) => (h.kv, h.absorbed_ids.len()),
        None => (CanonicalKvState::new(), 0),
    };

    let prefill_logits = model
        .runtime
        .prefill_into(&prompt_ids[reused_prompt_tokens..], &mut kv)
        .map_err(|e| ServerError::Internal(format!("v3 prefill: {e}")))?;
    let mut session = model
        .runtime
        .session_with_kv(&mut kv)
        .map_err(|e| ServerError::Internal(format!("v3 session: {e}")))?;

    let mut detok = Detokenizer::new(&model.tokenizer);
    detok.seed(prompt_ids);
    let mut texts = Vec::new();
    let mut emit = |id: u32| {
        let text = detok.push(id);
        on_token(id, &text);
        texts.push(text);
    };
    let result = match mask_fn {
        Some(mask_fn) => continue_session_masked(
            &mut session,
            prefill_logits,
            max_tokens,
            sampling,
            eos,
            mask_fn,
            &mut emit,
        ),
        None => continue_session(
            &mut session,
            prefill_logits,
            max_tokens,
            sampling,
            eos,
            &mut emit,
        ),
    }
    .map_err(|e| ServerError::Internal(format!("v3 decode: {e}")))?;
    drop(session);

    // The KV's logical position says exactly how many of
    // prompt + emitted it absorbed (the driver never steps the final
    // emitted token on a budget stop).
    let absorbed_len = larql_vindex::format::vindex3::opplan::exec::kv::KvState::position(&kv);
    let mut absorbed_ids = Vec::with_capacity(absorbed_len);
    absorbed_ids.extend_from_slice(prompt_ids);
    absorbed_ids.extend_from_slice(&result.tokens);
    absorbed_ids.truncate(absorbed_len);

    let stopped_early = result.tokens.len() < max_tokens;
    Ok((
        V3Generation {
            ids: result.tokens,
            texts,
            prompt_tokens: prompt_ids.len(),
            stopped_early,
            reused_prompt_tokens,
        },
        V3KvHandoff { kv, absorbed_ids },
    ))
}
