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
use larql_inference::vindex3::{continue_session, Vindex3Runtime};
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
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "vindex3".to_string());
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
    mut on_token: impl FnMut(u32, &str),
) -> Result<V3Generation, ServerError> {
    let mut kv = CanonicalKvState::new();
    let prefill_logits = model
        .runtime
        .prefill_into(prompt_ids, &mut kv)
        .map_err(|e| ServerError::Internal(format!("v3 prefill: {e}")))?;
    let mut session = model
        .runtime
        .session_with_kv(&mut kv)
        .map_err(|e| ServerError::Internal(format!("v3 session: {e}")))?;

    let mut detok = Detokenizer::new(&model.tokenizer);
    detok.seed(prompt_ids);
    let mut texts = Vec::new();
    let result = continue_session(
        &mut session,
        prefill_logits,
        max_tokens,
        sampling,
        eos,
        |id| {
            let text = detok.push(id);
            on_token(id, &text);
            texts.push(text);
        },
    )
    .map_err(|e| ServerError::Internal(format!("v3 decode: {e}")))?;

    let stopped_early = result.tokens.len() < max_tokens;
    Ok(V3Generation {
        ids: result.tokens,
        texts,
        prompt_tokens: prompt_ids.len(),
        stopped_early,
    })
}
