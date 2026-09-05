//! Session-scoped chat completions: decode through the edited overlay.
//!
//! `run_chat_completion` generates from `patched.base()`, the unedited
//! index, so a request that carried `X-Session-Id` was answered by the
//! base model even when the session held patches. This path is what lets
//! an edit reach a chat turn. It renders the conversation with the
//! model's template and then decodes greedily, one token per step,
//! through [`larql_inference::infer_patched`] under the session overlay:
//! the only forward pass that observes tombstones and the KnnStore.
//!
//! There is no KV cache here. Every step is a full walk pass over the
//! prompt plus the tokens generated so far, so cost grows with the
//! answer. It is a preview path: bound it with `max_tokens`, and keep
//! the base-index path for sessions that hold no edits.

use crate::error::ServerError;
use crate::routes::openai::prompt::{pick_template, render};
use crate::routes::openai::util::{
    contains_any, trim_at_stop, FINISH_REASON_LENGTH, FINISH_REASON_STOP,
};
use crate::state::{AppState, LoadedModel};

use super::handler::ChatGenerationOutput;
use super::types::ChatMessage;

/// Turn / sequence terminators across the families larql serves; resolved
/// to ids per model, missing ones skipped.
const END_OF_TURN_TOKENS: &[&str] = &[
    "<turn|>",
    "<end_of_turn>",
    "<eos>",
    "</s>",
    "<|endoftext|>",
    "<|im_end|>",
    "<|end_of_turn|>",
    "<|eot_id|>",
];

/// Greedy decode under the overlay named by `session_id` and/or `patch_set`
/// (the same resolution rule `/v1/infer` uses).
pub(super) fn run_chat_completion_patched(
    state: &AppState,
    model: &LoadedModel,
    session_id: Option<&str>,
    patch_set: Option<&crate::overlay_cache::PatchSetRef>,
    messages: &[ChatMessage],
    max_tokens: usize,
    stop_strings: &[String],
) -> Result<ChatGenerationOutput, ServerError> {
    if model.infer_disabled {
        return Err(ServerError::InferenceUnavailable(
            "inference disabled (--no-infer)".into(),
        ));
    }
    // A read guard, like `/v1/infer`: the walk pass does not touch the
    // per-layer dequant cache that the base-index generator mutates.
    let weights_guard = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;
    let weights: &larql_inference::ModelWeights = &weights_guard;

    let template = pick_template(weights);
    let prompt = render(template, messages);
    let encoding = model
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
    if ids.is_empty() {
        return Err(ServerError::BadRequest(
            "rendered prompt tokenises to empty".into(),
        ));
    }
    let prompt_tokens = ids.len();
    let route_mode = larql_inference::KnnRouteMode::from_env();
    // End-of-turn by id as well as by text: with `skip_special_tokens` a
    // marker such as `<turn|>` decodes to an empty string, so a text check
    // alone never fires and the model keeps writing past its own turn.
    let end_ids: Vec<u32> = END_OF_TURN_TOKENS
        .iter()
        .filter_map(|t| model.tokenizer.token_to_id(t))
        .collect();

    let mut text = String::new();
    let mut tokens: Vec<(String, f64)> = Vec::new();
    let mut finish_reason: &'static str = FINISH_REASON_LENGTH;

    for _ in 0..max_tokens {
        let step =
            crate::overlay_cache::with_overlay(state, model, session_id, patch_set, |patched| {
                larql_inference::infer_patched(
                    weights,
                    &model.tokenizer,
                    patched,
                    Some(&patched.knn_store),
                    &ids,
                    1,
                    &route_mode,
                )
            })?;
        let Some((token, prob)) = step.predictions.first().cloned() else {
            finish_reason = FINISH_REASON_STOP;
            break;
        };
        let ended_by_id = step.knn_override.is_none()
            && step.model_top1_id.is_some_and(|id| end_ids.contains(&id));
        if ended_by_id || larql_inference::vindex::is_end_of_turn(&token) {
            finish_reason = FINISH_REASON_STOP;
            break;
        }
        match next_ids(&model.tokenizer, &step, &token) {
            Some(next) if !next.is_empty() => ids.extend(next),
            _ => {
                finish_reason = FINISH_REASON_STOP;
                break;
            }
        }
        text.push_str(&token);
        tokens.push((token, prob));
        if !stop_strings.is_empty() && contains_any(&text, stop_strings) {
            text = trim_at_stop(&text, stop_strings);
            finish_reason = FINISH_REASON_STOP;
            break;
        }
    }

    let completion_tokens = tokens.len();
    Ok(ChatGenerationOutput {
        text,
        tokens,
        finish_reason,
        prompt_tokens,
        completion_tokens,
    })
}

/// The ids to append for the token the step chose. The walk's own answer
/// comes with its id; a KnnStore override is a stored surface string, so
/// it is re-tokenised (it was never a single model token to begin with).
fn next_ids(
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    step: &larql_inference::forward::InferPatchedResult,
    token: &str,
) -> Option<Vec<u32>> {
    if step.knn_override.is_none() {
        if let Some(id) = step.model_top1_id {
            return Some(vec![id]);
        }
    }
    tokenizer
        .encode(token, false)
        .ok()
        .map(|e| e.get_ids().to_vec())
}
