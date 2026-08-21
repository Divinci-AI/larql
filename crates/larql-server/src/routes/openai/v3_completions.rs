//! `/v1/completions` served by a VINDEX3 runtime (VI3-SERVE-1).
//!
//! Reached only through [`AppState::served`] resolving the request's
//! model to a [`V3Model`] — the one place the V2/V3 distinction is
//! decided. Everything wire-shaped is **shared** with the V2 path
//! (`build_text_completion_chunk`, `finalize_completion`, the SSE
//! assembly, the response structs), so the two runtimes cannot drift
//! apart in what a client sees; only the token source differs:
//!
//! ```text
//! V2: ModelWeights + VectorIndex → generate_streaming
//! V3: Vindex3Runtime → CanonicalKvState → prefill_into
//!       → session_with_kv → continue_session
//! ```

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use crate::error::ServerError;
use crate::vindex3::{generate_v3, V3Model};

use super::completions::{
    build_text_completion_chunk, finalize_completion, CompletionChoice, CompletionsResponse,
    CompletionsUsage, TEXT_COMPLETION_OBJECT,
};
use super::error::OpenAIError;
use super::util::{build_sampling_eos, contains_any, error_chunk, new_id_suffix, unix_now};

/// Serve one already-validated completions request on a V3 runtime.
/// The caller (the shared `/v1/completions` handler) has done all
/// request validation; this function only generates and shapes.
#[allow(clippy::too_many_arguments)]
pub(super) async fn respond(
    model: Arc<V3Model>,
    prompts: Vec<String>,
    max_tokens: usize,
    sampling_params: super::util::SamplingParams,
    stop_strings: Vec<String>,
    echo: bool,
    stream: bool,
    model_id: String,
    timeout: std::time::Duration,
) -> Result<Response, OpenAIError> {
    if stream {
        // Validation mirroring the V2 stream contract happened in the
        // shared handler; one prompt is guaranteed here.
        let prompt = prompts.into_iter().next().expect("validated non-empty");
        return Ok(stream_v3_completions(
            model,
            prompt,
            max_tokens,
            sampling_params,
            stop_strings,
            model_id,
        )
        .into_response());
    }

    let handle = tokio::task::spawn_blocking(move || -> Result<_, ServerError> {
        run_v3_completions_loop(
            &model,
            &prompts,
            max_tokens,
            sampling_params,
            &stop_strings,
            echo,
        )
    });
    let (choices, prompt_tokens, completion_tokens) = join_generation(handle, timeout).await?;

    Ok(Json(CompletionsResponse {
        id: format!("cmpl-{}", new_id_suffix()),
        object: TEXT_COMPLETION_OBJECT,
        created: unix_now(),
        model: model_id,
        choices,
        usage: CompletionsUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
    .into_response())
}

/// Await a blocking generation task under the server-side timeout —
/// the same 504-and-detach contract the V2 buffered path applies
/// (BUG-infer-deadlock §5.6): on timeout the JoinHandle is dropped,
/// the blocking thread finishes in the background, and the client
/// gets 504 rather than an indefinitely held connection.
async fn join_generation<T>(
    handle: tokio::task::JoinHandle<Result<T, ServerError>>,
    timeout: std::time::Duration,
) -> Result<T, OpenAIError> {
    if timeout.is_zero() {
        return Ok(handle
            .await
            .map_err(|e| ServerError::Internal(e.to_string()))??);
    }
    match tokio::time::timeout(timeout, handle).await {
        Ok(join_result) => Ok(join_result.map_err(|e| ServerError::Internal(e.to_string()))??),
        Err(_elapsed) => {
            tracing::warn!(
                target: "larql_server::openai::completions",
                "v3 completion timed out after {}s; responding 504",
                timeout.as_secs(),
            );
            Err(OpenAIError::from(ServerError::Timeout(format!(
                "completion exceeded server-side timeout of {}s",
                timeout.as_secs(),
            ))))
        }
    }
}

/// Buffered generation over every prompt; returns
/// `(choices, prompt_tokens_sum, completion_tokens_sum)` — the same
/// shape the V2 loop feeds the shared response struct.
fn run_v3_completions_loop(
    model: &V3Model,
    prompts: &[String],
    max_tokens: usize,
    sampling_params: super::util::SamplingParams,
    stop_strings: &[String],
    echo: bool,
) -> Result<(Vec<CompletionChoice>, usize, usize), ServerError> {
    let mut choices = Vec::with_capacity(prompts.len());
    let mut prompt_tokens_sum = 0;
    let mut completion_tokens_sum = 0;
    for (index, prompt) in prompts.iter().enumerate() {
        let prompt_ids = encode_prompt(model, prompt)?;
        let (sampling, eos) = build_sampling_eos(sampling_params, stop_strings);
        let generation = generate_v3(model, &prompt_ids, max_tokens, sampling, &eos, |_, _| {})?;

        // The V2 path's text-level EOS/stop-trim semantics, verbatim —
        // `finalize_completion` is shared, so the two runtimes agree
        // on finish_reason and trimming by construction.
        let scored: Vec<(String, f64)> =
            generation.texts.iter().map(|t| (t.clone(), 1.0)).collect();
        let (mut text, kept, mut finish_reason) = finalize_completion(&scored, stop_strings);
        if generation.stopped_early {
            finish_reason = "stop";
        }
        if echo {
            text = format!("{prompt}{text}");
        }
        prompt_tokens_sum += generation.prompt_tokens;
        completion_tokens_sum += kept.len();
        choices.push(CompletionChoice {
            text,
            index,
            finish_reason,
            logprobs: None,
        });
    }
    Ok((choices, prompt_tokens_sum, completion_tokens_sum))
}

/// SSE streaming over the V3 stack — chunk shape, stop handling, and
/// termination identical to the V2 `stream_completions`.
fn stream_v3_completions(
    model: Arc<V3Model>,
    prompt: String,
    max_tokens: usize,
    sampling_params: super::util::SamplingParams,
    stop_strings: Vec<String>,
    model_id: String,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
    let cmpl_id = format!("cmpl-{}", new_id_suffix());

    tokio::task::spawn_blocking(move || {
        let prompt_ids = match encode_prompt(&model, &prompt) {
            Ok(ids) => ids,
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&e.to_string()));
                return;
            }
        };
        let (sampling, eos) = build_sampling_eos(sampling_params, &stop_strings);

        let cmpl_id_cb = cmpl_id.clone();
        let model_id_cb = model_id.clone();
        let tx_cb = tx.clone();
        let stop_strings_cb = stop_strings.clone();
        let mut completion_text = String::new();
        let mut early_stop = false;
        let result = generate_v3(
            &model,
            &prompt_ids,
            max_tokens,
            sampling,
            &eos,
            |_id, text| {
                if early_stop {
                    return;
                }
                let chunk =
                    build_text_completion_chunk(&cmpl_id_cb, &model_id_cb, Some(text), None);
                if tx_cb.blocking_send(chunk).is_err() {
                    early_stop = true;
                    return;
                }
                completion_text.push_str(text);
                if !stop_strings_cb.is_empty() && contains_any(&completion_text, &stop_strings_cb) {
                    early_stop = true;
                }
            },
        );
        let finish_reason: &'static str = match result {
            Ok(generation) if early_stop || generation.stopped_early => "stop",
            Ok(_) => "length",
            Err(e) => {
                let _ = tx.blocking_send(error_chunk(&e.to_string()));
                return;
            }
        };
        let final_chunk =
            build_text_completion_chunk(&cmpl_id, &model_id, None, Some(finish_reason));
        let _ = tx.blocking_send(final_chunk);
    });

    let stream = ReceiverStream::new(rx)
        .map(|data| Event::default().data(data))
        .chain(tokio_stream::once(Event::default().data("[DONE]")))
        .map(Ok::<_, Infallible>);

    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn encode_prompt(model: &V3Model, prompt: &str) -> Result<Vec<u32>, ServerError> {
    let encoding = model
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| ServerError::BadRequest(format!("tokenize: {e}")))?;
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    if ids.is_empty() {
        return Err(ServerError::BadRequest(
            "prompt tokenises to empty".to_string(),
        ));
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn join_generation_awaits_with_timeout_disabled() {
        let handle = tokio::task::spawn_blocking(|| Ok::<_, ServerError>(7usize));
        let value = join_generation(handle, std::time::Duration::ZERO)
            .await
            .unwrap();
        assert_eq!(value, 7);
    }

    #[tokio::test]
    async fn join_generation_returns_within_the_timeout() {
        let handle = tokio::task::spawn_blocking(|| Ok::<_, ServerError>("done"));
        let value = join_generation(handle, std::time::Duration::from_secs(30))
            .await
            .unwrap();
        assert_eq!(value, "done");
    }

    #[tokio::test]
    async fn join_generation_responds_504_when_the_task_overruns() {
        let handle = tokio::task::spawn_blocking(|| {
            std::thread::sleep(std::time::Duration::from_millis(300));
            Ok::<_, ServerError>(())
        });
        let err = join_generation(handle, std::time::Duration::from_millis(5))
            .await
            .unwrap_err();
        let body = format!("{err:?}");
        assert!(body.contains("timeout"), "{body}");
    }

    #[tokio::test]
    async fn join_generation_surfaces_the_task_error() {
        let handle =
            tokio::task::spawn_blocking(|| Err::<(), _>(ServerError::Internal("boom".to_string())));
        let err = join_generation(handle, std::time::Duration::from_secs(30))
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("boom"));
    }
}
