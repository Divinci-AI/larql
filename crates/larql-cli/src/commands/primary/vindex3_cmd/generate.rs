//! `larql vindex3 exec --generate N` — greedy autoregressive decode
//! from the container's own program.
//!
//! Runs on a [`DecodeSession`]: every operand is loaded once (in the
//! backend's declared weight format, so a device buffer cache can keep
//! the model resident) and each token advances one position against the
//! session's KV cache. The phases are timed separately — weight load,
//! prompt ingestion, first generated token, steady decode — because
//! they are different costs and conflating them is how a decode number
//! lies.
//!
//! Sampling is greedy argmax on purpose: generation doubles as a
//! fixture (same ids in → same ids out per backend), and a sampler
//! would put a source of randomness between two runs of a parity
//! comparison. Token ids go in and come out as ids — a tokenizer is
//! part of the fixture and lives outside this binary.

use std::time::Instant;

use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

/// The steady-state window is the tail half of the decode steps — after
/// the page cache and device buffer pools have warmed on the early ones.
const STEADY_TAIL_DIVISOR: usize = 2;

/// Greedy decode: ingest the prompt one position at a time, then append
/// the argmax of each step's logits.
pub(super) fn run_generate<B: PlanBackend>(
    backend: &B,
    engine: &str,
    prompt: &[u32],
    new_tokens: usize,
    plan: &ComponentOpPlan,
    store: &OperandStore,
) -> Result<(), Box<dyn std::error::Error>> {
    let loading = Instant::now();
    let mut session = DecodeSession::new(plan, store, backend)?;
    let load_seconds = loading.elapsed().as_secs_f64();
    eprintln!("weights resident in {load_seconds:.1} s");

    // Prompt ingestion: every position must pass through the stack to
    // fill the KV cache; only the last position's logits are consumed.
    let prompt_started = Instant::now();
    let mut logits = None;
    for &token in prompt {
        logits = session.step(token)?.logits;
    }
    let prompt_seconds = prompt_started.elapsed().as_secs_f64();
    let logits = logits.ok_or("plan carries no output head — cannot generate")?;
    let (mut next, mut value) = argmax(&logits).ok_or("output head produced no logits")?;

    let mut ids = prompt.to_vec();
    let mut step_seconds = Vec::with_capacity(new_tokens);
    for step in 0..new_tokens {
        ids.push(next as u32);
        eprintln!(
            "token {:>3}/{new_tokens}  id {next:<8} ({value:+.3})  context {}",
            step + 1,
            ids.len(),
        );
        if step + 1 == new_tokens {
            break;
        }
        let started = Instant::now();
        let logits = session
            .step(next as u32)?
            .logits
            .ok_or("plan carries no output head — cannot generate")?;
        (next, value) = argmax(&logits).ok_or("output head produced no logits")?;
        step_seconds.push(started.elapsed().as_secs_f64());
    }

    println!("engine: {engine}");
    println!("prompt tokens: {}", prompt.len());
    println!("generated ids: {}", join_ids(&ids[prompt.len()..]));
    println!("sequence ids: {}", join_ids(&ids));
    println!("weights loaded: {load_seconds:.1} s");
    println!(
        "prompt: {} tokens in {prompt_seconds:.1} s ({:.0} ms/token) — first new token ready",
        prompt.len(),
        prompt_seconds * 1e3 / prompt.len().max(1) as f64,
    );
    if let Some(report) = DecodeReport::from_steps(&step_seconds) {
        println!("decode tokens: {}", report.decode_tokens);
        println!("decode elapsed: {:.1} s", report.decode_seconds);
        println!(
            "mean: {:.0} ms/token ({:.3} tok/s)",
            report.mean_seconds_per_token * 1e3,
            report.mean_seconds_per_token.recip(),
        );
        println!(
            "steady (last half): {:.0} ms/token ({:.3} tok/s)",
            report.steady_seconds_per_token * 1e3,
            report.steady_seconds_per_token.recip(),
        );
        // Split the token between device dispatch and everything else.
        // "Everything else" is the interpreter's elementwise glue —
        // norms, RoPE, softmax over the KV cache, activations,
        // residuals — which is a fixed per-token cost just as
        // submission is, and which a bytes-vs-time fit cannot separate
        // from it.
        if let Some(stats) = backend.dispatch_stats() {
            let device_s = stats.device_nanos as f64 / 1e9;
            let per_token = device_s / (report.decode_tokens + prompt.len()) as f64;
            println!(
                "device: {:.0} ms/token in {} submissions/token ({:.0} us each)",
                per_token * 1e3,
                stats.submissions / (report.decode_tokens + prompt.len()) as u64,
                per_token * 1e6
                    / (stats.submissions as f64 / (report.decode_tokens + prompt.len()) as f64),
            );
            println!(
                "glue:   {:.0} ms/token (everything not inside a device call)",
                (report.mean_seconds_per_token - per_token) * 1e3,
            );
        }
    }
    Ok(())
}

/// Index and value of the largest logit; ties keep the first, matching
/// the summary path's fold.
pub(super) fn argmax(logits: &[f32]) -> Option<(usize, f32)> {
    logits
        .iter()
        .enumerate()
        .fold(None, |best, (index, &value)| match best {
            Some((_, best_value)) if value <= best_value => best,
            _ => Some((index, value)),
        })
}

/// Steady-decode timing over the per-step seconds (prompt ingestion and
/// weight load are reported separately by the caller).
#[derive(Debug, PartialEq)]
pub(super) struct DecodeReport {
    pub(super) decode_tokens: usize,
    pub(super) decode_seconds: f64,
    pub(super) mean_seconds_per_token: f64,
    pub(super) steady_seconds_per_token: f64,
}

impl DecodeReport {
    /// `None` when no decode step beyond the first token ran — a single
    /// forward has no decode rate to report.
    pub(super) fn from_steps(step_seconds: &[f64]) -> Option<Self> {
        if step_seconds.is_empty() {
            return None;
        }
        let decode_seconds: f64 = step_seconds.iter().sum();
        let steady_len = (step_seconds.len() / STEADY_TAIL_DIVISOR).max(1);
        let steady = &step_seconds[step_seconds.len() - steady_len..];
        Some(Self {
            decode_tokens: step_seconds.len(),
            decode_seconds,
            mean_seconds_per_token: decode_seconds / step_seconds.len() as f64,
            steady_seconds_per_token: steady.iter().sum::<f64>() / steady.len() as f64,
        })
    }
}

/// Comma-separated ids, the same shape `--tokens` accepts, so a run's
/// output can be fed straight back in as a prompt.
fn join_ids(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
}
