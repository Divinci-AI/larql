//! The one place the two backends genuinely disagree today.
//!
//! **This is not the sliding-window gap.** It was first attributed to
//! per-layer SWA, and that attribution was wrong: on this fixture
//! `effective_attention_window_for_layer` resolves to `None` for every
//! layer on *both* backends (the arch declares its layers sliding but
//! supplies no width), so no window is applied on either side and the
//! divergence is unchanged by the CPU SWA implementation. Recorded here
//! so the next reader doesn't re-derive the same dead end.

use larql_inference::ffn::NullFfn;
use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

use super::support::{build, relative_l2};

/// Longest prompt swept by the reproducer.
const MAX_PROMPT_LEN: usize = 6;

/// Metal's **batched** prefill disagrees with everything else on the
/// Gemma-3 fixture. **Ignored** because it documents an open defect
/// rather than guarding a fixed behaviour; run with `--ignored` to
/// measure the current size.
///
/// ```text
/// prompt len 1  → relative L2 9.9e-7   (agrees)
/// prompt len 2  → relative L2 4.4e-1
/// prompt len 3  → relative L2 2.3e-1
/// prompt len 4  → relative L2 4.2e-1
/// prompt len 6  → relative L2 4.1e-1
/// ```
///
/// What is established:
///
/// - **Arch-specific, not shape-specific.** The identical comparison on
///   the SWA-free `tinymodel` fixture — same dims, same prompts, same
///   code — agrees to 1.6e-7 at every length.
/// - **Batched-prefill-specific.** `standard` reaches Metal through
///   `coarse_prefill` → `fused_prefill` (one batched pass) and is the
///   only engine that diverges. `markov-rs` and friends go through
///   `coarse_prefill_with_state`, which drives the same model
///   token-by-token, and match the CPU to ~1e-7. So Metal disagrees with
///   *itself*, which rules out a CPU-side reference error.
/// - **Position-dependent.** A single-token prefill agrees; the gap
///   appears as soon as there are two positions to relate.
/// - **Not reproducing in production.** `larql run` on qwen3-0.6b
///   (non-Gemma) emits byte-identical text on both backends across 24
///   tokens, and Gemma 3 4B generates coherent text on Metal.
///
/// What is NOT established: the cause. The remaining candidates are the
/// Gemma-3-specific stages a batched prefill kernel handles differently
/// from a per-token path — post-attention / post-FFN norms, QK-norm, and
/// the GeluTanh activation. Narrowing it needs a per-layer state dump
/// from the batched kernel, which `fused_prefill` does not currently
/// expose (only `coarse_prefill_with_state` does, and that is the path
/// that already agrees).
///
/// Until then, cross-backend numeric parity is asserted on the SWA-free
/// fixture (see [`super::numeric`]) and a real Gemma-3 Q4K vindex is the
/// missing piece for judging production impact.
#[test]
#[ignore = "open defect: Metal batched prefill vs CPU on Gemma-3 arch; run with --ignored to measure"]
fn metal_batched_prefill_diverges_on_gemma3_arch() {
    let weights = make_test_q4k_weights();
    let index = make_test_q4k_vindex(&weights);

    // Pin the premise this reproducer rests on: no window is in play, so
    // whatever this is, it is not the sliding-window path.
    assert!(
        weights.arch.is_sliding_window_layer(0),
        "fixture no longer declares sliding layers — re-check the premise"
    );
    assert_eq!(
        weights.arch.sliding_window_size(),
        None,
        "fixture now declares a window width; this reproducer assumed none, \
         and the SWA path would now be a live variable here"
    );

    for n in 1..=MAX_PROMPT_LEN {
        let prompt: Vec<u32> = (0..n as u32).collect();
        let mut gpu = build("standard", true);
        let mut cpu = build("standard", false);
        let hg = gpu
            .prefill_quant(
                &weights,
                &NullFfn,
                &index,
                &prompt,
                &*larql_compute::default_backend(),
            )
            .expect("gpu prefill");
        let hc = cpu
            .prefill_quant(
                &weights,
                &NullFfn,
                &index,
                &prompt,
                &*larql_compute::cpu_backend(),
            )
            .expect("cpu prefill");
        eprintln!("prompt len {n}: relative L2 {:.4e}", relative_l2(&hg, &hc));
    }
}
