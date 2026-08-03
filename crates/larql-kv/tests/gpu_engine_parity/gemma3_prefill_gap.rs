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
/// - **Not a production defect.** The same comparison on a real Gemma 3
///   4B Q4K vindex agrees to 4.0e-7 across prompt lengths 1-20 with zero
///   argmax mismatches (`examples/gemma_prefill_parity.rs`), and
///   `larql run` emits identical text on both backends. Whatever this
///   is, it does not reach a real checkpoint.
/// - **Arch-specific.** The identical comparison on the SWA-free
///   `tinymodel` fixture — same dims, same prompts, same code — agrees
///   to 1.6e-7 at every length.
/// - **Shape-gated, not window-gated.** Giving the Gemma fixture a real
///   512 window (`make_test_q4k_weights_rope_scaled`) diverges just as
///   far (5.5e-1), so the `sliding_window = 0` sentinel is not the
///   cause. The surviving difference is dimensional: this fixture is
///   `head_dim = 64, hidden = 256`, the real model is
///   `head_dim = 256, hidden = 2560`.
/// - **Batched-prefill-specific.** `standard` reaches Metal through
///   `coarse_prefill` → `fused_prefill` (one batched pass) and is the
///   only engine that diverges. `markov-rs` and friends go through
///   `coarse_prefill_with_state`, which drives the same model
///   token-by-token, and match the CPU to ~1e-7. Metal disagrees with
///   *itself*, which rules out a CPU-side reference error.
/// - **Intra-layer.** A single-layer model already diverges (3.4e-1),
///   and the error compounds with depth (2 layers 4.2e-1, 3 layers
///   4.9e-1) rather than originating across layers.
/// - **Position-dependent.** A single-token prefill agrees; the gap
///   appears as soon as there are two positions to relate.
///
/// Putting those together: a Gemma-specific stage in Metal's batched
/// prefill — QK-norm and the post-attention / post-FFN norms are what
/// this fixture has and `tinymodel` does not — carries a shape
/// assumption that `head_dim = 64` violates and `head_dim = 256`
/// satisfies. The defect is that the kernel returns wrong data on such a
/// shape instead of declining it, which is a robustness bug rather than
/// a correctness one for shipped models.
///
/// The practical consequence, and the reason this stays recorded: the
/// synthetic Gemma-3 fixture **cannot be used for cross-backend numeric
/// parity**. [`super::numeric`] uses `tinymodel` for that and keeps the
/// Gemma fixture for the structural assertions only.

#[test]
#[ignore = "records a Metal batched-prefill shape bug that does NOT reach real models; run with --ignored to measure"]
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
