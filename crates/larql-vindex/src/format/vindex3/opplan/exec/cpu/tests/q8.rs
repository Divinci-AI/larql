//! CPU-3A: does a fused Q8 consumer beat the BF16 one, and does it
//! compute what the format denotes?
//!
//! Two separate questions, deliberately not mixed. Whether Q8 is a good
//! enough REPRESENTATION of a checkpoint is CPU-3B and needs logits, a
//! trajectory, continuation state and PARITY-FLOOR-1. Nothing here
//! touches that. Here the format is a given and the only claims are
//! mechanical: the kernel realises `code * scale` faithfully, and it is
//! or is not faster.
//!
//! The bench reports **time per matrix** rather than GB/s, because GB/s
//! is the metric that makes a good Q8 kernel look bad: half the bytes at
//! a lower rate is still less time, and a rate comparison would hide
//! that. Both byte rates are printed — stored bytes and the f32 the
//! weights DENOTE — so neither can be quoted without the other.

use super::super::executor::CpuExecutor;
use super::super::kernels::{q8_block_dot_portable, q8_dot, FusedBf16, FusedQ8};
use super::super::projector::{DenseProjector, WeightRows};
use crate::format::vindex3::fixtures::lcg_values;

/// Elements per scale. Every real Qwen3.8 `in_dim` (5120, 6144, 17408) is
/// a multiple of it, so the model itself never exercises a tail — which
/// is exactly why the tail has its own test.
const BLOCK: usize = 64;

/// Symmetric per-block int8: `q = round(w / scale)`, `scale = max|w| /
/// 127`.
///
/// The simplest quantiser that can be stated in one line, chosen because
/// CPU-3A is about the KERNEL. Choosing a good quantiser is CPU-3B's
/// problem and it will be measured on logits, not here.
fn quantise(weights: &[f32], in_dim: usize, block: usize) -> (Vec<i8>, Vec<f32>) {
    let per_row = in_dim.div_ceil(block);
    let rows = weights.len() / in_dim;
    let mut codes = vec![0i8; weights.len()];
    let mut scales = vec![0.0f32; rows * per_row];
    for r in 0..rows {
        for b in 0..per_row {
            let lo = r * in_dim + b * block;
            let hi = (lo + block).min((r + 1) * in_dim);
            let peak = weights[lo..hi].iter().fold(0.0f32, |m, w| m.max(w.abs()));
            let scale = if peak > 0.0 { peak / 127.0 } else { 1.0 };
            scales[r * per_row + b] = scale;
            for i in lo..hi {
                codes[i] = (weights[i] / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (codes, scales)
}

/// **The kernel realises the format.**
///
/// Against a scalar definition of `sum over blocks of scale * sum(code *
/// x)`, not against the original f32 weights: the quantiser's error is
/// the FORMAT's, and folding it in here would let a broken kernel hide
/// inside a tolerance chosen for quantisation noise.
#[test]
fn the_q8_kernel_computes_what_the_format_denotes() {
    const OUT: usize = 9;
    for in_dim in [BLOCK, BLOCK * 3, 5120] {
        let w = lcg_values(OUT * in_dim, 5);
        let (codes, scales) = quantise(&w, in_dim, BLOCK);
        let x = lcg_values(in_dim, 6);
        let mut got = vec![0.0f32; OUT];
        FusedQ8.project_rows(
            WeightRows::Q8 {
                codes: &codes,
                scales: &scales,
                block: BLOCK,
            },
            &x,
            &mut got,
        );
        let per_row = in_dim.div_ceil(BLOCK);
        for (o, value) in got.iter().enumerate() {
            let mut want = 0.0f32;
            for b in 0..per_row {
                let lo = b * BLOCK;
                let hi = (lo + BLOCK).min(in_dim);
                want += scales[o * per_row + b]
                    * q8_block_dot_portable(&codes[o * in_dim + lo..o * in_dim + hi], &x[lo..hi]);
            }
            let tol = want.abs() * 1e-5 + 1e-4;
            assert!(
                (value - want).abs() <= tol,
                "row {o} at in_dim {in_dim}: {value} against the format's {want}"
            );
        }
    }
}

/// A block that does not divide `in_dim` is handled, and the tail is not
/// silently dropped.
///
/// No real Qwen3.8 shape has a tail — 5120, 6144 and 17408 are all
/// multiples of 64 — so the model could not catch a kernel that walked
/// whole blocks and stopped. The awkward shape is the instrument.
#[test]
fn a_ragged_final_block_is_not_dropped() {
    let in_dim = BLOCK * 2 + 5;
    let w = lcg_values(in_dim, 7);
    let (codes, scales) = quantise(&w, in_dim, BLOCK);
    assert_eq!(scales.len(), 3, "the tail must get its own scale");
    let x = lcg_values(in_dim, 8);

    let full = q8_dot(&codes, &scales, BLOCK, &x);
    // The same dot with the tail's codes zeroed: if the kernel ignored
    // the ragged block these would agree, and the test would be asserting
    // nothing at all.
    let mut truncated = codes.clone();
    for c in truncated[BLOCK * 2..].iter_mut() {
        *c = 0;
    }
    let without = q8_dot(&truncated, &scales, BLOCK, &x);
    assert!(
        (full - without).abs() > 1e-6,
        "the ragged tail contributed nothing, so the kernel is walking whole blocks only"
    );
}

/// The NEON block dot and the portable one agree.
///
/// On aarch64 the portable version is dead code, so the claim that it is
/// "the definition" would otherwise be tested by shipping x86 a wrong
/// answer.
#[test]
fn the_portable_and_neon_block_dots_agree() {
    for len in [1usize, 7, 16, 17, 64, 65] {
        let codes: Vec<i8> = (0..len).map(|i| ((i * 37) % 255) as i8).collect();
        let x = lcg_values(len, 21);
        let portable = q8_block_dot_portable(&codes, &x);
        let mut got = vec![0.0f32; 1];
        FusedQ8.project_rows(
            WeightRows::Q8 {
                codes: &codes,
                scales: &[1.0],
                block: len.max(1),
            },
            &x,
            &mut got,
        );
        let magnitude: f32 = codes
            .iter()
            .zip(&x)
            .map(|(c, v)| (*c as f32 * v).abs())
            .sum();
        assert!(
            (got[0] - portable).abs() <= 1e-6 * magnitude.max(1.0),
            "len {len}: {} against {portable}",
            got[0]
        );
    }
}

/// Slicing rows must cut the scales with the codes.
///
/// The executor partitions a projection across workers by ROWS. A cut
/// that moved the codes and not the scales would hand a worker the right
/// weights under a different row's scale — finite, plausible, wrong — and
/// only on multi-worker shapes.
#[test]
fn slicing_rows_cuts_the_scales_too() {
    const OUT: usize = 8;
    let in_dim = BLOCK * 2;
    let w = lcg_values(OUT * in_dim, 12);
    let (codes, scales) = quantise(&w, in_dim, BLOCK);
    let rows = WeightRows::Q8 {
        codes: &codes,
        scales: &scales,
        block: BLOCK,
    };
    let x = lcg_values(in_dim, 13);

    let mut whole = vec![0.0f32; OUT];
    FusedQ8.project_rows(rows, &x, &mut whole);
    for (start, want) in whole.iter().enumerate() {
        let cut = rows.slice_rows(in_dim, start, 1);
        assert_eq!(cut.rows(in_dim), 1);
        let mut one = vec![0.0f32; 1];
        FusedQ8.project_rows(cut, &x, &mut one);
        assert_eq!(
            one[0], *want,
            "row {start} changed value when sliced out — the scales did not travel with it"
        );
    }
}

/// **The comparison.** BF16 against Q8 on the shapes a token runs.
///
/// Measured through `CpuExecutor` with the SHIPPED kernels, in the same
/// binary and the same harness whose BF16 arm reproduces the model's
/// projection cost to -3.9% (`projection_cost`). A ratio from any other
/// harness would not license a claim about LARQL — CPU-2D spent a rung
/// learning that.
///
/// ```text
/// QW_Q8_BENCH=1 cargo test --release exec::cpu::tests::q8 -- --nocapture
/// ```
#[test]
fn bf16_against_q8_on_the_real_shapes() {
    if std::env::var("QW_Q8_BENCH").is_err() {
        eprintln!("SKIP bf16_against_q8: set QW_Q8_BENCH=1");
        return;
    }
    use std::time::Instant;
    let exec = CpuExecutor::new().unwrap();
    println!(
        "\n  BF16 against Q8 (block {BLOCK}), {} workers — TIME per matrix,\n  \
         because half the bytes at a lower rate is still less time.\n",
        exec.workers()
    );
    println!(
        "  {:<22} {:>6} {:>9} {:>9} {:>7} {:>10} {:>10}",
        "projection", "calls", "bf16 ms", "q8 ms", "speedup", "bf16 MB", "q8 MB"
    );

    let (mut bf_ms, mut q8_ms, mut bf_gb, mut q8_gb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (name, out_dim, in_dim, calls) in super::projection_cost::COMPACT.iter().copied() {
        let f32w = lcg_values(out_dim * in_dim, 11);
        let bf: Vec<u16> = f32w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let (codes, scales) = quantise(&f32w, in_dim, BLOCK);
        let x = lcg_values(in_dim, 22);
        let q8_rows = WeightRows::Q8 {
            codes: &codes,
            scales: &scales,
            block: BLOCK,
        };
        let iters = (3_000_000_000.0 / (out_dim * in_dim) as f64).clamp(3.0, 100.0) as usize;

        let mut sink = 0.0f32;
        let mut time_it = |f: &mut dyn FnMut(&mut f32)| {
            f(&mut sink);
            let t = Instant::now();
            for _ in 0..iters {
                f(&mut sink);
            }
            t.elapsed().as_secs_f64() / iters as f64
        };
        let bf_each = time_it(&mut |s: &mut f32| {
            *s += exec.project(&FusedBf16, WeightRows::Bf16(&bf), &x, out_dim)[0]
        });
        let q8_each =
            time_it(&mut |s: &mut f32| *s += exec.project(&FusedQ8, q8_rows, &x, out_dim)[0]);
        std::hint::black_box(sink);

        let bf_bytes = (out_dim * in_dim * 2) as f64;
        let q8_bytes = q8_rows.bytes() as f64;
        bf_ms += bf_each * calls as f64 * 1e3;
        q8_ms += q8_each * calls as f64 * 1e3;
        bf_gb += bf_bytes * calls as f64 / 1e9;
        q8_gb += q8_bytes * calls as f64 / 1e9;
        println!(
            "  {name:<22} {calls:>6} {:>9.3} {:>9.3} {:>6.2}x {:>10.1} {:>10.1}",
            bf_each * 1e3,
            q8_each * 1e3,
            bf_each / q8_each,
            bf_bytes / 1e6,
            q8_bytes / 1e6,
        );
    }

    println!("  {:-<80}", "");
    println!(
        "  {:<22} {:>16.2} {:>9.2} {:>6.2}x",
        "ms/token",
        bf_ms,
        q8_ms,
        bf_ms / q8_ms
    );
    println!(
        "  {:<22} {:>16.2} {:>9.2}   GB/token stored",
        "traffic", bf_gb, q8_gb
    );
    println!(
        "  {:<22} {:>16.1} {:>9.1}   GB/s stored   ({:.1} vs {:.1} GB/s of DENOTED f32)",
        "rate",
        bf_gb / (bf_ms / 1e3),
        q8_gb / (q8_ms / 1e3),
        bf_gb * 2.0 / (bf_ms / 1e3),
        bf_gb * 2.0 / (q8_ms / 1e3),
    );
    println!(
        "\n  bits/weight: bf16 16.00, q8 {:.2} (scales included)\n",
        q8_gb / bf_gb * 16.0
    );
}
