//! The three kernels must agree, and the executor's partitioning must
//! not change an answer.

use super::super::executor::CpuExecutor;
use super::super::projector::{CpuParallelism, DenseProjector, WeightRows};
use super::{BlasF32, FusedBf16, ScalarF32};
use crate::format::vindex3::fixtures::lcg_values;

fn narrow(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}
fn widen(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn metrics(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut num, mut den, mut mx) = (0.0f64, 0.0f64, 0.0f64);
    for (p, q) in a.iter().zip(b) {
        num += (*p as f64 - *q as f64).powi(2);
        den += (*q as f64).powi(2);
        mx = mx.max((*p as f64 - *q as f64).abs());
    }
    ((num / den.max(f64::MIN_POSITIVE)).sqrt(), mx)
}

/// The BF16 widen is EXACT — it is a bit-shift, not a conversion.
///
/// This is what lets the fused kernel claim it changes representation and
/// mechanics and no numerical value: every stored code unit denotes
/// exactly the f32 the scalar path would have multiplied.
#[test]
fn the_bf16_widen_is_exact_not_a_conversion() {
    for v in [0.0f32, 1.0, -1.0, 1e-30, 1e30, 0.1, -12345.678] {
        let round = widen(narrow(v));
        assert_eq!(
            round.to_bits() & 0xffff_0000,
            v.to_bits() & 0xffff_0000,
            "widen(narrow({v})) lost the top half"
        );
        assert_eq!(round, widen(narrow(round)), "not idempotent at {v}");
    }
}

/// All three kernels compute the same projection, to summation order.
#[test]
fn the_kernels_agree_on_the_same_projection() {
    let (out_dim, in_dim) = (257usize, 320usize); // deliberately not round
    let f32w: Vec<f32> = lcg_values(out_dim * in_dim, 7)
        .iter()
        .map(|v| widen(narrow(*v)))
        .collect();
    let bf: Vec<u16> = f32w.iter().map(|v| narrow(*v)).collect();
    let x = lcg_values(in_dim, 9);

    let exec = CpuExecutor::new().unwrap();
    let scalar = exec.project(&ScalarF32, WeightRows::F32(&f32w), &x, out_dim);
    let blas = exec.project(&BlasF32, WeightRows::F32(&f32w), &x, out_dim);
    let fused = exec.project(&FusedBf16, WeightRows::Bf16(&bf), &x, out_dim);

    let (rel_b, _) = metrics(&blas, &scalar);
    let (rel_f, _) = metrics(&fused, &scalar);
    assert!(rel_b < 1e-5, "blas vs scalar rel_rms {rel_b:e}");
    assert!(
        rel_f < 1e-5,
        "fused bf16 vs scalar rel_rms {rel_f:e} — the widen should introduce nothing \
         beyond summation order"
    );
}

/// **Partitioning must not change an answer.**
///
/// The executor is free to cut rows however it likes; a kernel that read
/// outside its slab, or an executor that mis-sliced one, would show up
/// here and nowhere else. Row counts chosen so the last partition is
/// short.
#[test]
fn the_row_partition_does_not_change_the_result() {
    let (out_dim, in_dim) = (1001usize, 128usize);
    let f32w: Vec<f32> = lcg_values(out_dim * in_dim, 3)
        .iter()
        .map(|v| widen(narrow(*v)))
        .collect();
    let bf: Vec<u16> = f32w.iter().map(|v| narrow(*v)).collect();
    let x = lcg_values(in_dim, 4);

    let exec = CpuExecutor::new().unwrap();
    // One call over everything — the definition.
    let mut whole = vec![0.0f32; out_dim];
    FusedBf16.project_rows(WeightRows::Bf16(&bf), &x, &mut whole);

    for workers in [1usize, 2, 3, 5, 8, 13] {
        let rows = out_dim.div_ceil(workers);
        let mut split = vec![0.0f32; out_dim];
        for (i, slot) in split.chunks_mut(rows).enumerate() {
            let slab = WeightRows::Bf16(&bf).slice_rows(in_dim, i * rows, slot.len());
            FusedBf16.project_rows(slab, &x, slot);
        }
        assert_eq!(
            split, whole,
            "{workers}-way partition changed the result — row slabs are independent \
             and must be bit-identical however they are cut"
        );
    }
    // And through the executor's own policy.
    let via = exec.project(&FusedBf16, WeightRows::Bf16(&bf), &x, out_dim);
    assert_eq!(via, whole);
}

/// Each kernel states who threads it, and the executor honours it.
///
/// The rule this file exists to protect: at most one layer of parallelism
/// owns the machine. A `LibraryOwned` kernel that the executor also
/// partitioned would nest Accelerate's threads inside Rayon's.
#[test]
fn threading_ownership_is_declared_not_assumed() {
    assert_eq!(ScalarF32.parallelism(), CpuParallelism::Serial);
    assert_eq!(BlasF32.parallelism(), CpuParallelism::LibraryOwned);
    assert_eq!(FusedBf16.parallelism(), CpuParallelism::ExternalPool);

    let exec = CpuExecutor::new().unwrap();
    assert!(exec.workers() >= 1);
}

/// The executor's policy, measured on the real shapes.
///
/// Env-gated. Confirms the seam reproduces CPU-1B's hand-rolled numbers
/// rather than losing them to the abstraction — an executor whose
/// dispatch cost ate the win would be worse than no seam at all.
///
/// ```text
/// QW_CPU_EXEC_BENCH=1 cargo test --release exec::cpu -- --nocapture
/// ```
#[test]
fn executor_policy_bench() {
    if std::env::var("QW_CPU_EXEC_BENCH").is_err() {
        eprintln!("SKIP executor_policy_bench: set QW_CPU_EXEC_BENCH=1");
        return;
    }
    use std::time::Instant;
    let exec = CpuExecutor::new().unwrap();
    println!("\n  executor workers: {}\n", exec.workers());
    println!(
        "  {:22} {:>10} {:>10} {:>10}   bytes read",
        "shape", "blas f32", "fused bf16", "speedup"
    );
    for (label, out_dim, in_dim) in [
        ("delta in_proj_qkv", 10240usize, 5120usize),
        ("delta out_proj", 5120, 6144),
        ("ffn gate/up", 17408, 5120),
        ("delta in_proj_a (tiny)", 48, 5120),
    ] {
        let f32w: Vec<f32> = lcg_values(out_dim * in_dim, 11)
            .iter()
            .map(|v| widen(narrow(*v)))
            .collect();
        let bf: Vec<u16> = f32w.iter().map(|v| narrow(*v)).collect();
        let x = lcg_values(in_dim, 22);
        let iters = (1_000_000_000.0 / (out_dim * in_dim) as f64).clamp(3.0, 200.0) as usize;

        let mut sink = 0.0f32;
        let t = Instant::now();
        for _ in 0..iters {
            sink += exec.project(&BlasF32, WeightRows::F32(&f32w), &x, out_dim)[0];
        }
        let b = t.elapsed().as_secs_f64() / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            sink += exec.project(&FusedBf16, WeightRows::Bf16(&bf), &x, out_dim)[0];
        }
        let f = t.elapsed().as_secs_f64() / iters as f64;
        std::hint::black_box(sink);
        println!(
            "  {label:22} {:8.2}ms {:8.2}ms {:9.2}x   f32 {:5.1} / bf16 {:5.1} GB/s",
            b * 1e3,
            f * 1e3,
            b / f,
            (out_dim * in_dim * 4) as f64 / b / 1e9,
            (out_dim * in_dim * 2) as f64 / f / 1e9,
        );
    }
    println!();
}
