//! The dense projection kernels, each declaring who threads it.
//!
//! None of them spawns. Every one computes exactly the output rows it is
//! handed; the executor decides how the rows were cut.

use super::projector::{CpuParallelism, DenseProjector, WeightRows};

/// The literal transcription: one scalar dot per row, f32 weights.
///
/// Measured at a flat 5.6 GB/s across every Qwen3.8 projection shape,
/// which is why it is the oracle rather than the execution strategy. Kept
/// [`CpuParallelism::Serial`] deliberately: the reference path's value is
/// that it can be read line-by-line beside the source it transcribes, and
/// threading it would buy speed in the one place speed is not the point.
pub struct ScalarF32;

impl DenseProjector for ScalarF32 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::Serial
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        let WeightRows::F32(w) = weight_rows else {
            panic!("the scalar reference kernel consumes f32 weights only");
        };
        let in_dim = x.len();
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for (a, b) in row.iter().zip(x) {
                acc += a * b;
            }
            *slot = acc;
        }
    }
}

/// BLAS `sgemv` through `larql-compute` — Accelerate on macOS, OpenBLAS
/// on Linux/FreeBSD, scalar on Windows by deliberate choice.
///
/// [`CpuParallelism::LibraryOwned`] because it already threads itself:
/// partitioning rows on top won 1.14x at best and lost on `5120 x 6144`.
pub struct BlasF32;

impl DenseProjector for BlasF32 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::LibraryOwned
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        let WeightRows::F32(w) = weight_rows else {
            panic!("the BLAS kernel consumes f32 weights only");
        };
        let y = larql_compute::cpu::ops::moe::math::matmul_vec(x, w, out.len(), x.len());
        out.copy_from_slice(&y);
    }
}

/// **Fused BF16.** Load the compact code units, widen in REGISTERS,
/// multiply by the f32 activation, accumulate f32, discard.
///
/// The representation stays compact all the way into SIMD registers.
/// CPU-1B measured the alternative — widen a tile into scratch, then call
/// `sgemv` — at 27.3 GB/s against this kernel's 122.0, i.e. slower than
/// plain f32 despite reading half the bytes. Compact-to-registers is the
/// architecture; BF16 is only its first instance.
///
/// **Not always the right kernel.** Measured through the executor, this
/// wins 2.07-2.68x on the streaming shapes and LOSES 0.26x on the tiny
/// `48 x 5120` delta projections: at ~0.5 MB they are cache-resident, so
/// there is no RAM traffic to halve, and Accelerate's cache-resident
/// `sgemv` (262 GB/s) beats a serial widen-and-FMA loop (34 GB/s). Format
/// choice belongs per matrix class alongside worker count, not to the
/// model as a whole.
///
/// The widen is EXACT: bf16 is the top half of f32, so `(bits as u32) <<
/// 16` reproduces the value with no rounding and no table. The activation
/// stays f32 and the accumulator stays f32, so this changes
/// representation and mechanics and no numerical value — measured at
/// rel_rms 3.6e-7 against BLAS, which is summation order alone. Rounding
/// activations to bf16 to reach `BFDOT` is a separate precision decision
/// and is not made here.
pub struct FusedBf16;

impl DenseProjector for FusedBf16 {
    fn parallelism(&self) -> CpuParallelism {
        CpuParallelism::ExternalPool
    }

    fn project_rows(&self, weight_rows: WeightRows<'_>, x: &[f32], out: &mut [f32]) {
        let WeightRows::Bf16(w) = weight_rows else {
            panic!("the fused bf16 kernel consumes bf16 weights only");
        };
        let in_dim = x.len();
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            *slot = bf16_dot(row, x);
        }
    }
}

/// One row's dot product, widening in registers.
#[inline]
pub(super) fn bf16_dot(w: &[u16], x: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on every aarch64 target Rust supports,
        // and the loop reads only within `w` and `x`, which are equal
        // length by the caller's contract.
        return unsafe { bf16_dot_neon(w, x) };
    }
    #[allow(unreachable_code)]
    bf16_dot_portable(w, x)
}

/// The portable widen-and-accumulate, and the definition the NEON
/// version must agree with.
pub(super) fn bf16_dot_portable(w: &[u16], x: &[f32]) -> f32 {
    w.iter()
        .zip(x)
        .map(|(b, v)| f32::from_bits((*b as u32) << 16) * v)
        .sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn bf16_dot_neon(w: &[u16], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = x.len().min(w.len());
    let (wp, xp) = (w.as_ptr(), x.as_ptr());
    // Four accumulators to hide FMA latency.
    let (mut a0, mut a1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let (mut a2, mut a3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let mut i = 0usize;
    while i + 16 <= n {
        let w0 = vld1q_u16(wp.add(i));
        let w1 = vld1q_u16(wp.add(i + 8));
        let f0 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(w0)), 16));
        let f1 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(w0)), 16));
        let f2 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(w1)), 16));
        let f3 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(w1)), 16));
        a0 = vfmaq_f32(a0, f0, vld1q_f32(xp.add(i)));
        a1 = vfmaq_f32(a1, f1, vld1q_f32(xp.add(i + 4)));
        a2 = vfmaq_f32(a2, f2, vld1q_f32(xp.add(i + 8)));
        a3 = vfmaq_f32(a3, f3, vld1q_f32(xp.add(i + 12)));
        i += 16;
    }
    let mut acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    while i < n {
        acc += f32::from_bits((*w.get_unchecked(i) as u32) << 16) * *x.get_unchecked(i);
        i += 1;
    }
    acc
}
