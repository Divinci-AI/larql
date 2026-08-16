//! NVFP4 matrix-vector multiply — direct compressed execution.
//!
//! **Q2-R1.** The MXFP4 sibling ([`super::mxfp4_matvec`]) proved E2M1 can
//! be a compute format; this one changes only the *scale* geometry, which
//! is the variable VINDEX3-Q2 is testing:
//!
//! ```text
//!            elements   group   group scale   tensor scale
//! MXFP4      E2M1       32      E8M0          —
//! NVFP4      E2M1       16      E4M3          one f32
//! ```
//!
//! A weight-reconstruction sweep over Muse-Glimmer's real tensors, with
//! an equal-bit-budget control (E8M0 at group 16, also 4.5 bpw), found
//! the group size worth nothing — 0.996x on attention — and the scale
//! format worth 1.265x in relative RMS and 1.68x in worst-element error.
//! So the format under test here is specifically E4M3-scaled, and the
//! kernel keeps E2M1 decode byte-identical to the MXFP4 path so the two
//! differ in nothing else.
//!
//! ## Format
//!
//! Per output row, `groups = K / 16`. Group `g` holds:
//!   - 8 packed bytes at `packed[(row * groups + g) * 8 ..][..8]`, each
//!     carrying two 4-bit codes: **lo nibble first**, then hi.
//!   - one E4M3 scale byte at `scales[row * groups + g]`.
//!
//! and one f32 `tensor_scale` multiplies every decoded element:
//!
//! ```text
//! w[row, g*16 + i] = tensor_scale * e4m3(scale) * e2m1(code)
//! ```
//!
//! The association matters: the CPU reference folds `tensor_scale *
//! e4m3(scale)` into one step per group and multiplies the E2M1 code by
//! it, and this kernel does the same, so the two agree to fp rounding
//! rather than by luck.
//!
//! E4M3 decode follows OCP FP8 v1.0 and mirrors `quant::fp8::e4m3_to_f32`
//! exactly, including subnormals (`exp == 0` → `mant * 2^-9`) and the two
//! NaN encodings (`0x7F`, `0xFF`). Subnormals are not decorative here:
//! the tensor scale normalises the largest group to E4M3's *top*, so a
//! matrix with a wide spread of group amaxes pushes its quietest groups
//! into the subnormal range, and flushing them to zero would silently
//! delete whole groups of weights.
//!
//! ## Parallelism
//!
//! One simdgroup per output row, `ROWS_PER_TG` simdgroups per
//! threadgroup — the MXFP4 geometry unchanged, deliberately: a dispatch
//! shape that collapses threadgroup count has cost more than it saved
//! before, and this rung is an accuracy question, not a tuning one.
//!
//! Lane `l` walks groups `l, l+32, ...`, reading one contiguous 8-byte
//! group each; adjacent lanes cover 256 contiguous bytes per step. Half
//! the per-lane bytes of the MXFP4 kernel because the group is half as
//! wide, so a row of the same `K` takes the same number of steps with
//! twice the scale reads. The K reduction closes with `simd_sum`.
//!
//! Accumulation order differs from the CPU reference (which sums
//! left-to-right), so parity is a bounded-error contract, not
//! bit-equality — the same contract the MXFP4 rung established.

/// Output rows per threadgroup — one simdgroup each.
pub const ROWS_PER_TG: u64 = 4;
/// 4 simdgroups x 32 lanes.
pub const THREADS_PER_TG: u64 = 128;

pub const SHADER: &str = r#"
constant uint NVFP4_ROWS_PER_TG = 4;
constant uint NVFP4_GROUP_ELEMS = 16;
constant uint NVFP4_GROUP_BYTES = 8;

// ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} — sign in bit 3, then exp(2) and mantissa(1).
// Identical to MXFP4_LUT: the element grid is the shared half of the two
// formats, and Q2 is about the scale.
constant float NVFP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// E4M3 -> f32, matching quant::fp8::e4m3_to_f32 including subnormals and
// both NaN encodings. 1 sign, 4 exponent (bias 7), 3 mantissa; no Inf.
inline float nvfp4_e4m3(uchar b) {
    const uint sign = uint(b) >> 7;
    const uint exp  = (uint(b) >> 3) & 0xFu;
    const uint mant = uint(b) & 0x7u;
    float mag;
    if (exp == 0u) {
        // Subnormal: mant/8 * 2^-6 == mant * 2^-9. Reached routinely,
        // because the tensor scale pins the loudest group at E4M3's top
        // and pushes quiet groups down here.
        mag = float(mant) * 0.001953125f;   // 2^-9
    } else if (exp == 0xFu && mant == 0x7u) {
        mag = NAN;
    } else {
        mag = (1.0f + float(mant) * 0.125f) * exp2(float(int(exp) - 7));
    }
    return (sign != 0u) ? -mag : mag;
}

kernel void nvfp4_matvec(
    device const uchar*  Wp     [[buffer(0)]],   // packed [M, groups, 8]
    device const uchar*  Ws     [[buffer(1)]],   // scales [M, groups] E4M3
    device const float*  X      [[buffer(2)]],   // [K]
    device float*        out    [[buffer(3)]],   // [M]
    constant uint&       M      [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    constant float&      Tscale [[buffer(6)]],   // one f32 for the matrix
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * NVFP4_ROWS_PER_TG + sg_id;
    if (row >= M) { return; }

    const uint groups = K / NVFP4_GROUP_ELEMS;
    device const uchar* row_p = Wp + (ulong)row * (ulong)groups * NVFP4_GROUP_BYTES;
    device const uchar* row_s = Ws + (ulong)row * (ulong)groups;

    float acc = 0.0f;

    // Lane l walks groups l, l+32, ... — one contiguous 8-byte read each,
    // 256 contiguous bytes across the simdgroup per step.
    for (uint g = lane; g < groups; g += 32u) {
        // Fold both scale levels once per group, exactly as the CPU
        // reference does, then apply to the E2M1 codes.
        const float step = Tscale * nvfp4_e4m3(row_s[g]);
        device const uchar* blk = row_p + (ulong)g * NVFP4_GROUP_BYTES;
        const uint base = g * NVFP4_GROUP_ELEMS;

        // Scalar byte loads, deliberately. A `uint2` + `float4` variant
        // measured *slower* (101.0 vs 110.3 GB/s over one layer's four
        // projections), so the compiler is already vectorising this and
        // load width is not what the kernel is short of.
        float part = 0.0f;
        for (uint b = 0u; b < NVFP4_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            part += NVFP4_LUT[byte & 0x0Fu]         * X[base + 2u * b];
            part += NVFP4_LUT[(byte >> 4u) & 0x0Fu] * X[base + 2u * b + 1u];
        }
        acc += step * part;
    }

    acc = simd_sum(acc);
    if (lane == 0u) { out[row] = acc; }
}
"#;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "nvfp4_matvec";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
