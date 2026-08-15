//! Elementwise glue the VINDEX3 plan needs and the serving path has no
//! kernel for (VINDEX3-G6b).
//!
//! Two operations, both judged semantics rather than conveniences:
//!
//! - **Parameter-free QK norm.** Weightless per-head RMS. The existing
//!   `qk_norm` kernels all take a weight tensor, and Muse-Glimmer's Q and
//!   K normalisation has none — nothing in the checkpoint evidences it,
//!   which is exactly why it is carried as a judged fact
//!   (`ParameterFreeQkNorm { q: true, k: true }`) rather than inferred
//!   from operands.
//!
//! - **Sigmoid attention output gate.** `AttentionGateSpec` with
//!   `source: AttentionInput`, `activation: Sigmoid`,
//!   `combine: ElementwiseMultiply`, applied after head aggregation and
//!   before the output projection.
//!
//! The CPU reference accumulates the QK-norm sum of squares in **f64**
//! and casts the resulting RMS to f32. Metal has no f64, so this
//! accumulates in f32 — a genuine realisation difference, bounded by
//! `head_dim` terms (128 for Glimmer) and judged by the parity gate
//! rather than assumed harmless.

pub const SHADER: &str = r#"
// Weightless per-head RMS: out = x / sqrt(mean(x^2) + eps), one
// threadgroup per head. Matches `rms_norm_heads_no_weight_eps`.
kernel void qk_norm_parameter_free(
    device float*    x        [[buffer(0)]],
    constant uint&   head_dim [[buffer(1)]],
    constant float&  eps      [[buffer(2)]],
    uint  head  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_index_in_threadgroup]],
    uint  tg_sz [[threads_per_threadgroup]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg    [[simdgroup_index_in_threadgroup]])
{
    device float* h = x + (ulong)head * (ulong)head_dim;

    float partial = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_sz) {
        const float v = h[i];
        partial += v * v;
    }
    partial = simd_sum(partial);

    // Combine simdgroup partials. 32 slots covers the largest
    // threadgroup this is dispatched with (1024 threads / 32 lanes).
    threadgroup float sums[32];
    if (lane == 0u) { sums[sg] = partial; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_sg = (tg_sz + 31u) / 32u;
    float total = 0.0f;
    for (uint i = 0u; i < num_sg; ++i) { total += sums[i]; }

    const float inv = 1.0f / sqrt(total / float(head_dim) + eps);
    for (uint i = tid; i < head_dim; i += tg_sz) {
        h[i] = h[i] * inv;
    }
}

// out = a * sigmoid(g) — the judged attention output gate.
kernel void sigmoid_gate_multiply(
    device const float* a   [[buffer(0)]],
    device const float* g   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      N   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) { return; }
    out[tid] = a[tid] * (1.0f / (1.0f + exp(-g[tid])));
}
"#;

/// Marker for the weightless per-head Q/K RMS pipeline.
pub struct QkNormParameterFreeKernel;
impl crate::kernels::ShaderKernel for QkNormParameterFreeKernel {
    const KERNEL_NAME: &'static str = "qk_norm_parameter_free";
}

/// Marker for the judged sigmoid attention-gate pipeline.
pub struct SigmoidGateMultiplyKernel;
impl crate::kernels::ShaderKernel for SigmoidGateMultiplyKernel {
    const KERNEL_NAME: &'static str = "sigmoid_gate_multiply";
}
