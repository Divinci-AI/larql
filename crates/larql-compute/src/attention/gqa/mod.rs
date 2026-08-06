//! Grouped-Query Attention (GQA) — causal attention with BLAS-fused dot products.
//!
//! Memory-efficient: O(seq) per position, never materializes full [seq, seq] matrix.
//! Uses BLAS gemv for both Q·K scores and softmax·V accumulation.

use super::span::AttentionSpan;
use super::{AttentionAllWeights, AttentionWeights};
use ndarray::Array2;

/// GQA with causal masking (no weight capture).
/// q: (seq, num_q * head_dim), k: (seq, num_kv * head_dim), v: same as k
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
) -> Array2<f32> {
    let (out, _) = gqa_attention_with_weights(
        q, k, v, num_q, head_dim, reps, scale, seq_len, false, None, None,
    );
    out
}

/// GQA that optionally captures per-head attention weights for the last token.
/// `softcap`: if Some(cap), apply tanh(scores/cap)*cap before softmax.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    capture: bool,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
) -> (Array2<f32>, Option<AttentionWeights>) {
    gqa_attention_with_weights_in_span(
        q,
        k,
        v,
        num_q,
        head_dim,
        reps,
        scale,
        seq_len,
        capture,
        softcap,
        sinks,
        AttentionSpan::Full,
    )
}

/// [`gqa_attention_with_weights`] restricted to an [`AttentionSpan`].
///
/// The span-carrying form is the real entry point; the unspanned name above
/// is the `AttentionSpan::Full` wrapper, the same layering `rope.rs` uses for
/// its scaling variants. Callers that know the layer must use this one — a
/// hybrid model served through the wrapper runs full attention on layers the
/// reference windows.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_weights_in_span(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    capture: bool,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
    span: AttentionSpan,
) -> (Array2<f32>, Option<AttentionWeights>) {
    let (out, last, _) = gqa_attention_capture(
        q, k, v, num_q, head_dim, reps, scale, seq_len, capture, false, softcap, sinks, span,
    );
    (out, last)
}

/// GQA that captures every query-position attention distribution.
///
/// Diagnostic/capture tooling uses this for relation-state probes. Production
/// inference should use [`gqa_attention`] or [`gqa_attention_with_weights`].
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_all_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
) -> (Array2<f32>, AttentionAllWeights) {
    gqa_attention_with_all_weights_in_span(
        q,
        k,
        v,
        num_q,
        head_dim,
        reps,
        scale,
        seq_len,
        softcap,
        sinks,
        AttentionSpan::Full,
    )
}

/// [`gqa_attention_with_all_weights`] restricted to an [`AttentionSpan`].
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_all_weights_in_span(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
    span: AttentionSpan,
) -> (Array2<f32>, AttentionAllWeights) {
    let (out, _, all) = gqa_attention_capture(
        q, k, v, num_q, head_dim, reps, scale, seq_len, false, true, softcap, sinks, span,
    );
    (
        out,
        all.expect("all-position attention capture requested but missing"),
    )
}

/// Capture every query-position attention distribution using only the first
/// `qk_rank` dimensions of each Q/K head. This is a diagnostic surface for
/// reduced-QK address probes; it does not compute a V-weighted output.
#[allow(clippy::too_many_arguments)]
pub fn gqa_reduced_qk_all_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
    qk_rank: usize,
    span: AttentionSpan,
) -> AttentionAllWeights {
    let rank = qk_rank.clamp(1, head_dim);
    let mut captured_all_heads: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_q);
    let scale_f32 = scale as f32;
    let mut scores_buf = vec![0.0f32; seq_len];

    for h in 0..num_q {
        // Per-head learned sink logit; `None` for architectures without them.
        let sink = sinks.map(|s| s[h]);
        let mut captured_positions: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        for qi in 0..seq_len {
            let keys = span.range(qi);
            let visible = keys.len();
            let q_row = q.slice(ndarray::s![qi, q_off..q_off + rank]);
            let k_block = k.slice(ndarray::s![keys.clone(), kv_off..kv_off + rank]);
            let raw_scores = k_block.dot(&q_row);

            for i in 0..visible {
                let mut s = raw_scores[i] * scale_f32;
                if let Some(cap) = softcap {
                    s = (s / cap).tanh() * cap;
                }
                scores_buf[i] = s;
            }

            super::softmax::softmax_in_place(&mut scores_buf[..visible], sink);

            let mut captured = vec![0.0f32; seq_len];
            captured[keys].copy_from_slice(&scores_buf[..visible]);
            captured_positions.push(captured);
        }
        captured_all_heads.push(captured_positions);
    }

    AttentionAllWeights {
        heads: captured_all_heads,
    }
}

#[allow(clippy::too_many_arguments)]
fn gqa_attention_capture(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    capture_last: bool,
    capture_all: bool,
    softcap: Option<f32>,
    sinks: Option<&[f32]>,
    span: AttentionSpan,
) -> (
    Array2<f32>,
    Option<AttentionWeights>,
    Option<AttentionAllWeights>,
) {
    let mut out = Array2::<f32>::zeros((seq_len, num_q * head_dim));
    let mut captured_heads: Vec<Vec<f32>> = if capture_last {
        Vec::with_capacity(num_q)
    } else {
        Vec::new()
    };
    let mut captured_all_heads: Vec<Vec<Vec<f32>>> = if capture_all {
        Vec::with_capacity(num_q)
    } else {
        Vec::new()
    };

    let scale_f32 = scale as f32;
    let last_pos = seq_len - 1;
    let mut scores_buf = vec![0.0f32; seq_len];

    for h in 0..num_q {
        // Per-head learned sink logit; `None` for architectures without them.
        let sink = sinks.map(|s| s[h]);
        let mut captured_positions: Vec<Vec<f32>> = if capture_all {
            Vec::with_capacity(seq_len)
        } else {
            Vec::new()
        };
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        for qi in 0..seq_len {
            // Causal end, span-dependent start. `visible` indexes the score
            // buffer; `keys` indexes absolute positions in K/V. Captured
            // distributions are written back at their absolute offset so a
            // consumer still reads position `j` at index `j`.
            let keys = span.range(qi);
            let visible = keys.len();

            let q_row = q.slice(ndarray::s![qi, q_off..q_off + head_dim]);
            let k_block = k.slice(ndarray::s![keys.clone(), kv_off..kv_off + head_dim]);
            let raw_scores = k_block.dot(&q_row);

            for i in 0..visible {
                let mut s = raw_scores[i] * scale_f32;
                if let Some(cap) = softcap {
                    s = (s / cap).tanh() * cap;
                }
                scores_buf[i] = s;
            }

            super::softmax::softmax_in_place(&mut scores_buf[..visible], sink);

            if capture_last && qi == last_pos {
                let mut captured = vec![0.0f32; seq_len];
                captured[keys.clone()].copy_from_slice(&scores_buf[..visible]);
                captured_heads.push(captured);
            }
            if capture_all {
                let mut captured = vec![0.0f32; seq_len];
                captured[keys.clone()].copy_from_slice(&scores_buf[..visible]);
                captured_positions.push(captured);
            }

            let v_block = v.slice(ndarray::s![keys, kv_off..kv_off + head_dim]);
            let scores_view = ndarray::ArrayView1::from(&scores_buf[..visible]);
            let weighted_v = v_block.t().dot(&scores_view);

            for d in 0..head_dim {
                out[[qi, q_off + d]] = weighted_v[d];
            }
        }
        if capture_all {
            captured_all_heads.push(captured_positions);
        }
    }

    let weights = if capture_last {
        Some(AttentionWeights {
            heads: captured_heads,
        })
    } else {
        None
    };

    let all_weights = if capture_all {
        Some(AttentionAllWeights {
            heads: captured_all_heads,
        })
    } else {
        None
    };

    (out, weights, all_weights)
}

/// GQA with asymmetric Q/K vs V head dimensions — required for MLA-absorbed attention.
///
/// `qk_head_dim`: head dimension for Q and K (e.g. 192 for DS-V3: nope=128 + rope=64).
/// `v_head_dim`: head dimension for V and the output (e.g. 128 for DS-V3).
///
/// q: (seq, num_q * qk_head_dim), k: (seq, num_kv * qk_head_dim), v: (seq, num_kv * v_head_dim)
/// Returns: (seq, num_q * v_head_dim)
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_asym(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    sinks: Option<&[f32]>,
) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((seq_len, num_q * v_head_dim));
    let scale_f32 = scale as f32;
    let mut scores_buf = vec![0.0f32; seq_len];

    for h in 0..num_q {
        // Per-head learned sink logit; `None` for architectures without them.
        let sink = sinks.map(|s| s[h]);
        let kv_h = h / reps;
        let q_off = h * qk_head_dim;
        let kv_qk_off = kv_h * qk_head_dim;
        let kv_v_off = kv_h * v_head_dim;
        let out_off = h * v_head_dim;

        for qi in 0..seq_len {
            let causal_len = qi + 1;
            let q_row = q.slice(ndarray::s![qi, q_off..q_off + qk_head_dim]);
            let k_block = k.slice(ndarray::s![
                0..causal_len,
                kv_qk_off..kv_qk_off + qk_head_dim
            ]);
            let raw_scores = k_block.dot(&q_row);

            for i in 0..causal_len {
                scores_buf[i] = raw_scores[i] * scale_f32;
            }
            super::softmax::softmax_in_place(&mut scores_buf[..causal_len], sink);

            let v_block = v.slice(ndarray::s![0..causal_len, kv_v_off..kv_v_off + v_head_dim]);
            for d in 0..v_head_dim {
                let mut acc = 0.0f32;
                for i in 0..causal_len {
                    acc += scores_buf[i] * v_block[(i, d)];
                }
                out[(qi, out_off + d)] = acc;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
