//! Q4_K dense FFN backend — the quantised sibling of [`super::weight::WeightFfn`].
//!
//! Weights are quantised once at construction (row-major Q4_K
//! super-blocks) and every forward consumes the packed representation
//! directly through the Q4K·Q8K integer-dot kernels. The lesson the TTS
//! funnel's frame benchmark made explicit (`docs/tts-funnel.md`):
//! quantisation only wins when the kernel eats the compressed format —
//! the dequantise-then-BLAS path is dramatically *slower* than fp32.
//!
//! Scope: gated-SiLU FFNs without biases (the Qwen3 family, which covers
//! both MOSS transformers). Construction refuses anything else rather
//! than silently computing a different function.

use ndarray::Array2;

use larql_models::{Activation, FfnType, ModelWeights};

use crate::cpu::ops::q4_common::quantize_q4_k;
use crate::cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_parallel;
use crate::quantize_x_to_q8k;

use super::silu_gate_up;
use super::FfnBackend;

/// Q4_K super-blocks span this many weights; every matrix dimension fed
/// to [`quantize_q4_k`] must divide into them row by row.
const Q4K_BLOCK_ELEMS: usize = 256;
/// Registry tag for the packed format, as the kernel dispatch expects.
const Q4K_FORMAT_TAG: &str = "Q4_K";

struct Q4kLayer {
    gate: Vec<u8>,
    up: Vec<u8>,
    down: Vec<u8>,
    hidden: usize,
    intermediate: usize,
}

/// Dense gated-SiLU FFN over Q4_K-packed weights.
pub struct Q4kFfn {
    layers: Vec<Q4kLayer>,
}

impl Q4kFfn {
    /// Quantise `weights`' FFN projections. Errors (by message) on
    /// non-gated / non-SiLU architectures, missing tensors, biased FFNs,
    /// or dimensions that don't divide into Q4_K blocks — every case
    /// where "quantised" would quietly mean "different math".
    pub fn quantize_from(weights: &ModelWeights) -> Result<Self, String> {
        let arch = &*weights.arch;
        if arch.ffn_type() != FfnType::Gated || arch.activation() != Activation::Silu {
            return Err(format!(
                "Q4kFfn supports gated-SiLU FFNs; {} is {:?}/{:?}",
                arch.family(),
                arch.ffn_type(),
                arch.activation()
            ));
        }
        let mut layers = Vec::with_capacity(weights.num_layers);
        for layer in 0..weights.num_layers {
            if arch.ffn_up_bias_key(layer).is_some() || arch.ffn_down_bias_key(layer).is_some() {
                return Err("Q4kFfn does not support FFN biases".to_string());
            }
            let pack = |key: String| -> Result<(Vec<u8>, usize, usize), String> {
                let tensor = weights
                    .tensors
                    .get(&key)
                    .ok_or_else(|| format!("missing FFN tensor {key}"))?;
                let (rows, cols) = tensor.dim();
                if cols % Q4K_BLOCK_ELEMS != 0 {
                    return Err(format!(
                        "{key}: {cols} columns not a multiple of {Q4K_BLOCK_ELEMS}"
                    ));
                }
                let data = tensor
                    .as_standard_layout()
                    .as_slice()
                    .expect("standard layout")
                    .to_vec();
                Ok((quantize_q4_k(&data), rows, cols))
            };
            let (gate, intermediate, hidden) = pack(arch.ffn_gate_key(layer))?;
            let (up, up_rows, up_cols) = pack(arch.ffn_up_key(layer))?;
            let (down, down_rows, down_cols) = pack(arch.ffn_down_key(layer))?;
            if (up_rows, up_cols) != (intermediate, hidden)
                || (down_rows, down_cols) != (hidden, intermediate)
            {
                return Err(format!(
                    "layer {layer}: inconsistent FFN shapes \
                     gate [{intermediate},{hidden}] up [{up_rows},{up_cols}] \
                     down [{down_rows},{down_cols}]"
                ));
            }
            layers.push(Q4kLayer {
                gate,
                up,
                down,
                hidden,
                intermediate,
            });
        }
        Ok(Self { layers })
    }
}

impl FfnBackend for Q4kFfn {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let l = &self.layers[layer];
        let rows = x.nrows();
        let mut gate = Array2::<f32>::zeros((rows, l.intermediate));
        let mut up = Array2::<f32>::zeros((rows, l.intermediate));
        for (row, x_row) in x.rows().into_iter().enumerate() {
            let x_vec: Vec<f32> = x_row.to_vec();
            let q8 = quantize_x_to_q8k(&x_vec);
            q4k_q8k_matvec_parallel(
                gate.row_mut(row).as_slice_mut().expect("contiguous"),
                &q8,
                &l.gate,
                l.intermediate,
                l.hidden,
                Q4K_FORMAT_TAG,
            );
            q4k_q8k_matvec_parallel(
                up.row_mut(row).as_slice_mut().expect("contiguous"),
                &q8,
                &l.up,
                l.intermediate,
                l.hidden,
                Q4K_FORMAT_TAG,
            );
        }
        let activation = silu_gate_up(&gate, &up);
        let mut out = Array2::<f32>::zeros((rows, l.hidden));
        for (row, act_row) in activation.rows().into_iter().enumerate() {
            let act_vec: Vec<f32> = act_row.to_vec();
            let q8 = quantize_x_to_q8k(&act_vec);
            q4k_q8k_matvec_parallel(
                out.row_mut(row).as_slice_mut().expect("contiguous"),
                &q8,
                &l.down,
                l.hidden,
                l.intermediate,
                Q4K_FORMAT_TAG,
            );
        }
        out
    }

    fn name(&self) -> &str {
        "q4k-weights"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffn::WeightFfn;
    use larql_models::test_fixtures::make_test_weights;
    use larql_models::WeightArray;
    use std::collections::HashMap;

    /// A 1-layer gated-SiLU model with Q4K-divisible dims, values small
    /// enough that Q4_K error stays well-behaved.
    fn q4k_divisible_weights() -> ModelWeights {
        const HIDDEN: usize = 256;
        const INTER: usize = 512;
        let arch_json = serde_json::json!({
            "model_type": "tinymodel",
            "hidden_size": HIDDEN,
            "num_hidden_layers": 1,
            "intermediate_size": INTER,
            "head_dim": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 32,
        });
        let arch = larql_models::detect_from_json(&arch_json);
        let mut state = 0x1234_5678_u64;
        let mut mat = |rows: usize, cols: usize| -> WeightArray {
            let data: Vec<f32> = (0..rows * cols)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 33) as f32 / u32::MAX as f32 - 0.25) * 0.2
                })
                .collect();
            Array2::from_shape_vec((rows, cols), data)
                .unwrap()
                .into_shared()
        };
        let mut tensors: HashMap<String, WeightArray> = HashMap::new();
        tensors.insert(arch.ffn_gate_key(0), mat(INTER, HIDDEN));
        tensors.insert(arch.ffn_up_key(0), mat(INTER, HIDDEN));
        tensors.insert(arch.ffn_down_key(0), mat(HIDDEN, INTER));
        let embed = mat(32, HIDDEN);
        ModelWeights {
            tensors,
            vectors: HashMap::new(),
            raw_bytes: HashMap::new(),
            packed_mmaps: HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_byte_ranges: HashMap::new(),
            embed: embed.clone(),
            lm_head: embed,
            position_embed: None,
            num_layers: 1,
            hidden_size: HIDDEN,
            intermediate_size: INTER,
            vocab_size: 32,
            head_dim: 64,
            num_q_heads: 4,
            num_kv_heads: 2,
            rope_base: 10000.0,
            arch,
        }
    }

    #[test]
    fn q4k_ffn_close_to_dense() {
        let weights = q4k_divisible_weights();
        let dense = WeightFfn { weights: &weights };
        let q4k = Q4kFfn::quantize_from(&weights).unwrap();

        let x = Array2::from_shape_fn((3, weights.hidden_size), |(r, c)| {
            ((r * 31 + c) % 17) as f32 * 0.02 - 0.15
        });
        let reference = dense.forward(0, &x);
        let quantised = q4k.forward(0, &x);

        assert_eq!(reference.dim(), quantised.dim());
        // Quantisation noise on a gated FFN with near-cancelling sums is
        // genuinely nonzero; what distinguishes "kernel wrong" from
        // "4-bit weights" is the overall direction and energy of the
        // output, not per-element deltas.
        let (mut dot, mut ref_sq, mut q_sq, mut diff_sq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in reference.iter().zip(quantised.iter()) {
            dot += a as f64 * b as f64;
            ref_sq += (a as f64).powi(2);
            q_sq += (b as f64).powi(2);
            diff_sq += (a as f64 - b as f64).powi(2);
        }
        let cosine = dot / (ref_sq.sqrt() * q_sq.sqrt());
        let relative = (diff_sq / ref_sq).sqrt();
        assert!(
            cosine > 0.98 && relative < 0.2,
            "Q4K FFN drifted: cosine {cosine:.4}, relative error {relative:.4}"
        );
    }

    #[test]
    fn refuses_non_divisible_dims() {
        let weights = make_test_weights();
        assert!(Q4kFfn::quantize_from(&weights).is_err());
    }
}
