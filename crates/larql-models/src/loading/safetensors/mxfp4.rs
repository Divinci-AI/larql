//! MXFP4 packed-expert detection and per-expert dequantisation.
//!
//! Split out of `safetensors.rs` (ROADMAP H5b). GPT-OSS ships its MoE
//! experts as packed `*_blocks` / `*_scales` pairs rather than one tensor
//! per expert; this module recognises that layout and expands it into the
//! per-expert Mixtral-style keys the rest of the loader expects.

use std::collections::HashMap;

use ndarray::Array2;

use crate::detect::ModelError;

use super::{
    normalize_key, tensor_to_f32, BLOCK_SPARSE_ROUTER_WEIGHT, MIXTRAL_DOWN_PROJ, MIXTRAL_GATE_PROJ,
    MIXTRAL_UP_PROJ, MXFP4_ROUTER_WEIGHT,
};

// Packed-expert tensor-name suffixes. These name the MXFP4 layout, so
// they live with the code that understands it rather than in the shard
// walker that merely passes names through.
pub(super) const MXFP4_GATE_UP_BLOCKS_SUFFIX: &str = ".gate_up_proj_blocks";
pub(super) const MXFP4_BLOCKS_SUFFIX: &str = "_blocks";
pub(super) const MXFP4_SCALES_SUFFIX: &str = "_scales";
pub(super) const MXFP4_GATE_UP_BLOCKS: &str = "gate_up_proj_blocks";
pub(super) const MXFP4_EXPERTS_GATE_UP_BLOCKS: &str = "experts.gate_up_proj_blocks";
pub(super) const MXFP4_DOWN_BLOCKS: &str = "down_proj_blocks";
pub(super) const MXFP4_DOWN_SCALES: &str = "down_proj_scales";

/// Load GPT-OSS MXFP4 packed expert tensors from a safetensors file into the
/// weights map, using per-expert Mixtral-style key names.
///
/// GPT-OSS stores experts as:
///   layers.{L}.mlp.experts.gate_up_proj_blocks: [experts, 2*hidden, groups, 16] U8
///   layers.{L}.mlp.experts.gate_up_proj_scales: [experts, 2*hidden, groups] U8
///   layers.{L}.mlp.experts.down_proj_blocks: [experts, hidden, groups, 16] U8
///   layers.{L}.mlp.experts.down_proj_scales: [experts, hidden, groups] U8
///
/// Dequantization and gate/up splitting are handled by `quant::mxfp4`.
/// Output keys follow Mixtral conventions:
///   layers.{L}.block_sparse_moe.experts.{E}.w1.weight (gate)
///   layers.{L}.block_sparse_moe.experts.{E}.w3.weight (up)
///   layers.{L}.block_sparse_moe.experts.{E}.w2.weight (down)
pub(super) fn load_mxfp4_expert_tensors(
    st: &safetensors::SafeTensors,
    tensor_names: &[String],
    prefixes: &[&str],
    skip_key: &impl Fn(&str) -> bool,
    tensors: &mut HashMap<String, crate::WeightArray>,
) -> Result<(), ModelError> {
    for name in tensor_names {
        if !name.ends_with(MXFP4_GATE_UP_BLOCKS_SUFFIX) {
            continue;
        }

        let scales_name = name.replace(MXFP4_BLOCKS_SUFFIX, MXFP4_SCALES_SUFFIX);
        let down_blocks_name = name.replace(MXFP4_GATE_UP_BLOCKS, MXFP4_DOWN_BLOCKS);
        let down_scales_name = name.replace(MXFP4_GATE_UP_BLOCKS, MXFP4_DOWN_SCALES);

        let blocks_view = st
            .tensor(name)
            .map_err(|e| ModelError::Parse(format!("MXFP4 blocks: {e}")))?;
        let scales_view = st
            .tensor(&scales_name)
            .map_err(|e| ModelError::Parse(format!("MXFP4 scales: {e}")))?;

        let shape = blocks_view.shape();
        if shape.len() != 4 {
            continue;
        }

        let num_experts = shape[0];
        let out_features = shape[1]; // = 2 * hidden (gate + up fused)
        let groups = shape[2];
        let in_features = groups * 32;
        let half = out_features / 2;

        let base_key = normalize_key(name, prefixes);
        let layer_prefix = base_key.split(".mlp.").next().unwrap_or("");
        let should_load_gate_up = (0..num_experts).any(|e| {
            !skip_key(&mxfp4_expert_key(layer_prefix, e, MIXTRAL_GATE_PROJ))
                || !skip_key(&mxfp4_expert_key(layer_prefix, e, MIXTRAL_UP_PROJ))
        });

        // Dequantize and split fused gate_up → separate gate (w1) and up (w3).
        if should_load_gate_up {
            let (gate_experts, up_experts) = crate::quant::mxfp4::split_gate_up_experts(
                blocks_view.data(),
                scales_view.data(),
                num_experts,
                out_features,
                groups,
            )?;

            for (e, (gate_data, up_data)) in gate_experts.into_iter().zip(up_experts).enumerate() {
                let gate_key = mxfp4_expert_key(layer_prefix, e, MIXTRAL_GATE_PROJ);
                if !skip_key(&gate_key) {
                    tensors.insert(
                        gate_key,
                        Array2::from_shape_vec((half, in_features), gate_data)
                            .map_err(|e| ModelError::Parse(e.to_string()))?
                            .into_shared(),
                    );
                }
                let up_key = mxfp4_expert_key(layer_prefix, e, MIXTRAL_UP_PROJ);
                if !skip_key(&up_key) {
                    tensors.insert(
                        up_key,
                        Array2::from_shape_vec((half, in_features), up_data)
                            .map_err(|e| ModelError::Parse(e.to_string()))?
                            .into_shared(),
                    );
                }
            }
        }

        // Dequantize down projection.
        if let (Ok(db), Ok(ds)) = (st.tensor(&down_blocks_name), st.tensor(&down_scales_name)) {
            let down_shape = db.shape();
            if down_shape.len() == 4 {
                let down_out = down_shape[1];
                let down_groups = down_shape[2];
                let down_in = down_groups * 32;
                let should_load_down = (0..num_experts)
                    .any(|e| !skip_key(&mxfp4_expert_key(layer_prefix, e, MIXTRAL_DOWN_PROJ)));
                if should_load_down {
                    let down_experts = crate::quant::mxfp4::dequantize_all_experts(
                        db.data(),
                        ds.data(),
                        num_experts,
                        down_out,
                        down_groups,
                    )?;
                    for (e, data) in down_experts.into_iter().enumerate() {
                        let down_key = mxfp4_expert_key(layer_prefix, e, MIXTRAL_DOWN_PROJ);
                        if !skip_key(&down_key) {
                            tensors.insert(
                                down_key,
                                Array2::from_shape_vec((down_out, down_in), data)
                                    .map_err(|e| ModelError::Parse(e.to_string()))?
                                    .into_shared(),
                            );
                        }
                    }
                }
            }
        }

        // Remap router: mlp.router.weight → block_sparse_moe.gate.weight
        let router_name = name.replace(MXFP4_EXPERTS_GATE_UP_BLOCKS, MXFP4_ROUTER_WEIGHT);
        if let Ok(router_view) = st.tensor(&router_name) {
            if let Ok(data) = tensor_to_f32(&router_view) {
                let s = router_view.shape();
                if s.len() == 2 {
                    let router_key = format!("{layer_prefix}.{BLOCK_SPARSE_ROUTER_WEIGHT}");
                    if !skip_key(&router_key) {
                        tensors.insert(
                            router_key,
                            Array2::from_shape_vec((s[0], s[1]), data)
                                .map_err(|e| ModelError::Parse(e.to_string()))?
                                .into_shared(),
                        );
                    }
                }
            }
        }
    }

    Ok(())
}

/// Key for one dequantised MXFP4 expert projection.
///
/// `layer_prefix` here comes from splitting a packed tensor name at `.mlp.`,
/// so it has no trailing separator; the shared builder expects one, the way
/// [`ModelArchitecture::layer_prefix`](crate::ModelArchitecture::layer_prefix)
/// supplies it. The convention itself lives in
/// [`crate::tensor_keys::mxfp4_dequantised`] so the architecture that
/// advertises these keys and the loader that writes them cannot drift.
fn mxfp4_expert_key(layer_prefix: &str, expert_id: usize, projection: &str) -> String {
    crate::tensor_keys::mxfp4_dequantised::projection(
        &format!("{layer_prefix}."),
        expert_id,
        projection,
    )
}

/// Per-expert MXFP4 dequantization (DeepSeek-V4 family).
///
/// DeepSeek-V4 stores expert weights one (.weight, .scale) pair per
/// (expert, projection) — `layers.X.ffn.experts.E.w1.weight` (I8 packed FP4) +
/// `layers.X.ffn.experts.E.w1.scale` (F8_E8M0 scales), ditto w2/w3. This is
/// distinct from GPT-OSS's fused `experts.gate_up_proj_blocks` layout that
/// `load_mxfp4_expert_tensors` handles.
///
/// Detects the format by scanning for `*.experts.<digit>.w[123].weight` tensors
/// with `I8` dtype. For each match, looks up the companion `.scale` (`F8_E8M0`)
/// and dequantizes via `quant::mxfp4::dequantize_expert`.
///
/// Returns the set of tensor names that were consumed (both `.weight` and
/// `.scale`) so the main loading loop can skip them.
pub(super) fn dequantize_per_expert_mxfp4(
    st: &safetensors::SafeTensors,
    tensor_names: &[String],
    prefixes: &[&str],
    tensors: &mut HashMap<String, crate::WeightArray>,
) -> Result<std::collections::HashSet<String>, ModelError> {
    use std::collections::HashSet;
    let mut consumed: HashSet<String> = HashSet::new();

    // Match V4-style per-expert weights: any tensor name containing
    // ".experts.<int>.w<1|2|3>.weight" — broad enough to catch both the
    // full `model.layers.X.ffn.experts.E.wY.weight` (HF default) and any
    // shortened variant (`layers.X.ffn.experts.E.wY.weight`).
    let is_v4_expert_weight = |name: &str| -> bool {
        if !name.ends_with(".w1.weight")
            && !name.ends_with(".w2.weight")
            && !name.ends_with(".w3.weight")
        {
            return false;
        }
        // Must have ".experts.<digit>" before the .wN.weight suffix
        if let Some(idx) = name.rfind(".experts.") {
            let after = &name[idx + ".experts.".len()..];
            if let Some(dot) = after.find('.') {
                return after[..dot].chars().all(|c| c.is_ascii_digit());
            }
        }
        false
    };

    for name in tensor_names {
        if !is_v4_expert_weight(name) {
            continue;
        }

        let weight_view = match st.tensor(name) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // V4 packed FP4 weights are stored as I8 (signed) per the safetensors header.
        if weight_view.dtype() != safetensors::Dtype::I8 {
            continue;
        }

        let scale_name = name.replacen(".weight", ".scale", 1);
        let scale_view = match st.tensor(&scale_name) {
            Ok(v) => v,
            Err(_) => continue, // No scale companion → not MXFP4, leave to main loop.
        };
        if scale_view.dtype() != safetensors::Dtype::F8_E8M0 {
            continue;
        }

        // Shape sanity. weight: (out_features, packed_in/2). scale: (out_features, groups).
        let w_shape = weight_view.shape();
        let s_shape = scale_view.shape();
        if w_shape.len() != 2 || s_shape.len() != 2 {
            continue;
        }
        if w_shape[0] != s_shape[0] {
            continue;
        }

        let out_features = w_shape[0];
        let groups = s_shape[1];
        let in_features = groups * 32;

        // Assert layout consistency: weight cols × 2 (nibbles per byte) == groups × 32.
        if w_shape[1] * 2 != in_features {
            continue;
        }

        let unpacked = crate::quant::mxfp4::dequantize_expert(
            weight_view.data(),
            scale_view.data(),
            out_features,
            groups,
        )?;

        let key = normalize_key(name, prefixes);
        let arr = Array2::from_shape_vec((out_features, in_features), unpacked)
            .map_err(|e| ModelError::Parse(e.to_string()))?;
        tensors.insert(key, arr.into_shared());

        consumed.insert(name.clone());
        consumed.insert(scale_name);
    }

    Ok(consumed)
}
