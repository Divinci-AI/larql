//! Model loading — safetensors, MLX, GGUF → ModelWeights.
//!
//! Handles dtype conversion (f16, bf16 → f32), HuggingFace cache resolution,
//! and architecture detection.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::Array2;

use crate::weights::ModelWeights;
use crate::detect::ModelError;

/// Returns true when `key` names a FFN weight tensor (gate/up/down projection
/// or packed expert block). Used by `load_model_dir_walk_only` to skip
/// decoding these entirely — critical for large models where decoding them
/// into f32 heap would blow RAM before they can be dropped.
pub fn is_ffn_tensor(key: &str) -> bool {
    let ffn_patterns = ["gate_proj", "up_proj", "down_proj",
                       "ffn_gate", "ffn_up", "ffn_down",
                       "mlp.experts", "block_sparse_moe.experts",
                       "packed_gate_up_blocks", "packed_down_blocks"];
    ffn_patterns.iter().any(|p| key.contains(p))
}

/// Load model weights from a directory or file, never reading FFN tensors.
///
/// Equivalent to `load_model_dir` + `drop_ffn_weights` but without the heap
/// spike: FFN tensors are skipped at deserialisation time, so peak RSS
/// tracks only the retained (attention / embed / lm_head / norms) weights.
/// Use this with vindex-backed FFN (walk-only inference).
pub fn load_model_dir_walk_only(path: impl AsRef<Path>) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered(path, |k| is_ffn_tensor(k))
}

/// Load model weights from a directory or file.
///
/// Auto-detects the format:
/// - Directory with `.safetensors` files → safetensors loading
/// - Directory with `.gguf` file → GGUF loading (dequantized to f32)
/// - Single `.gguf` file → GGUF loading
///
/// Detects architecture from config.json (safetensors) or GGUF metadata.
pub fn load_model_dir(path: impl AsRef<Path>) -> Result<ModelWeights, ModelError> {
    load_model_dir_filtered(path, |_| false)
}

/// Same as `load_model_dir` but `skip_key` returning true causes a tensor to
/// be dropped before decode — its bytes are never read from the mmap and no
/// f32 heap allocation occurs for it.
pub fn load_model_dir_filtered(
    path: impl AsRef<Path>,
    skip_key: impl Fn(&str) -> bool,
) -> Result<ModelWeights, ModelError> {
    let path = path.as_ref();

    // Single GGUF file
    if path.is_file() {
        if path.extension().is_some_and(|ext| ext == "gguf") {
            return super::gguf::load_gguf(path);
        }
        return Err(ModelError::NotADirectory(path.to_path_buf()));
    }

    if !path.is_dir() {
        return Err(ModelError::NotADirectory(path.to_path_buf()));
    }

    // Check for GGUF files in directory
    let gguf_files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "gguf"))
        .collect();

    if !gguf_files.is_empty() {
        // Use the first (or largest) GGUF file
        let gguf_path = gguf_files.into_iter()
            .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .unwrap();
        return super::gguf::load_gguf(&gguf_path);
    }

    // Safetensors loading (also handles MLX format — same files, sometimes in weights/ subdir)
    let arch = crate::detect_architecture(path)
        .map_err(|e| ModelError::Parse(e.to_string()))?;
    let prefixes = arch.key_prefixes_to_strip();

    let mut st_files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();

    // MLX models sometimes put weights in a weights/ subdirectory
    if st_files.is_empty() {
        let weights_dir = path.join("weights");
        if weights_dir.is_dir() {
            st_files = std::fs::read_dir(&weights_dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
                .collect();
        }
    }
    st_files.sort();

    if st_files.is_empty() {
        return Err(ModelError::NoSafetensors(path.to_path_buf()));
    }

    let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut raw_bytes: HashMap<String, Vec<u8>> = HashMap::new();

    let expert_format = arch.expert_format();
    let is_packed_mxfp4 = expert_format == crate::ExpertFormat::PackedMxfp4;
    let is_packed_bf16 = expert_format == crate::ExpertFormat::PackedBF16;

    // Keys that must be preserved as raw bytes rather than converted to f32.
    // For PackedBF16 (Gemma 4 26B A4B): experts.gate_up_proj and experts.down_proj
    // are 3D tensors [num_experts, out_dim, in_dim] in BF16. Converting them to f32
    // would double their memory footprint; the compute path dequantizes per-expert on demand.
    let should_keep_raw = |key: &str| -> bool {
        is_packed_bf16 && (key.contains("experts.gate_up_proj") || key.contains("experts.down_proj"))
    };

    for st_path in &st_files {
        let file = std::fs::File::open(st_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| ModelError::Parse(e.to_string()))?;

        // Check for MXFP4 packed expert tensors (GPT-OSS format)
        let tensor_names: Vec<String> = st.names().iter().map(|n| n.to_string()).collect();

        if is_packed_mxfp4 {
            // MXFP4 path: dequantize packed expert blocks+scales into per-expert tensors
            dequantize_mxfp4_experts(&st, &tensor_names, prefixes, &mut tensors, &mut vectors)?;
            // Also load normal float tensors (router, norms, attn, embeddings)
            for (name, view) in st.tensors() {
                let key = normalize_key(&name, prefixes);
                let shape = view.shape();
                if name.ends_with("_blocks") || name.ends_with("_scales") { continue; }
                if skip_key(&key) { continue; }
                let data = match tensor_to_f32(&view) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                match shape.len() {
                    2 => {
                        let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                            .map_err(|e| ModelError::Parse(e.to_string()))?;
                        tensors.insert(key, arr.into_shared());
                    }
                    1 => { vectors.insert(key, data); }
                    _ => {}
                }
            }
        } else {
            // Per-expert MXFP4 path (DeepSeek-V4 family): each expert's gate/up/down
            // is stored as a separate (.weight, .scale) pair instead of GPT-OSS's
            // fused gate_up_proj_blocks + scales tensors. Detect by looking for
            // an `experts.<digit>.w[123].weight` tensor with an I8 dtype.
            //
            // We dequantize these per-expert and skip the corresponding I8/.scale
            // tensors below in the main loop.
            let v4_dequantized_keys = dequantize_per_expert_mxfp4(
                &st, &tensor_names, prefixes, &mut tensors,
            )?;

            for (name, view) in st.tensors() {
                let key = normalize_key(&name, prefixes);
                let shape = view.shape();
                if skip_key(&key) { continue; }

                // Skip tensors that the V4 per-expert MXFP4 dequantizer already produced
                // (or whose .weight/.scale companion was consumed by it).
                if v4_dequantized_keys.contains(&name) { continue; }

                // PackedBF16 expert tensors: preserve raw bytes, skip f32 conversion
                if should_keep_raw(&key) {
                    raw_bytes.insert(key, view.data().to_vec());
                    continue;
                }

                let data = match tensor_to_f32(&view) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                match shape.len() {
                    2 => {
                        let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                            .map_err(|e| ModelError::Parse(e.to_string()))?;
                        tensors.insert(key, arr.into_shared());
                    }
                    1 => { vectors.insert(key, data); }
                    // 0D scalar tensors (e.g., layer_scalar) → store as 1-element vector
                    0 => { vectors.insert(key, data); }
                    _ => {}
                }
            }
        }
    }

    let embed_key = arch.embed_key();
    let embed = tensors
        .get(embed_key)
        .ok_or_else(|| ModelError::MissingTensor(embed_key.into()))?
        .clone();

    let lm_head = tensors
        .get("lm_head.weight")
        .cloned()
        .unwrap_or_else(|| embed.clone());

    let vocab_size = lm_head.shape()[0];
    let cfg = arch.config();

    Ok(ModelWeights {
        tensors,
        vectors,
        raw_bytes,
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed,
        lm_head,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}

/// Resolve a HuggingFace model ID or path to a local directory or GGUF file.
pub fn resolve_model_path(model: &str) -> Result<PathBuf, ModelError> {
    let path = PathBuf::from(model);
    if path.is_dir() {
        return Ok(path);
    }
    // Single GGUF file
    if path.is_file() && path.extension().is_some_and(|ext| ext == "gguf") {
        return Ok(path);
    }

    // Try HuggingFace cache
    let cache_name = format!("models--{}", model.replace('/', "--"));
    let home = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    let hf_cache = home.join(format!(".cache/huggingface/hub/{cache_name}/snapshots"));

    if hf_cache.is_dir() {
        // Find the snapshot that has actual model files (safetensors or config.json+weights)
        let mut best: Option<PathBuf> = None;
        if let Ok(entries) = std::fs::read_dir(&hf_cache) {
            for entry in entries.flatten() {
                let p = entry.path();
                if !p.is_dir() { continue; }
                // Prefer snapshot with safetensors files
                let has_st = std::fs::read_dir(&p).ok().map(|rd| {
                    rd.flatten().any(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
                }).unwrap_or(false);
                if has_st {
                    return Ok(p);
                }
                // Fallback: any snapshot with config.json
                if p.join("config.json").exists() {
                    best = Some(p);
                }
            }
        }
        if let Some(p) = best {
            return Ok(p);
        }
    }

    Err(ModelError::NotADirectory(path))
}

/// Normalize a tensor key by stripping known prefixes.
pub fn normalize_key_pub(key: &str, prefixes: &[&str]) -> String {
    normalize_key(key, prefixes)
}

/// Dequantize MXFP4 packed expert tensors into per-expert standard weight matrices.
///
/// GPT-OSS stores experts as:
///   layers.{L}.mlp.experts.gate_up_proj_blocks: [experts, 2*hidden, groups, 16] U8
///   layers.{L}.mlp.experts.gate_up_proj_scales: [experts, 2*hidden, groups] U8
///   layers.{L}.mlp.experts.down_proj_blocks: [experts, hidden, groups, 16] U8
///   layers.{L}.mlp.experts.down_proj_scales: [experts, hidden, groups] U8
///
/// We dequantize and split into per-expert Mixtral-style keys:
///   layers.{L}.block_sparse_moe.experts.{E}.w1.weight (gate)
///   layers.{L}.block_sparse_moe.experts.{E}.w3.weight (up)
///   layers.{L}.block_sparse_moe.experts.{E}.w2.weight (down)
fn dequantize_mxfp4_experts(
    st: &safetensors::SafeTensors,
    tensor_names: &[String],
    prefixes: &[&str],
    tensors: &mut HashMap<String, crate::WeightArray>,
    _vectors: &mut HashMap<String, Vec<f32>>,
) -> Result<(), ModelError> {
    // Find all gate_up_proj_blocks tensors (one per layer)
    for name in tensor_names {
        if !name.ends_with(".gate_up_proj_blocks") { continue; }

        let scales_name = name.replace("_blocks", "_scales");
        let down_blocks_name = name.replace("gate_up_proj_blocks", "down_proj_blocks");
        let down_scales_name = name.replace("gate_up_proj_blocks", "down_proj_scales");

        // Get tensor views
        let blocks_view = st.tensor(name)
            .map_err(|e| ModelError::Parse(format!("MXFP4 blocks: {e}")))?;
        let scales_view = st.tensor(&scales_name)
            .map_err(|e| ModelError::Parse(format!("MXFP4 scales: {e}")))?;

        let shape = blocks_view.shape();
        if shape.len() != 4 { continue; }

        let num_experts = shape[0];
        let out_features = shape[1]; // 2*hidden for gate_up, hidden for down
        let groups = shape[2];
        let in_features = groups * 32; // 16 bytes * 2 nibbles per group
        let _hidden = in_features; // = hidden_size

        // Dequantize gate_up (fused: first half = gate, second half = up)
        let expert_data = crate::quant::mxfp4::dequantize_all_experts(
            blocks_view.data(), scales_view.data(),
            num_experts, out_features, groups,
        );

        // Extract layer number from key
        let base_key = normalize_key(name, prefixes);
        let layer_prefix = base_key.split(".mlp.").next().unwrap_or("");

        let half = out_features / 2; // gate vs up split

        for (e, data) in expert_data.iter().enumerate() {
            // Split fused gate_up: rows [0..half] = gate (w1), rows [half..] = up (w3)
            let gate_data: Vec<f32> = data[..half * in_features].to_vec();
            let up_data: Vec<f32> = data[half * in_features..].to_vec();

            let gate_key = format!("{layer_prefix}.block_sparse_moe.experts.{e}.w1.weight");
            let up_key = format!("{layer_prefix}.block_sparse_moe.experts.{e}.w3.weight");

            tensors.insert(gate_key,
                Array2::from_shape_vec((half, in_features), gate_data)
                    .map_err(|e| ModelError::Parse(e.to_string()))?.into_shared());
            tensors.insert(up_key,
                Array2::from_shape_vec((half, in_features), up_data)
                    .map_err(|e| ModelError::Parse(e.to_string()))?.into_shared());
        }

        // Dequantize down projection
        if let (Ok(db), Ok(ds)) = (st.tensor(&down_blocks_name), st.tensor(&down_scales_name)) {
            let down_shape = db.shape();
            if down_shape.len() == 4 {
                let down_out = down_shape[1];
                let down_groups = down_shape[2];
                let down_in = down_groups * 32;

                let down_experts = crate::quant::mxfp4::dequantize_all_experts(
                    db.data(), ds.data(), num_experts, down_out, down_groups,
                );

                for (e, data) in down_experts.iter().enumerate() {
                    let down_key = format!("{layer_prefix}.block_sparse_moe.experts.{e}.w2.weight");
                    tensors.insert(down_key,
                        Array2::from_shape_vec((down_out, down_in), data.clone())
                            .map_err(|e| ModelError::Parse(e.to_string()))?.into_shared());
                }
            }
        }

        // Also remap router: mlp.router.weight → block_sparse_moe.gate.weight
        let router_name = name.replace("experts.gate_up_proj_blocks", "router.weight");
        if let Ok(router_view) = st.tensor(&router_name) {
            if let Ok(data) = tensor_to_f32(&router_view) {
                let s = router_view.shape();
                if s.len() == 2 {
                    let router_key = format!("{layer_prefix}.block_sparse_moe.gate.weight");
                    tensors.insert(router_key,
                        Array2::from_shape_vec((s[0], s[1]), data)
                            .map_err(|e| ModelError::Parse(e.to_string()))?.into_shared());
                }
            }
        }
    }

    Ok(())
}

/// Per-expert MXFP4 dequantization (DeepSeek-V4 family).
///
/// DeepSeek-V4 stores expert weights one (.weight, .scale) pair per
/// (expert, projection) — `layers.X.ffn.experts.E.w1.weight` (I8 packed FP4) +
/// `layers.X.ffn.experts.E.w1.scale` (F8_E8M0 scales), ditto w2/w3. This is
/// distinct from GPT-OSS's fused `experts.gate_up_proj_blocks` layout that
/// `dequantize_mxfp4_experts` handles.
///
/// Detects the format by scanning for `*.experts.<digit>.w[123].weight` tensors
/// with `I8` dtype. For each match, looks up the companion `.scale` (`F8_E8M0`)
/// and dequantizes via `quant::mxfp4::dequantize_expert`.
///
/// Returns the set of tensor names that were consumed (both `.weight` and
/// `.scale`) so the main loading loop can skip them.
fn dequantize_per_expert_mxfp4(
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
        if !name.ends_with(".w1.weight") && !name.ends_with(".w2.weight") && !name.ends_with(".w3.weight") {
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
        if !is_v4_expert_weight(name) { continue; }

        let weight_view = match st.tensor(name) {
            Ok(v) => v,
            Err(_) => continue,
        };
        // V4 packed FP4 weights are stored as I8 (signed) per the safetensors header.
        if weight_view.dtype() != safetensors::Dtype::I8 { continue; }

        let scale_name = name.replacen(".weight", ".scale", 1);
        let scale_view = match st.tensor(&scale_name) {
            Ok(v) => v,
            Err(_) => continue, // No scale companion → not MXFP4, leave to main loop.
        };
        if scale_view.dtype() != safetensors::Dtype::F8_E8M0 { continue; }

        // Shape sanity. weight: (out_features, packed_in/2). scale: (out_features, groups).
        let w_shape = weight_view.shape();
        let s_shape = scale_view.shape();
        if w_shape.len() != 2 || s_shape.len() != 2 { continue; }
        if w_shape[0] != s_shape[0] { continue; }

        let out_features = w_shape[0];
        let groups = s_shape[1];
        let in_features = groups * 32;

        // Assert layout consistency: weight cols × 2 (nibbles per byte) == groups × 32.
        if w_shape[1] * 2 != in_features { continue; }

        let unpacked = crate::quant::mxfp4::dequantize_expert(
            weight_view.data(),
            scale_view.data(),
            out_features,
            groups,
        );

        let key = normalize_key(name, prefixes);
        let arr = Array2::from_shape_vec((out_features, in_features), unpacked)
            .map_err(|e| ModelError::Parse(e.to_string()))?;
        tensors.insert(key, arr.into_shared());

        consumed.insert(name.clone());
        consumed.insert(scale_name);
    }

    Ok(consumed)
}

fn normalize_key(key: &str, prefixes: &[&str]) -> String {
    for prefix in prefixes {
        if let Some(stripped) = key.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    key.to_string()
}

fn tensor_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>, ModelError> {
    use crate::quant::half;
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let bytes = view.data();
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        safetensors::Dtype::F16 => Ok(half::decode_f16(view.data())),
        safetensors::Dtype::BF16 => Ok(half::decode_bf16(view.data())),

        // ── FP8 / I8 — used by DeepSeek-V4 (MXFP4 experts), GPT-OSS, etc. ──
        // Decoded bit-pattern → f32 in isolation. MXFP4 unpacking proper (where
        // an I8 packed-nibble weight is paired with its F8_E8M0 scale companion)
        // happens at the FFN tensor loading layer — `tensor_to_f32` sees one
        // tensor at a time and can't look at companions.
        safetensors::Dtype::F8_E4M3 => Ok(decode_f8_e4m3(view.data())),
        safetensors::Dtype::F8_E5M2 => Ok(decode_f8_e5m2(view.data())),
        safetensors::Dtype::F8_E8M0 => Ok(decode_f8_e8m0(view.data())),
        safetensors::Dtype::I8 => Ok(view.data().iter().map(|&b| (b as i8) as f32).collect()),

        other => Err(ModelError::UnsupportedDtype(format!("{other:?}"))),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// FP8 / E8M0 decoders — bit-pattern → f32. Operate per-byte on the raw view.
// Standard Open Compute Project encodings; verified against the F8_E*M* table
// in the safetensors crate (≥ 0.7).
// ────────────────────────────────────────────────────────────────────────────

/// FP8 E4M3 (FN, finite-only): 1 sign + 4 exponent + 3 mantissa bits, bias 7.
/// NaN encoded at 0x7F / 0xFF (Open Compute convention).
#[inline]
fn decode_f8_e4m3(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| {
        let sign = (b >> 7) & 1;
        let exp_bits = (b >> 3) & 0x0F;
        let mant_bits = b & 0x07;
        let v = if exp_bits == 0 {
            (mant_bits as f32) / 8.0 * 2f32.powi(1 - 7)
        } else if exp_bits == 0x0F && mant_bits == 0x07 {
            f32::NAN
        } else {
            let m = 1.0 + (mant_bits as f32) / 8.0;
            m * 2f32.powi(exp_bits as i32 - 7)
        };
        if sign == 1 { -v } else { v }
    }).collect()
}

/// FP8 E5M2: 1 sign + 5 exponent + 2 mantissa bits, bias 15.
#[inline]
fn decode_f8_e5m2(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| {
        let sign = (b >> 7) & 1;
        let exp_bits = (b >> 2) & 0x1F;
        let mant_bits = b & 0x03;
        let v = if exp_bits == 0 {
            (mant_bits as f32) / 4.0 * 2f32.powi(1 - 15)
        } else if exp_bits == 0x1F {
            if mant_bits == 0 { f32::INFINITY } else { f32::NAN }
        } else {
            let m = 1.0 + (mant_bits as f32) / 4.0;
            m * 2f32.powi(exp_bits as i32 - 15)
        };
        if sign == 1 { -v } else { v }
    }).collect()
}

/// FP8 E8M0 (Open Compute Microscaling MX format scale): 8 exponent bits, no
/// sign or mantissa. Value = 2^(byte - 127). Byte 0xFF reserved as NaN.
#[inline]
fn decode_f8_e8m0(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|&b| {
        if b == 0xFF { f32::NAN } else { 2f32.powi(b as i32 - 127) }
    }).collect()
}
