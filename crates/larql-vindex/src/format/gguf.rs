//! GGUF format reader — parse GGUF files and load tensors as f32.
//!
//! GGUF is the GGML Universal Format used by llama.cpp.
//! We support reading unquantized (F32, F16, BF16) and quantized (Q4_0, Q4_1, Q8_0) tensors.
//! All tensors are dequantized to f32 for use with ModelWeights.

use std::collections::HashMap;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use ndarray::Array2;

use larql_models::ModelWeights;
use crate::error::VindexError;

// ═══════════════════════════════════════════════════════════════
// GGUF constants
// ═══════════════════════════════════════════════════════════════

const GGUF_MAGIC: u32 = 0x46475547; // "GGUF"

// Metadata value types
const GGUF_TYPE_UINT8: u32 = 0;
const GGUF_TYPE_INT8: u32 = 1;
const GGUF_TYPE_UINT16: u32 = 2;
const GGUF_TYPE_INT16: u32 = 3;
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_UINT64: u32 = 10;
const GGUF_TYPE_INT64: u32 = 11;
const GGUF_TYPE_FLOAT64: u32 = 12;

// Tensor types we support
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q4_1: u32 = 3;
const GGML_TYPE_Q8_0: u32 = 6;
const GGML_TYPE_Q5_0: u32 = 8;
const GGML_TYPE_Q5_1: u32 = 9;
const GGML_TYPE_BF16: u32 = 30;

// ═══════════════════════════════════════════════════════════════
// GGUF metadata value
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::F32(v) => Some(*v as f64),
            GgufValue::F64(v) => Some(*v),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// GGUF tensor info
// ═══════════════════════════════════════════════════════════════

pub struct GgufTensorInfo {
    name: String,
    n_dims: u32,
    dims: Vec<u64>,
    tensor_type: u32,
    offset: u64,
}

// ═══════════════════════════════════════════════════════════════
// GGUF reader
// ═══════════════════════════════════════════════════════════════

pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensor_infos: Vec<GgufTensorInfo>,
    pub data_offset: u64,
    pub path: std::path::PathBuf,
}

impl GgufFile {
    /// Parse a GGUF file header and tensor info (does not read tensor data yet).
    pub fn open(path: &Path) -> Result<Self, VindexError> {
        let file = std::fs::File::open(path)?;
        let mut r = BufReader::new(file);

        // Magic
        let magic = read_u32(&mut r)?;
        if magic != GGUF_MAGIC {
            return Err(VindexError::Parse(format!(
                "not a GGUF file (magic: 0x{:08X}, expected 0x{:08X})", magic, GGUF_MAGIC
            )));
        }

        // Version
        let version = read_u32(&mut r)?;
        if version < 2 || version > 3 {
            return Err(VindexError::Parse(format!("unsupported GGUF version: {version}")));
        }

        let n_tensors = read_u64(&mut r)? as usize;
        let n_metadata = read_u64(&mut r)? as usize;

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_metadata {
            let key = read_string(&mut r)?;
            let value = read_value(&mut r)?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let mut tensor_infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_string(&mut r)?;
            let n_dims = read_u32(&mut r)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut r)?);
            }
            let tensor_type = read_u32(&mut r)?;
            let offset = read_u64(&mut r)?;
            tensor_infos.push(GgufTensorInfo { name, n_dims, dims, tensor_type, offset });
        }

        // Data starts at next alignment boundary (32 bytes)
        let pos = r.stream_position()
            .map_err(|e| VindexError::Io(e))?;
        let alignment = 32u64;
        let data_offset = (pos + alignment - 1) / alignment * alignment;

        Ok(GgufFile {
            metadata,
            tensor_infos,
            data_offset,
            path: path.to_path_buf(),
        })
    }

    /// Load all tensors, dequantizing to f32.
    pub fn load_tensors(&self) -> Result<(HashMap<String, Array2<f32>>, HashMap<String, Vec<f32>>), VindexError> {
        let file = std::fs::File::open(&self.path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();

        for info in &self.tensor_infos {
            let abs_offset = self.data_offset + info.offset;
            let n_elements: u64 = info.dims.iter().product();

            let data_size = tensor_data_size(info.tensor_type, n_elements as usize)?;
            if abs_offset as usize + data_size > mmap.len() {
                return Err(VindexError::Parse(format!(
                    "tensor {} data out of bounds (offset {} + size {} > file {})",
                    info.name, abs_offset, data_size, mmap.len()
                )));
            }

            let raw = &mmap[abs_offset as usize..abs_offset as usize + data_size];
            let floats = dequantize(raw, info.tensor_type, n_elements as usize)?;

            // Normalize key name (strip GGUF prefixes)
            let key = normalize_gguf_key(&info.name);

            match info.n_dims {
                2 => {
                    // GGUF stores in row-major, dims[0] = rows, dims[1] = cols
                    let rows = info.dims[0] as usize;
                    let cols = info.dims[1] as usize;
                    let arr = Array2::from_shape_vec((rows, cols), floats)
                        .map_err(|e| VindexError::Parse(format!("tensor {}: {}", info.name, e)))?;
                    tensors.insert(key, arr);
                }
                1 => {
                    vectors.insert(key, floats);
                }
                _ => {} // skip higher-dim tensors
            }
        }

        Ok((tensors, vectors))
    }

    /// Build a config.json-equivalent from GGUF metadata for architecture detection.
    pub fn to_config_json(&self) -> serde_json::Value {
        let get_str = |k: &str| self.metadata.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let _get_u32 = |k: &str| self.metadata.get(k).and_then(|v| v.as_u32()).unwrap_or(0);

        // GGUF uses "general.architecture" and "{arch}.*" keys
        let arch = get_str("general.architecture");
        let prefix = format!("{arch}.");

        let get_arch_u32 = |suffix: &str| {
            self.metadata.get(&format!("{prefix}{suffix}"))
                .and_then(|v| v.as_u32())
                .unwrap_or(0)
        };
        let get_arch_f64 = |suffix: &str| {
            self.metadata.get(&format!("{prefix}{suffix}"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0)
        };

        // Map GGUF architecture names to HF model_type
        let model_type = match arch.as_str() {
            "llama" => "llama",
            "gemma" | "gemma2" | "gemma3" => &arch,
            "qwen" | "qwen2" => "qwen2",
            "mistral" => "mistral",
            "mixtral" => "mixtral",
            "phi" | "phi2" | "phi3" => "phi",
            "gpt2" => "gpt2",
            "deepseek" | "deepseek2" => "deepseek_v2",
            other => other,
        };

        serde_json::json!({
            "model_type": model_type,
            "hidden_size": get_arch_u32("embedding_length"),
            "num_hidden_layers": get_arch_u32("block_count"),
            "intermediate_size": get_arch_u32("feed_forward_length"),
            "num_attention_heads": get_arch_u32("attention.head_count"),
            "num_key_value_heads": get_arch_u32("attention.head_count_kv"),
            "head_dim": get_arch_u32("attention.key_length"),
            "rope_theta": get_arch_f64("rope.freq_base"),
            "vocab_size": get_arch_u32("vocab_size"),
        })
    }
}

/// Load a GGUF file into ModelWeights (dequantized to f32).
pub fn load_gguf(path: &Path) -> Result<ModelWeights, VindexError> {
    let gguf = GgufFile::open(path)?;

    // Detect architecture from GGUF metadata
    let config_json = gguf.to_config_json();
    let arch = larql_models::detect_from_json(&config_json);
    let prefixes = arch.key_prefixes_to_strip();

    // Load and dequantize all tensors
    let (mut tensors, vectors) = gguf.load_tensors()?;

    // Re-normalize keys through the architecture's prefix stripping
    let mut normalized_tensors = HashMap::new();
    for (k, v) in tensors.drain() {
        let key = super::loader::normalize_key_pub(&k, prefixes);
        normalized_tensors.insert(key, v);
    }

    let embed_key = arch.embed_key();
    let embed = normalized_tensors
        .get(embed_key)
        .ok_or_else(|| VindexError::MissingTensor(embed_key.into()))?
        .clone();

    let lm_head = normalized_tensors
        .get("lm_head.weight")
        .or_else(|| normalized_tensors.get("output.weight"))
        .cloned()
        .unwrap_or_else(|| embed.clone());

    let vocab_size = lm_head.shape()[0];
    let cfg = arch.config();

    Ok(ModelWeights {
        tensors: normalized_tensors,
        vectors,
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

// ═══════════════════════════════════════════════════════════════
// GGUF binary reading helpers
// ═══════════════════════════════════════════════════════════════

fn read_u8(r: &mut impl Read) -> Result<u8, VindexError> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(r: &mut impl Read) -> Result<i8, VindexError> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut impl Read) -> Result<u16, VindexError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(r: &mut impl Read) -> Result<i16, VindexError> {
    Ok(read_u16(r)? as i16)
}

fn read_u32(r: &mut impl Read) -> Result<u32, VindexError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> Result<i32, VindexError> {
    Ok(read_u32(r)? as i32)
}

fn read_u64(r: &mut impl Read) -> Result<u64, VindexError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> Result<i64, VindexError> {
    Ok(read_u64(r)? as i64)
}

fn read_f32(r: &mut impl Read) -> Result<f32, VindexError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> Result<f64, VindexError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string(r: &mut impl Read) -> Result<String, VindexError> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| VindexError::Parse(e.to_string()))
}

fn read_value(r: &mut impl Read) -> Result<GgufValue, VindexError> {
    let vtype = read_u32(r)?;
    match vtype {
        GGUF_TYPE_UINT8 => Ok(GgufValue::U8(read_u8(r)?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::I8(read_i8(r)?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::U16(read_u16(r)?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::I16(read_i16(r)?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::U32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::I32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(r)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(read_string(r)?)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::U64(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::I64(read_i64(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(r)?)),
        GGUF_TYPE_ARRAY => {
            let elem_type = read_u32(r)?;
            let len = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_array_element(r, elem_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        _ => Err(VindexError::Parse(format!("unknown GGUF metadata type: {vtype}"))),
    }
}

fn read_array_element(r: &mut impl Read, elem_type: u32) -> Result<GgufValue, VindexError> {
    match elem_type {
        GGUF_TYPE_UINT8 => Ok(GgufValue::U8(read_u8(r)?)),
        GGUF_TYPE_INT8 => Ok(GgufValue::I8(read_i8(r)?)),
        GGUF_TYPE_UINT16 => Ok(GgufValue::U16(read_u16(r)?)),
        GGUF_TYPE_INT16 => Ok(GgufValue::I16(read_i16(r)?)),
        GGUF_TYPE_UINT32 => Ok(GgufValue::U32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(GgufValue::I32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(GgufValue::F32(read_f32(r)?)),
        GGUF_TYPE_BOOL => Ok(GgufValue::Bool(read_u8(r)? != 0)),
        GGUF_TYPE_STRING => Ok(GgufValue::String(read_string(r)?)),
        GGUF_TYPE_UINT64 => Ok(GgufValue::U64(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(GgufValue::I64(read_i64(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(GgufValue::F64(read_f64(r)?)),
        _ => Err(VindexError::Parse(format!("unknown GGUF array element type: {elem_type}"))),
    }
}

// ═══════════════════════════════════════════════════════════════
// Dequantization
// ═══════════════════════════════════════════════════════════════

fn tensor_data_size(tensor_type: u32, n_elements: usize) -> Result<usize, VindexError> {
    match tensor_type {
        GGML_TYPE_F32 => Ok(n_elements * 4),
        GGML_TYPE_F16 | GGML_TYPE_BF16 => Ok(n_elements * 2),
        GGML_TYPE_Q4_0 => Ok(n_elements / 32 * 18),      // 32 elements per block, 18 bytes per block
        GGML_TYPE_Q4_1 => Ok(n_elements / 32 * 20),      // 32 elements per block, 20 bytes per block
        GGML_TYPE_Q5_0 => Ok(n_elements / 32 * 22),      // 32 elements per block, 22 bytes per block
        GGML_TYPE_Q5_1 => Ok(n_elements / 32 * 24),      // 32 elements per block, 24 bytes per block
        GGML_TYPE_Q8_0 => Ok(n_elements / 32 * 34),      // 32 elements per block, 34 bytes per block
        other => Err(VindexError::UnsupportedDtype(format!("GGML type {other}"))),
    }
}

fn dequantize(data: &[u8], tensor_type: u32, n_elements: usize) -> Result<Vec<f32>, VindexError> {
    match tensor_type {
        GGML_TYPE_F32 => {
            Ok(data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        GGML_TYPE_F16 => {
            Ok(data.chunks_exact(2)
                .map(|b| super::loader::half_to_f32_pub(u16::from_le_bytes([b[0], b[1]])))
                .collect())
        }
        GGML_TYPE_BF16 => {
            Ok(data.chunks_exact(2)
                .map(|b| super::loader::bf16_to_f32_pub(u16::from_le_bytes([b[0], b[1]])))
                .collect())
        }
        GGML_TYPE_Q4_0 => dequantize_q4_0(data, n_elements),
        GGML_TYPE_Q4_1 => dequantize_q4_1(data, n_elements),
        GGML_TYPE_Q8_0 => dequantize_q8_0(data, n_elements),
        other => Err(VindexError::UnsupportedDtype(format!("GGML type {other}"))),
    }
}

/// Q4_0: 32 elements per block. Block = f16 scale (2 bytes) + 16 bytes of 4-bit quants.
fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, VindexError> {
    let block_size = 18; // 2 (scale) + 16 (quants)
    let n_blocks = n_elements / 32;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];
        let scale = super::loader::half_to_f32_pub(u16::from_le_bytes([block[0], block[1]]));
        let quants = &block[2..];

        for j in 0..16 {
            let byte = quants[j];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;
            out.push(lo as f32 * scale);
            out.push(hi as f32 * scale);
        }
    }

    Ok(out)
}

/// Q4_1: 32 elements per block. Block = f16 scale + f16 min + 16 bytes of 4-bit quants.
fn dequantize_q4_1(data: &[u8], n_elements: usize) -> Result<Vec<f32>, VindexError> {
    let block_size = 20;
    let n_blocks = n_elements / 32;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];
        let scale = super::loader::half_to_f32_pub(u16::from_le_bytes([block[0], block[1]]));
        let min = super::loader::half_to_f32_pub(u16::from_le_bytes([block[2], block[3]]));
        let quants = &block[4..];

        for j in 0..16 {
            let byte = quants[j];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            out.push(lo * scale + min);
            out.push(hi * scale + min);
        }
    }

    Ok(out)
}

/// Q8_0: 32 elements per block. Block = f16 scale (2 bytes) + 32 signed int8 quants.
fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, VindexError> {
    let block_size = 34; // 2 (scale) + 32 (quants)
    let n_blocks = n_elements / 32;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];
        let scale = super::loader::half_to_f32_pub(u16::from_le_bytes([block[0], block[1]]));
        let quants = &block[2..];

        for j in 0..32 {
            out.push(quants[j] as i8 as f32 * scale);
        }
    }

    Ok(out)
}

/// Normalize GGUF tensor key names to match HuggingFace conventions.
pub fn normalize_gguf_key(name: &str) -> String {
    // GGUF uses "blk.N.attn_q.weight" format
    // HF uses "model.layers.N.self_attn.q_proj.weight" format
    // We normalize to the HF style since that's what ModelArchitecture expects

    let name = name
        .replace("blk.", "layers.")
        .replace("attn_q.", "self_attn.q_proj.")
        .replace("attn_k.", "self_attn.k_proj.")
        .replace("attn_v.", "self_attn.v_proj.")
        .replace("attn_output.", "self_attn.o_proj.")
        .replace("ffn_gate.", "mlp.gate_proj.")
        .replace("ffn_up.", "mlp.up_proj.")
        .replace("ffn_down.", "mlp.down_proj.")
        .replace("attn_norm.", "input_layernorm.")
        .replace("ffn_norm.", "post_attention_layernorm.")
        .replace("token_embd.", "embed_tokens.")
        .replace("output_norm.", "norm.")
        .replace("output.", "lm_head.");

    name
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_gguf_key() {
        assert_eq!(
            normalize_gguf_key("blk.0.attn_q.weight"),
            "layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            normalize_gguf_key("blk.15.ffn_gate.weight"),
            "layers.15.mlp.gate_proj.weight"
        );
        assert_eq!(
            normalize_gguf_key("token_embd.weight"),
            "embed_tokens.weight"
        );
        assert_eq!(
            normalize_gguf_key("output.weight"),
            "lm_head.weight"
        );
    }

    #[test]
    fn test_dequantize_q4_0() {
        // One block: 32 elements
        // Scale = 1.0 as f16 = 0x3C00
        let mut block = vec![0x00, 0x3C]; // f16 scale = 1.0
        // 16 bytes: 0x12 means lo=2-8=-6, hi=1-8=-7
        block.extend_from_slice(&[0x12; 16]);

        let result = dequantize_q4_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        // lo nibble = 2, 2-8 = -6, * 1.0 = -6.0
        // hi nibble = 1, 1-8 = -7, * 1.0 = -7.0
        assert!((result[0] - (-6.0)).abs() < 0.01);
        assert!((result[1] - (-7.0)).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_q8_0() {
        // One block: 32 elements
        // Scale = 0.5 as f16 = 0x3800
        let mut block = vec![0x00, 0x38]; // f16 scale = 0.5
        // 32 signed int8 quants: alternating 2 and -2
        for _ in 0..16 {
            block.push(2u8);  // +2
            block.push(0xFEu8); // -2 as i8
        }

        let result = dequantize_q8_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - 1.0).abs() < 0.01); // 2 * 0.5 = 1.0
        assert!((result[1] - (-1.0)).abs() < 0.01); // -2 * 0.5 = -1.0
    }
}
