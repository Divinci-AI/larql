//! GGML block quantization — dequantize Q4_0, Q4_1, Q5_0, Q5_1, Q8_0.
//!
//! Used by GGUF model files. Each format stores blocks of 32 elements
//! with shared scale factors.

use crate::detect::ModelError;
use super::half::f16_to_f32;

// GGML tensor type IDs
pub const TYPE_F32: u32 = 0;
pub const TYPE_F16: u32 = 1;
pub const TYPE_Q4_0: u32 = 2;
pub const TYPE_Q4_1: u32 = 3;
pub const TYPE_Q8_0: u32 = 6;
pub const TYPE_Q5_0: u32 = 8;
pub const TYPE_Q5_1: u32 = 9;
pub const TYPE_BF16: u32 = 30;

/// Compute byte size for a tensor of given type and element count.
pub fn tensor_data_size(tensor_type: u32, n_elements: usize) -> Result<usize, ModelError> {
    match tensor_type {
        TYPE_F32 => Ok(n_elements * 4),
        TYPE_F16 | TYPE_BF16 => Ok(n_elements * 2),
        TYPE_Q4_0 => Ok(n_elements / 32 * 18),
        TYPE_Q4_1 => Ok(n_elements / 32 * 20),
        TYPE_Q5_0 => Ok(n_elements / 32 * 22),
        TYPE_Q5_1 => Ok(n_elements / 32 * 24),
        TYPE_Q8_0 => Ok(n_elements / 32 * 34),
        other => Err(ModelError::UnsupportedDtype(format!("GGML type {other}"))),
    }
}

/// Human-readable name for a GGML tensor type.
pub fn type_name(tensor_type: u32) -> &'static str {
    match tensor_type {
        TYPE_F32 => "F32",
        TYPE_F16 => "F16",
        TYPE_Q4_0 => "Q4_0",
        TYPE_Q4_1 => "Q4_1",
        TYPE_Q8_0 => "Q8_0",
        TYPE_Q5_0 => "Q5_0",
        TYPE_Q5_1 => "Q5_1",
        TYPE_BF16 => "BF16",
        _ => "unknown",
    }
}

/// Dequantize raw bytes to f32 based on GGML tensor type.
pub fn dequantize(data: &[u8], tensor_type: u32, n_elements: usize) -> Result<Vec<f32>, ModelError> {
    match tensor_type {
        TYPE_F32 => {
            Ok(data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        TYPE_F16 => Ok(super::half::decode_f16(data)),
        TYPE_BF16 => Ok(super::half::decode_bf16(data)),
        TYPE_Q4_0 => dequantize_q4_0(data, n_elements),
        TYPE_Q4_1 => dequantize_q4_1(data, n_elements),
        TYPE_Q8_0 => dequantize_q8_0(data, n_elements),
        other => Err(ModelError::UnsupportedDtype(format!("GGML type {other}"))),
    }
}

/// Q4_0: block = f16 scale (2B) + 16 bytes of 4-bit quants. 32 elements per block.
/// Each 4-bit value is unsigned [0,15], offset by -8 to give signed [-8, 7].
fn dequantize_q4_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 18;
    let n_blocks = n_elements / 32;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
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

/// Q4_1: block = f16 scale + f16 min + 16 bytes of 4-bit quants.
/// value = quant * scale + min
fn dequantize_q4_1(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 20;
    let n_blocks = n_elements / 32;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
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

/// Q8_0: block = f16 scale (2B) + 32 signed int8 quants.
fn dequantize_q8_0(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 34;
    let n_blocks = n_elements / 32;
    let mut out = Vec::with_capacity(n_elements);

    for i in 0..n_blocks {
        let block = &data[i * block_size..(i + 1) * block_size];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let quants = &block[2..];

        for j in 0..32 {
            out.push(quants[j] as i8 as f32 * scale);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Q4_0 ──

    #[test]
    fn q4_0_basic() {
        // Scale = 1.0, quants = 0x12 → lo=2-8=-6, hi=1-8=-7
        let mut block = vec![0x00, 0x3C]; // f16 1.0
        block.extend_from_slice(&[0x12; 16]);
        let result = dequantize_q4_0(&block, 32).unwrap();
        assert_eq!(result.len(), 32);
        assert!((result[0] - (-6.0)).abs() < 0.01);
        assert!((result[1] - (-7.0)).abs() < 0.01);
    }

    #[test]
    fn q4_0_zero_scale() {
        let mut block = vec![0x00, 0x00]; // f16 0.0
        block.extend_from_slice(&[0xFF; 16]);
        let result = dequantize_q4_0(&block, 32).unwrap();
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q4_0_two_blocks() {
        let mut data = vec![0x00, 0x3C]; // block 0: scale=1.0
        data.extend_from_slice(&[0x88; 16]); // quants: lo=8-8=0, hi=8-8=0
        data.extend_from_slice(&[0x00, 0x40]); // block 1: scale=2.0
        data.extend_from_slice(&[0x19; 16]); // lo=9-8=1, hi=1-8=-7
        let result = dequantize_q4_0(&data, 64).unwrap();
        assert_eq!(result.len(), 64);
        assert!((result[0] - 0.0).abs() < 0.01); // block 0
        assert!((result[32] - 2.0).abs() < 0.01); // block 1: 1*2.0 = 2.0
        assert!((result[33] - (-14.0)).abs() < 0.01); // block 1: -7*2.0 = -14.0
    }

    // ── Q4_1 ──

    #[test]
    fn q4_1_basic() {
        // Scale=1.0, min=0.5, quants=0x00 → lo=0*1+0.5=0.5, hi=0*1+0.5=0.5
        let mut block = vec![0x00, 0x3C, 0x00, 0x38]; // scale=1.0, min=0.5
        block.extend_from_slice(&[0x00; 16]);
        let result = dequantize_q4_1(&block, 32).unwrap();
        assert!((result[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn q4_1_with_offset() {
        // Scale=2.0, min=-1.0, quants=0x31 → lo=1*2-1=1, hi=3*2-1=5
        let mut block = vec![0x00, 0x40, 0x00, 0xBC]; // scale=2.0, min=-1.0
        block.extend_from_slice(&[0x31; 16]);
        let result = dequantize_q4_1(&block, 32).unwrap();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 5.0).abs() < 0.01);
    }

    // ── Q8_0 ──

    #[test]
    fn q8_0_basic() {
        let mut block = vec![0x00, 0x38]; // f16 scale = 0.5
        for _ in 0..16 {
            block.push(2u8);    // +2 → 2*0.5 = 1.0
            block.push(0xFEu8); // -2 as i8 → -2*0.5 = -1.0
        }
        let result = dequantize_q8_0(&block, 32).unwrap();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn q8_0_zero_scale() {
        let mut block = vec![0x00, 0x00]; // scale = 0
        block.extend_from_slice(&[127u8; 32]); // max int8
        let result = dequantize_q8_0(&block, 32).unwrap();
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q8_0_full_range() {
        let mut block = vec![0x00, 0x3C]; // scale = 1.0
        block.push(127); // max positive
        block.push(0x81); // -127 as i8
        block.extend_from_slice(&[0u8; 30]); // rest zeros
        let result = dequantize_q8_0(&block, 32).unwrap();
        assert!((result[0] - 127.0).abs() < 0.01);
        assert!((result[1] - (-127.0)).abs() < 0.01);
        assert!((result[2] - 0.0).abs() < 0.01);
    }

    // ── Type metadata ──

    #[test]
    fn tensor_sizes() {
        assert_eq!(tensor_data_size(TYPE_F32, 32).unwrap(), 128);
        assert_eq!(tensor_data_size(TYPE_F16, 32).unwrap(), 64);
        assert_eq!(tensor_data_size(TYPE_Q4_0, 32).unwrap(), 18);
        assert_eq!(tensor_data_size(TYPE_Q4_1, 32).unwrap(), 20);
        assert_eq!(tensor_data_size(TYPE_Q8_0, 32).unwrap(), 34);
    }

    #[test]
    fn type_names() {
        assert_eq!(type_name(TYPE_F32), "F32");
        assert_eq!(type_name(TYPE_Q4_0), "Q4_0");
        assert_eq!(type_name(TYPE_Q8_0), "Q8_0");
        assert_eq!(type_name(99), "unknown");
    }

    // ── F32 passthrough ──

    #[test]
    fn f32_passthrough() {
        let data: Vec<u8> = [1.0f32, -2.0, 3.0].iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let result = dequantize(&data, TYPE_F32, 3).unwrap();
        assert_eq!(result, vec![1.0, -2.0, 3.0]);
    }
}
