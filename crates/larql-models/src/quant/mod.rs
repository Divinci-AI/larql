//! Quantization and dequantization — convert between float and quantized formats.
//!
//! Supports:
//! - **half**: f16/bf16 ↔ f32 conversion
//! - **ggml**: GGML block quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
//! - **mxfp4**: Microscaling 4-bit floats with e8m0 scales (GPT-OSS/OpenAI)
//!
//! Dequantizers produce `Vec<f32>`. Quantizers produce packed byte vectors.
//! Used by GGUF loading, MXFP4 expert unpacking, and future COMPILE output.

pub mod half;
pub mod ggml;
pub mod mxfp4;
