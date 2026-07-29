//! GPT-OSS architecture — OpenAI's MoE model with MXFP4 packed experts.
//!
//! Key differences from standard MoE (Mixtral):
//! - Expert weights are packed as MXFP4 (e8m0 scales + 4-bit values)
//! - Gate and up projections are fused: `gate_up_proj_blocks` (first half = gate)
//! - All experts packed in one tensor per layer, not per-expert files
//! - Router at `mlp.router.weight` (not `block_sparse_moe.gate`)
//! - Attention has biases and sinks (both declared below and asserted in
//!   the tests — this comment claimed them for months while nothing
//!   returned the keys, so extraction dropped them), and uses GQA
//! - YaRN RoPE scaling

use crate::config::{ExpertFormat, ModelArchitecture, ModelConfig};

pub struct GptOssArch {
    config: ModelConfig,
}

impl GptOssArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for GptOssArch {
    fn family(&self) -> &str {
        "gpt_oss"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn key_prefixes_to_strip(&self) -> &[&str] {
        &["model."]
    }

    // ── Attention ──

    fn attn_q_key(&self, layer: usize) -> String {
        format!("{}self_attn.q_proj.weight", self.layer_prefix(layer))
    }

    fn attn_k_key(&self, layer: usize) -> String {
        format!("{}self_attn.k_proj.weight", self.layer_prefix(layer))
    }

    fn attn_v_key(&self, layer: usize) -> String {
        format!("{}self_attn.v_proj.weight", self.layer_prefix(layer))
    }

    fn attn_o_key(&self, layer: usize) -> String {
        format!("{}self_attn.o_proj.weight", self.layer_prefix(layer))
    }

    // ── MoE ──

    fn is_moe(&self) -> bool {
        true
    }

    fn expert_format(&self) -> ExpertFormat {
        ExpertFormat::PackedMxfp4
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(128)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config.num_experts_per_token.unwrap_or(4)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}mlp.router.weight", self.layer_prefix(layer)))
    }

    // ── Attention biases + sinks ──
    //
    // The module header has claimed "attention has biases, sinks" since
    // this file was written, but nothing declared them, so extraction
    // silently dropped 5 of the 11 attention tensors each layer (four
    // projection biases + sinks = 120 tensors on the 20B). Declared here
    // 2026-07-29; see `docs/k3-funnel.md` §4.6.

    fn attn_q_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.q_proj.bias", self.layer_prefix(layer)))
    }

    fn attn_k_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.k_proj.bias", self.layer_prefix(layer)))
    }

    fn attn_v_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.v_proj.bias", self.layer_prefix(layer)))
    }

    fn attn_o_bias_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.o_proj.bias", self.layer_prefix(layer)))
    }

    fn attn_sinks_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}self_attn.sinks", self.layer_prefix(layer)))
    }

    // ── Packed MXFP4 expert keys ──

    fn packed_gate_up_blocks_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.gate_up_proj_blocks",
            self.layer_prefix(layer)
        ))
    }

    fn packed_gate_up_scales_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.gate_up_proj_scales",
            self.layer_prefix(layer)
        ))
    }

    fn packed_down_blocks_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.down_proj_blocks",
            self.layer_prefix(layer)
        ))
    }

    fn packed_down_scales_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}mlp.experts.down_proj_scales",
            self.layer_prefix(layer)
        ))
    }

    // Per-expert keys are not available for GPT-OSS (packed format).
    // Callers should check expert_format() and use packed_* keys instead.
}

#[cfg(test)]
mod tests {
    use crate::config::{ExpertFormat, ModelArchitecture};

    /// Minimal `config.json` for `openai/gpt-oss-20b`, matching the real
    /// checkpoint's shape fields.
    fn arch() -> Box<dyn ModelArchitecture> {
        crate::detect_from_json(&serde_json::json!({
            "model_type": "gpt_oss",
            "num_hidden_layers": 24,
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 201088,
            "num_local_experts": 32,
            "num_experts_per_tok": 4,
            "rope_theta": 150000.0,
        }))
    }

    #[test]
    fn detects_as_packed_mxfp4_moe() {
        let a = arch();
        assert_eq!(a.family(), "gpt_oss");
        assert!(a.is_moe());
        assert_eq!(a.expert_format(), ExpertFormat::PackedMxfp4);
        assert_eq!(a.num_experts(), 32);
        assert_eq!(a.num_experts_per_token(), 4);
    }

    // ── Attention biases and sinks ──────────────────────────────────────
    // Regression tests for the silent-drop bug: the module header claimed
    // these tensors existed while every accessor returned `None`, so
    // extraction discarded 5 of 11 attention tensors per layer.
    // See `docs/k3-funnel.md` §4.6.1.

    #[test]
    fn declares_all_four_projection_biases() {
        let a = arch();
        assert_eq!(
            a.attn_q_bias_key(3).as_deref(),
            Some("layers.3.self_attn.q_proj.bias")
        );
        assert_eq!(
            a.attn_k_bias_key(3).as_deref(),
            Some("layers.3.self_attn.k_proj.bias")
        );
        assert_eq!(
            a.attn_v_bias_key(3).as_deref(),
            Some("layers.3.self_attn.v_proj.bias")
        );
        assert_eq!(
            a.attn_o_bias_key(3).as_deref(),
            Some("layers.3.self_attn.o_proj.bias")
        );
    }

    #[test]
    fn declares_attention_sinks() {
        assert_eq!(
            arch().attn_sinks_key(7).as_deref(),
            Some("layers.7.self_attn.sinks")
        );
    }

    #[test]
    fn every_attention_tensor_in_the_checkpoint_is_named() {
        // The real checkpoint carries 11 attention-related tensors per
        // layer. All 11 must be reachable through the trait, or
        // extraction silently drops whatever is missing.
        let a = arch();
        let named: Vec<String> = [
            Some(a.attn_q_key(0)),
            Some(a.attn_k_key(0)),
            Some(a.attn_v_key(0)),
            Some(a.attn_o_key(0)),
            a.attn_q_bias_key(0),
            a.attn_k_bias_key(0),
            a.attn_v_bias_key(0),
            a.attn_o_bias_key(0),
            a.attn_sinks_key(0),
            Some(a.input_layernorm_key(0)),
            Some(a.post_attention_layernorm_key(0)),
        ]
        .into_iter()
        .flatten()
        .collect();
        assert_eq!(named.len(), 11, "named tensors: {named:?}");
    }

    #[test]
    fn packed_expert_keys_follow_the_fused_layout() {
        let a = arch();
        assert_eq!(
            a.packed_gate_up_blocks_key(1).as_deref(),
            Some("layers.1.mlp.experts.gate_up_proj_blocks")
        );
        assert_eq!(
            a.packed_gate_up_scales_key(1).as_deref(),
            Some("layers.1.mlp.experts.gate_up_proj_scales")
        );
        assert_eq!(
            a.packed_down_blocks_key(1).as_deref(),
            Some("layers.1.mlp.experts.down_proj_blocks")
        );
        assert_eq!(
            a.packed_down_scales_key(1).as_deref(),
            Some("layers.1.mlp.experts.down_proj_scales")
        );
    }

    #[test]
    fn router_key_is_mlp_router_not_block_sparse_gate() {
        assert_eq!(
            arch().moe_router_key(2).as_deref(),
            Some("layers.2.mlp.router.weight")
        );
    }
}
