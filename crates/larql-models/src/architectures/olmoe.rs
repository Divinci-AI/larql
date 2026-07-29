//! OLMoE architecture (Allen AI OLMoE-1B-7B) — Llama attention + QK-norm + MoE.
//!
//! Tensor naming is identical to Qwen3-MoE:
//! - Router at `mlp.gate.weight`
//! - Per-expert `mlp.experts.{E}.{gate,up,down}_proj.weight`
//! - QK norms at `self_attn.{q,k}_norm.weight`
//!
//! It gets its own module rather than aliasing `QwenArch` because of two
//! config differences that would otherwise be silent:
//!
//! 1. **No `moe_intermediate_size` field.** OLMoE stores the per-expert
//!    intermediate width in plain `intermediate_size` (1024 for 1B-7B).
//!    `QwenArch` reads `moe_intermediate_size.unwrap_or(0)`, so aliasing
//!    would report zero-width experts.
//! 2. **`norm_topk_prob: false`.** OLMoE keeps the raw softmax probabilities
//!    of the selected experts; it does not renormalize them to sum to 1.
//!    That is `MoeTopKWeightPolicy::RawSoftmax`, which is what the routing
//!    dispatch already selects for any model not tagged
//!    `gemma4_top_k_softmax` — so no policy work is needed here, but the
//!    dependency is recorded so a future default change can't quietly
//!    invert it.
//!
//! OLMoE also sets `attention_bias: false` and `num_key_value_heads ==
//! num_attention_heads` (MHA, not GQA), so no bias keys are emitted.

use crate::config::{ModelArchitecture, ModelConfig};

pub struct OlmoeArch {
    config: ModelConfig,
}

impl OlmoeArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for OlmoeArch {
    fn family(&self) -> &str {
        "olmoe"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // ── MoE ──

    fn is_moe(&self) -> bool {
        self.config.num_experts.unwrap_or(0) > 0
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(0)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config
            .num_experts_per_token
            .or(self.config.top_k_experts)
            .unwrap_or(0)
    }

    /// OLMoE has no `moe_intermediate_size`; the per-expert width is the
    /// model's `intermediate_size`. Falling back to 0 here would size every
    /// expert to nothing.
    fn moe_intermediate_size(&self) -> usize {
        self.config
            .moe_intermediate_size
            .unwrap_or(self.config.intermediate_size)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        Some(format!("{}mlp.gate.weight", self.layer_prefix(layer)))
    }

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        Some(format!(
            "{}mlp.experts.{expert_id}.gate_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        Some(format!(
            "{}mlp.experts.{expert_id}.up_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        Some(format!(
            "{}mlp.experts.{expert_id}.down_proj.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── QK norms ──

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.q_norm.weight",
            self.layer_prefix(layer)
        ))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.k_norm.weight",
            self.layer_prefix(layer)
        ))
    }
}
