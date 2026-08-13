//! Gemma 2 architecture.
//!
//! Key differences from Gemma 3:
//! - attn_logit_softcapping (typically 50.0)
//! - final_logit_softcapping (typically 30.0)
//! - No sliding window (uses full attention on all layers)
//! - No local RoPE base (single rope_theta for all layers)
//! - query_pre_attn_scalar may differ from head_dim

use crate::config::{Activation, ModelArchitecture, ModelConfig, PostNormEps};
use crate::tensor_keys::qk_norm;

pub struct Gemma2Arch {
    config: ModelConfig,
}

impl Gemma2Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for Gemma2Arch {
    fn family(&self) -> &str {
        "gemma2"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::q(&self.layer_prefix(layer))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::k(&self.layer_prefix(layer))
    }

    fn norm_weight_offset(&self) -> f32 {
        1.0
    }

    fn qk_norm_weight_offset(&self) -> f32 {
        1.0
    }

    fn activation(&self) -> Activation {
        Activation::GeluTanh
    }

    fn embed_scale(&self) -> Option<f32> {
        Some((self.config.hidden_size as f32).sqrt())
    }

    fn has_post_norms(&self) -> bool {
        true
    }

    /// Gemma 2's post-norms use `rms_norm_eps` — the same epsilon as its
    /// pre-norms. The checkpoint declares no separate `post_norm_eps` and
    /// the reference implementation builds all four norms from the one
    /// value, so sharing is established rather than assumed. Stated
    /// explicitly because a four-norm stack that leaves this unjudged is
    /// refused, and silence would otherwise read as "unknown".
    fn post_norm_eps(&self) -> Option<PostNormEps> {
        Some(PostNormEps::Shared)
    }

    // No sliding window — all layers use full attention with the same rope_theta
}
