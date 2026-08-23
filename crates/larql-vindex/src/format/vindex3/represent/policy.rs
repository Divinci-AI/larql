//! Which tensors a representation applies to, by **role**.
//!
//! The obvious policy — "it is a 2-D matrix, so quantise it" — is not a
//! policy, it is a shape test. It happens to produce a good number on a
//! dense model and it silently 4-bits the embedding table and the output
//! head, which are among the places you would least want to be first.
//!
//! Eligibility is therefore semantic: a tensor's role decides, and the
//! default is conservative at every role where 4-bit is known to be
//! delicate. Explicit opt-in makes it more aggressive; nothing makes it
//! more aggressive by accident.
//!
//! ```text
//! decoder linear weights   REPRESENT   attention q/k/v/o, mlp gate/up/down
//! expert weights           REPRESENT   the prize at MoE scale
//!
//! embedding                PRESERVE
//! output head              PRESERVE
//! norms                    PRESERVE
//! router / gate            PRESERVE    tiny, and routing errors compound
//! small vectors, biases    PRESERVE
//! anything unrecognised    PRESERVE    fail safe, never fail small
//! ```
//!
//! The last line is the one that matters most. A tensor this classifier
//! cannot name is a tensor nobody has reasoned about, and quantising it
//! because it happened to be 2-D is how a policy acquires behaviour its
//! author never chose.

use std::fmt;

/// What a tensor does, as far as representation eligibility is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// Attention and dense-FFN projections — the bulk of a dense model.
    DecoderLinear,
    /// Routed-expert weights — the bulk of an MoE model.
    ExpertWeight,
    /// Token embedding table.
    Embedding,
    /// Output / LM head.
    OutputHead,
    /// Any normalisation weight.
    Norm,
    /// Router or expert-gate weights.
    Router,
    /// 1-D vectors and biases.
    SmallVector,
    /// Recognised as nothing in particular.
    Unknown,
}

impl Role {
    /// Every role, for CLI parsing and exhaustive reporting.
    pub const ALL: &'static [Role] = &[
        Role::DecoderLinear,
        Role::ExpertWeight,
        Role::Embedding,
        Role::OutputHead,
        Role::Norm,
        Role::Router,
        Role::SmallVector,
        Role::Unknown,
    ];

    /// Lower-kebab name, used by `--include-role` and in reports.
    pub fn name(self) -> &'static str {
        match self {
            Role::DecoderLinear => "decoder-linear",
            Role::ExpertWeight => "expert-weight",
            Role::Embedding => "embedding",
            Role::OutputHead => "output-head",
            Role::Norm => "norm",
            Role::Router => "router",
            Role::SmallVector => "small-vector",
            Role::Unknown => "unknown",
        }
    }

    pub fn parse(s: &str) -> Option<Role> {
        Role::ALL.iter().copied().find(|r| r.name() == s)
    }

    /// Whether the conservative default compiles this role.
    pub fn in_default_policy(self) -> bool {
        matches!(self, Role::DecoderLinear | Role::ExpertWeight)
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Classify one tensor from the object that holds it and its own name.
///
/// Both signals are needed. The object says what kind of thing this is —
/// `target.embedding` holds an embedding whatever its tensor is called —
/// and the tensor name discriminates within a decoder stack, where a norm
/// and a projection sit side by side under the same object.
pub fn classify(object: &str, tensor: &str, shape: &[usize]) -> Role {
    // A tensor that is not a matrix cannot carry a block-quantised
    // representation regardless of what it means, so this is settled first.
    if shape.len() != 2 {
        return Role::SmallVector;
    }

    let obj = object.to_ascii_lowercase();
    let name = tensor.to_ascii_lowercase();

    // Object-level roles: the object *is* the thing.
    if obj.contains("embedding") || obj.contains("embed_tokens") {
        return Role::Embedding;
    }
    if obj.contains("output_head") || obj.contains("lm_head") {
        return Role::OutputHead;
    }
    if obj.contains("final_norm") {
        return Role::Norm;
    }

    // Tensor-level roles within a stack or a bank.
    if name.contains("norm") {
        return Role::Norm;
    }
    // `router` and a bare `gate` select experts; `gate_proj` is the GLU
    // gate half of a dense FFN and is ordinary decoder linear work. The
    // two are one token apart and mean entirely different things.
    if name.contains("router") || name.contains("gate.weight") || name.ends_with(".gate") {
        return Role::Router;
    }
    if name.ends_with("bias") {
        return Role::SmallVector;
    }

    if obj.contains("expert") {
        return Role::ExpertWeight;
    }

    let is_projection = name.contains("_proj.")
        || name.ends_with("_proj")
        || name.contains("self_attn.")
        || name.contains("attention.")
        || name.contains("mlp.");
    if is_projection {
        return Role::DecoderLinear;
    }

    Role::Unknown
}

/// Which roles a compilation compiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RolePolicy {
    included: Vec<Role>,
}

impl Default for RolePolicy {
    fn default() -> Self {
        Self {
            included: Role::ALL
                .iter()
                .copied()
                .filter(|r| r.in_default_policy())
                .collect(),
        }
    }
}

impl RolePolicy {
    /// Add a role the default leaves preserved. The escape hatch for a
    /// profile that has decided, deliberately, to be more aggressive.
    pub fn including(mut self, role: Role) -> Self {
        if !self.included.contains(&role) {
            self.included.push(role);
            self.included.sort();
        }
        self
    }

    pub fn compiles(&self, role: Role) -> bool {
        self.included.contains(&role)
    }

    pub fn roles(&self) -> &[Role] {
        &self.included
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: &[usize] = &[2560, 2560];

    #[test]
    fn decoder_projections_are_compiled_by_default() {
        let p = RolePolicy::default();
        for t in [
            "0.self_attn.q_proj.weight",
            "7.self_attn.k_proj.weight",
            "7.self_attn.v_proj.weight",
            "7.self_attn.o_proj.weight",
            "39.mlp.gate_proj.weight",
            "39.mlp.up_proj.weight",
            "39.mlp.down_proj.weight",
        ] {
            let role = classify("target.decoder_stack", t, M);
            assert_eq!(role, Role::DecoderLinear, "{t}");
            assert!(p.compiles(role), "{t}");
        }
    }

    #[test]
    fn the_embedding_and_head_are_preserved_by_default() {
        let p = RolePolicy::default();
        // The whole point of the change: both are 2-D matrices, and both
        // are places 4-bit is known to be delicate.
        assert_eq!(classify("target.embedding", "weight", M), Role::Embedding);
        assert_eq!(
            classify("target.output_head", "weight", M),
            Role::OutputHead
        );
        assert!(!p.compiles(Role::Embedding));
        assert!(!p.compiles(Role::OutputHead));
    }

    #[test]
    fn expert_weights_are_compiled_but_their_router_is_not() {
        let p = RolePolicy::default();
        assert_eq!(
            classify("target.expert_bank", "3.experts.gate_up_proj", M),
            Role::ExpertWeight
        );
        assert!(p.compiles(Role::ExpertWeight));

        // Routing errors select the wrong expert entirely, which is not a
        // small numerical perturbation.
        assert_eq!(
            classify("target.expert_bank", "3.router.weight", M),
            Role::Router
        );
        assert!(!p.compiles(Role::Router));
    }

    #[test]
    fn a_glu_gate_half_is_not_a_router() {
        // `mlp.gate_proj` and `router`/`gate` are one token apart and mean
        // different things; conflating them would preserve half of every
        // dense FFN or quantise every routing decision.
        assert_eq!(
            classify("target.decoder_stack", "5.mlp.gate_proj.weight", M),
            Role::DecoderLinear
        );
        assert_eq!(
            classify("target.decoder_stack", "5.mlp.gate.weight", M),
            Role::Router
        );
    }

    #[test]
    fn norms_and_vectors_are_never_compiled() {
        let p = RolePolicy::default();
        assert_eq!(
            classify("target.decoder_stack", "0.input_layernorm.weight", &[2560]),
            Role::SmallVector
        );
        assert_eq!(
            classify(
                "target.decoder_stack",
                "0.post_attention_layernorm.weight",
                M
            ),
            Role::Norm
        );
        assert_eq!(classify("target.final_norm", "weight", M), Role::Norm);
        assert_eq!(
            classify("target.decoder_stack", "0.self_attn.q_proj.bias", M),
            Role::SmallVector
        );
        for r in [Role::Norm, Role::SmallVector] {
            assert!(!p.compiles(r));
        }
    }

    #[test]
    fn an_unrecognised_matrix_is_preserved_not_compiled() {
        // Fail safe: a tensor nobody has reasoned about must not acquire a
        // lossy representation because it happened to be 2-D.
        let role = classify("target.something_new", "mystery.weight", M);
        assert_eq!(role, Role::Unknown);
        assert!(!RolePolicy::default().compiles(role));
    }

    #[test]
    fn a_role_can_be_opted_in_explicitly() {
        let p = RolePolicy::default().including(Role::Embedding);
        assert!(p.compiles(Role::Embedding));
        assert!(p.compiles(Role::DecoderLinear));
        // Opting one role in must not opt others in with it.
        assert!(!p.compiles(Role::OutputHead));
        assert!(!p.compiles(Role::Router));
    }

    #[test]
    fn role_names_round_trip() {
        for r in Role::ALL {
            assert_eq!(Role::parse(r.name()), Some(*r), "{}", r.name());
        }
        assert_eq!(Role::parse("not-a-role"), None);
    }
}
