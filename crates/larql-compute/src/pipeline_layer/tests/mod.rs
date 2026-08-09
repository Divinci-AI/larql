//! Coverage for the simple per-arch helpers (kv shapes, format
//! parsing, routing policy). The big MoE branches in
//! `build_moe_weights` need a Gemma 4 MoE fixture and live in the
//! `larql-inference` integration tests where that fixture is
//! reachable.
use super::moe_build::moe_routing_policy;
use super::*;
use larql_models::test_fixtures::make_test_weights;

/// Capacity must accommodate the compaction trigger, not merely the
/// window: occupancy is allowed to reach `SLACK * W` before
/// compaction reclaims it, so a capacity of `W` would be an overrun
/// rather than a smaller allocation.
#[test]
fn kv_capacity_accommodates_the_compaction_trigger() {
    assert_eq!(
        kv_capacity_for_window(1024, 4096),
        1024 * KV_COMPACTION_SLACK
    );
    assert_eq!(kv_capacity_for_window(1024, 4096), 2048);
}

/// `0` is the unbounded sentinel — a global layer keeps the full
/// default because nothing bounds what it may read.
#[test]
fn kv_capacity_leaves_unbounded_layers_at_the_default() {
    assert_eq!(kv_capacity_for_window(0, 4096), 4096);
}

/// A window wider than the default cannot ask for more than it, and
/// a window near `usize::MAX` must not overflow into a tiny capacity.
#[test]
fn kv_capacity_is_clamped_by_the_default() {
    assert_eq!(kv_capacity_for_window(8192, 4096), 4096);
    assert_eq!(kv_capacity_for_window(usize::MAX, 4096), 4096);
}

/// A window small enough that the trigger stays under the default
/// gets the smaller allocation, which is the whole point.
#[test]
fn kv_capacity_shrinks_a_narrow_window() {
    assert_eq!(kv_capacity_for_window(64, 4096), 128);
}

#[test]
fn kv_capacities_for_arch_returns_one_entry_per_layer() {
    let weights = make_test_weights();
    let caps = kv_capacities_for_arch(&weights, 4096);
    assert_eq!(caps.len(), weights.num_layers);
    // Every entry is a real allocation and never exceeds the default.
    for c in &caps {
        assert!(*c > 0 && *c <= 4096, "capacity {c} out of range");
    }
    // And each agrees with the per-window rule applied to that layer.
    for (layer, &c) in caps.iter().enumerate() {
        let window =
            crate::forward_overrides::effective_attention_window_for_layer(&*weights.arch, layer)
                .unwrap_or(0);
        assert_eq!(c, kv_capacity_for_window(window, 4096), "layer {layer}");
    }
}

#[test]
fn kv_cache_shapes_for_arch_returns_one_pair_per_layer() {
    let weights = make_test_weights();
    let shapes = kv_cache_shapes_for_arch(&weights);
    assert_eq!(shapes.len(), weights.num_layers);
    for (num_kv, head_dim) in &shapes {
        assert!(*num_kv > 0);
        assert!(*head_dim > 0);
    }
}

#[test]
fn attn_str_to_format_maps_known_tags() {
    assert_eq!(attn_str_to_format("Q4_K"), QuantFormat::Q4_K);
    assert_eq!(attn_str_to_format("Q6_K"), QuantFormat::Q6_K);
}

#[test]
#[should_panic(expected = "no compute::QuantFormat mapping")]
fn attn_str_to_format_panics_on_unknown_tag() {
    let _ = attn_str_to_format("Q42_X");
}

#[test]
fn ffn_str_to_format_maps_known_tags() {
    assert_eq!(
        ffn_str_to_format("Q4_K", QuantFormat::Q4_K),
        QuantFormat::Q4_K
    );
    assert_eq!(
        ffn_str_to_format("Q6_K", QuantFormat::Q4_K),
        QuantFormat::Q6_K
    );
    assert_eq!(
        ffn_str_to_format("Q4_0", QuantFormat::Q4_K),
        QuantFormat::Q4_0
    );
    // Empty tag falls through to the caller's fallback.
    assert_eq!(ffn_str_to_format("", QuantFormat::Q4_0), QuantFormat::Q4_0);
    assert_eq!(ffn_str_to_format("", QuantFormat::Q4_K), QuantFormat::Q4_K);
}

#[test]
#[should_panic(expected = "no compute::QuantFormat mapping")]
fn ffn_str_to_format_panics_on_unknown_tag() {
    let _ = ffn_str_to_format("unknown", QuantFormat::Q4_K);
}

/// Each router kind must map to a *distinct* policy.
///
/// The predecessor of this test called `moe_routing_policy` twice and
/// asserted nothing, so it passed while the string `match` silently sent
/// GPT-OSS's router to the default arm.
#[test]
fn every_router_kind_maps_to_its_own_policy() {
    use larql_models::MoeRouterKind::*;
    let gemma4 = moe_routing_policy(Gemma4Hybrid);
    let plain = moe_routing_policy(TopKSoftmax);
    let selected = moe_routing_policy(TopKThenSoftmax);
    assert_ne!(gemma4, plain);
    assert_ne!(
        plain, selected,
        "top-k-then-softmax must not equal the default"
    );
    assert_ne!(gemma4, selected);
}

/// The distinction that was being lost: selected weights summing to 1
/// versus keeping the raw softmax mass. Confusing them rescales the whole
/// expert branch.
#[test]
fn top_k_then_softmax_renormalises_where_the_default_does_not() {
    use larql_models::MoeRouterKind::*;
    assert_eq!(
        moe_routing_policy(TopKThenSoftmax).selected_weight,
        crate::MoeTopKWeightPolicy::RenormalizedSoftmax
    );
    assert_eq!(
        moe_routing_policy(TopKSoftmax).selected_weight,
        crate::MoeTopKWeightPolicy::RawSoftmax
    );
}

/// GPT-OSS's router lives inside the MLP block in the reference, so BOTH
/// the router and the experts read the pre-experts-normed hidden — on the
/// quantised path that norm is the policy's to apply, since the caller
/// hands the MoE the raw residual. Serving with `Residual` here ran every
/// router and expert on the un-normed stream: structurally sane routing,
/// incoherent generation, no crash. This pins the topology so a revert
/// fails a test instead of a generation.
#[test]
fn top_k_then_softmax_routes_and_runs_on_the_pre_experts_normed_input() {
    use larql_models::MoeRouterKind::TopKThenSoftmax;
    let policy = moe_routing_policy(TopKThenSoftmax);
    assert_eq!(policy.expert_input, crate::MoeInputSource::PreExpertsNorm);
    assert_eq!(policy.router_input, crate::MoeInputSource::PreExpertsNorm);
}

/// `resolve_attn_weights` falls through to the Q8 branch when the
/// index returns Q8 data instead of Q4_K.
#[test]
fn resolve_attn_weights_uses_q8_branch_when_index_returns_q8() {
    struct Q8Idx {
        bytes: Vec<u8>,
        scales: Vec<f32>,
    }
    impl crate::KvIndex for Q8Idx {
        fn attn_q8_layer_data(&self, _l: usize) -> Option<[(&[u8], &[f32]); 4]> {
            Some([
                (self.bytes.as_slice(), self.scales.as_slice()),
                (self.bytes.as_slice(), self.scales.as_slice()),
                (self.bytes.as_slice(), self.scales.as_slice()),
                (self.bytes.as_slice(), self.scales.as_slice()),
            ])
        }
    }
    let idx = Q8Idx {
        bytes: vec![0u8; 16],
        scales: vec![1.0f32; 4],
    };
    let result = resolve_attn_weights(&idx, 0);
    let (q, _k, _v, _o) = result.expect("Q8 fallback returns Some");
    assert_eq!(q.format(), QuantFormat::Q8_0);
}

/// `build_arch_params` rotary_dim branch fires when `rotary_fraction`
/// is < 1.0 (partial-rotary archs like StarCoder2).
#[test]
fn build_arch_params_handles_partial_rotary_fraction() {
    let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
    let dummy = crate::QuantWeight::new(QuantFormat::Q4_K, &[], crate::QuantAux::None);
    // The partial-rotary branch is shape-dependent on the arch; what
    // we want is just to ensure no panic on a non-full-rotary arch.
    let layer = build_arch_params(&weights, 0, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
    let _ = layer.rotary_dim;
}

/// `build_arch_params` on Llama2-style (Silu activation) fixture —
/// covers the Silu fallback branch in the activation match.
#[test]
fn build_arch_params_handles_silu_activation() {
    let weights = make_test_weights();
    let dummy = crate::QuantWeight::new(QuantFormat::Q4_K, &[], crate::QuantAux::None);
    let layer = build_arch_params(&weights, 0, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
    assert!(matches!(layer.activation, crate::Activation::Silu));
}

/// `build_arch_params` on Starcoder2-style fixture covers the
/// LayerNorm branch and the Standard (non-gated) FFN type.
#[test]
fn build_arch_params_handles_layernorm_and_standard_ffn() {
    let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
    let dummy = crate::QuantWeight::new(QuantFormat::Q4_K, &[], crate::QuantAux::None);
    let layer = build_arch_params(&weights, 0, dummy, dummy, dummy, dummy, dummy, dummy, dummy);
    assert!(matches!(layer.norm_type, crate::NormType::LayerNorm));
    assert!(matches!(layer.ffn_type, crate::FfnType::Standard));
}

/// `build_moe_weights` happy path on the Gemma 4 hybrid-MoE fixture
/// — exercises the per-layer FFN router + packed expert slicing,
/// the BF16-stride math, and the routing-policy assignment.
#[test]
fn build_moe_weights_succeeds_on_hybrid_moe_fixture() {
    let weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
    assert!(weights.arch.is_hybrid_moe());
    let arch = &*weights.arch;
    for layer in 0..weights.num_layers {
        let result = build_moe_weights(&weights, arch, layer);
        assert!(
            result.is_some(),
            "MoE weights should resolve for layer {layer} on Gemma 4 hybrid-MoE"
        );
    }
}

/// `build_moe_weights` returns None on a non-MoE arch — covers the
/// `arch.moe_router_key(layer)?` short-circuit.
#[test]
fn build_moe_weights_returns_none_on_non_moe_arch() {
    let weights = make_test_weights();
    assert!(!weights.arch.is_hybrid_moe());
    assert!(build_moe_weights(&weights, &*weights.arch, 0).is_none());
}

/// `patch_pipeline_layers_for_remote_moe` injects MoE stubs on
/// MoE-capable layers when the local moe slot is still None.
#[test]
fn patch_pipeline_layers_for_remote_moe_injects_stubs() {
    let weights = larql_models::test_fixtures::make_test_gemma4_moe_weights();
    // Build pipeline layers with no MoE locally — simulates the
    // remote-MoE client deployment.
    let dummy = crate::QuantWeight::new(QuantFormat::Q4_K, &[], crate::QuantAux::None);
    let mut layers: Vec<crate::FullPipelineLayer<'_>> = (0..weights.num_layers)
        .map(|_| crate::FullPipelineLayer {
            wq: dummy,
            wk: dummy,
            wv: dummy,
            wo: dummy,
            gate: dummy,
            up: dummy,
            down: dummy,
            ..crate::FullPipelineLayer::default()
        })
        .collect();
    // Pre-patch: every layer has moe = None.
    for l in &layers {
        assert!(l.moe.is_none());
    }
    patch_pipeline_layers_for_remote_moe(&mut layers, &weights);
    // Post-patch: every MoE-capable layer has Some moe stub.
    let mut any_patched = false;
    for l in &layers {
        if l.moe.is_some() {
            any_patched = true;
        }
    }
    assert!(any_patched, "patch must inject at least one MoE stub");
}

#[test]
fn patch_pipeline_layers_for_remote_ffn_sets_remote_flag() {
    // Build a 1-layer pipeline and patch to remote FFN.
    let layer = crate::FullPipelineLayer::default();
    let mut layers = vec![layer];
    assert!(!layers[0].ffn_is_remote);
    patch_pipeline_layers_for_remote_ffn(&mut layers);
    for l in &layers {
        assert!(l.ffn_is_remote, "patch should set ffn_is_remote = true");
    }
}

/// Pure-MoE vindexes (e.g. Gemma-4 26B A4B) ship a zero-byte
/// `interleaved_q4k.bin` because there is no dense FFN. Before the
/// `is_empty()` guard, `resolve_ffn_weights` would panic on the first
/// `q4_ffn_mmap[fs..fs + q4_ffn_per_matrix]` slice. The guard returns
/// empty `QuantWeight` stubs — `patch_pipeline_layers_for_remote_moe`
/// overwrites the per-layer MoE weights afterward and the dense FFN
/// path is bypassed entirely by `moe_fn` during decode, so the empty
/// slices are never read.
#[test]
fn resolve_ffn_weights_returns_empty_stubs_when_q4_ffn_mmap_is_empty() {
    struct EmptyIdx;
    impl crate::KvIndex for EmptyIdx {}
    let idx = EmptyIdx;
    let empty_mmap: &[u8] = &[];
    // q4_ffn_per_matrix is irrelevant on this path — what we're pinning
    // is "no slice happens against the empty mmap" (i.e. no panic).
    let (gate, up, down) = resolve_ffn_weights(&idx, 7, empty_mmap, 1_115_136, QuantFormat::Q4_K);
    assert!(gate.data.is_empty());
    assert!(up.data.is_empty());
    assert!(down.data.is_empty());
    assert_eq!(gate.format(), QuantFormat::Q4_K);
    assert_eq!(up.format(), QuantFormat::Q4_K);
    assert_eq!(down.format(), QuantFormat::Q4_K);
}
