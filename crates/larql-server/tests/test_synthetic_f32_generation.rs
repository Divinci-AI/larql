//! An unquantised (f32-only) vindex must generate, not panic.
//!
//! `generate_with_sampling` on a CPU backend routes through the Q4K dequant
//! loop, which used to die with "attn Q4K slices missing for layer 0" on a
//! vindex that carries no Q4K attention — the shape of every `--include-weights`
//! f32 extract, gemma4 b20ff753 included. It now decodes through the dense f32
//! forward instead.

mod common;

#[test]
fn f32_only_vindex_generates_instead_of_panicking() {
    let (model, _fixture) = common::model_with_real_weights("synthetic-f32");

    let mut weights_guard = model.lock_weights_for_gen().expect("lock weights");
    let weights: &mut larql_inference::ModelWeights = &mut weights_guard;

    let encoding = model.tokenizer.encode("the capital", true).expect("encode");
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    assert!(!prompt_ids.is_empty());

    let patched = model.patched.blocking_read();
    let index = patched.base();
    assert!(
        index.attn_kquant_layer_data(0).is_none(),
        "fixture must be f32-only for this test to mean anything"
    );
    let backend = larql_compute::default_backend();
    let cached_layers = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
    let num_layers = weights.num_layers;
    let (sampling, eos) = larql_server::routes::openai::util::build_sampling_eos(
        larql_server::routes::openai::util::SamplingParams {
            temperature: None,
            top_p: None,
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
        },
        &[],
    );

    let result = larql_inference::layer_graph::generate_with_sampling(
        weights,
        &model.tokenizer,
        &prompt_ids,
        3,
        index,
        &*backend,
        &cached_layers,
        0..num_layers,
        sampling,
        &eos,
    );

    assert!(
        result.error.is_none(),
        "generation reported an error: {:?}",
        result.error
    );
    assert!(
        !result.tokens.is_empty(),
        "expected at least one token from the f32 path"
    );
    for (_, prob) in &result.tokens {
        assert!(prob.is_finite());
    }
}
