//! Colocated tests for the separate-tensor MoE layer writer.
//!
//! The regression these defend against is specific and was live: extraction of
//! a separate-tensor MoE reported success, verified clean, sliced clean, and
//! then panicked on the first decoded token because no expert store had been
//! written. So the assertions are about *files appearing on disk with the
//! right shape*, not about the quantiser — which `write_layers_parts_tests`
//! covers separately.

use std::collections::HashMap;
use std::path::Path;

use super::super::moe_layers_per_expert::write_per_layer_moe_per_expert;
use crate::format::weights::write_f32::WeightSource;
use crate::format::weights::write_layers::parse_layer_weights_header;

const HIDDEN: usize = 256;
const INTER: usize = 256;
const NUM_LAYERS: usize = 2;
const NUM_EXPERTS: usize = 3;

/// A `WeightSource` backed by an explicit tensor map, so a test can express
/// "this expert is missing" precisely.
struct MapSource {
    arch: Box<dyn larql_models::ModelArchitecture>,
    tensors: HashMap<String, (Vec<f32>, usize, usize)>,
}

impl MapSource {
    /// An OLMoE-shaped architecture — the real separate-tensor case.
    fn olmoe(num_experts: usize) -> Box<dyn larql_models::ModelArchitecture> {
        larql_models::detect_from_json(&serde_json::json!({
            "model_type": "olmoe",
            "hidden_size": HIDDEN,
            "intermediate_size": INTER,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
        }))
    }

    /// Every expert of every layer present and correctly shaped.
    fn complete() -> Self {
        let arch = Self::olmoe(NUM_EXPERTS);
        let mut tensors = HashMap::new();
        for layer in 0..NUM_LAYERS {
            for expert in 0..NUM_EXPERTS {
                insert_expert(&mut tensors, &*arch, layer, expert);
            }
        }
        Self { arch, tensors }
    }

    /// Layer 0 complete; layer 1 missing every expert (a hybrid dense layer).
    fn first_layer_only() -> Self {
        let arch = Self::olmoe(NUM_EXPERTS);
        let mut tensors = HashMap::new();
        for expert in 0..NUM_EXPERTS {
            insert_expert(&mut tensors, &*arch, 0, expert);
        }
        Self { arch, tensors }
    }

    /// Layer 0 has expert 0 but not expert 1 — a genuinely malformed layer.
    fn partial_layer() -> Self {
        let arch = Self::olmoe(NUM_EXPERTS);
        let mut tensors = HashMap::new();
        insert_expert(&mut tensors, &*arch, 0, 0);
        Self { arch, tensors }
    }
}

fn insert_expert(
    tensors: &mut HashMap<String, (Vec<f32>, usize, usize)>,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
    expert: usize,
) {
    // Distinct fill per (layer, expert) so a mixed-up write is detectable.
    let fill = (layer * 10 + expert) as f32;
    if let Some(k) = arch.expert_ffn_gate_key(layer, expert) {
        tensors.insert(k, (vec![fill; INTER * HIDDEN], INTER, HIDDEN));
    }
    if let Some(k) = arch.expert_ffn_up_key(layer, expert) {
        tensors.insert(k, (vec![-fill; INTER * HIDDEN], INTER, HIDDEN));
    }
    if let Some(k) = arch.expert_ffn_down_key(layer, expert) {
        tensors.insert(k, (vec![fill; HIDDEN * INTER], HIDDEN, INTER));
    }
}

impl WeightSource for MapSource {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        self.tensors.get(key).cloned()
    }
    fn get_vector(&self, _key: &str) -> Option<Vec<f32>> {
        None
    }
    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        &*self.arch
    }
    fn num_layers(&self) -> usize {
        NUM_LAYERS
    }
    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        None
    }
    fn vector_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn get_packed_bf16(&self, _key: &str) -> Option<Vec<u8>> {
        None
    }
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("moe-per-expert-tests").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn layer_file(dir: &Path, layer: usize) -> std::path::PathBuf {
    dir.join(crate::format::filenames::layer_weights_filename(layer))
}

#[test]
fn a_separate_tensor_moe_gets_an_expert_store() {
    // The headline regression: this used to write nothing at all.
    let dir = temp_dir("complete");
    let written = write_per_layer_moe_per_expert(&MapSource::complete(), &dir, NUM_LAYERS).unwrap();
    assert_eq!(written, NUM_LAYERS);
    for layer in 0..NUM_LAYERS {
        assert!(
            layer_file(&dir, layer).exists(),
            "layer {layer} not written"
        );
    }
}

#[test]
fn each_layer_file_declares_every_expert() {
    let dir = temp_dir("entries");
    write_per_layer_moe_per_expert(&MapSource::complete(), &dir, NUM_LAYERS).unwrap();
    let bytes = std::fs::read(layer_file(&dir, 0)).unwrap();
    let (_, num_entries, inter, hidden, offsets) = parse_layer_weights_header(&bytes).unwrap();
    assert_eq!(num_entries, NUM_EXPERTS);
    assert_eq!(inter, INTER);
    assert_eq!(hidden, HIDDEN);
    assert_eq!(offsets.len(), NUM_EXPERTS);
}

#[test]
fn every_expert_region_is_inside_the_file() {
    // An offset that is merely a plausible number is the failure mode the
    // whole layer format is designed against; check it holds here too.
    let dir = temp_dir("bounds");
    write_per_layer_moe_per_expert(&MapSource::complete(), &dir, NUM_LAYERS).unwrap();
    let bytes = std::fs::read(layer_file(&dir, 1)).unwrap();
    let (_, _, _, _, offsets) = parse_layer_weights_header(&bytes).unwrap();
    for (gu_off, gu_len, dn_off, dn_len) in offsets {
        assert!(gu_off + gu_len <= bytes.len(), "gate_up region overruns");
        assert!(dn_off + dn_len <= bytes.len(), "down region overruns");
    }
}

#[test]
fn a_layer_without_experts_is_skipped_not_written_short() {
    // A hybrid stack's dense layer. Writing a zero-entry file here would look
    // like a valid expert store to every downstream check.
    let dir = temp_dir("hybrid");
    let written =
        write_per_layer_moe_per_expert(&MapSource::first_layer_only(), &dir, NUM_LAYERS).unwrap();
    assert_eq!(written, 1);
    assert!(layer_file(&dir, 0).exists());
    assert!(!layer_file(&dir, 1).exists());
}

#[test]
fn a_partially_present_layer_is_refused() {
    // Expert 0 resolves but expert 1 does not. Writing the short list would
    // silently drop experts that routing will later select.
    let dir = temp_dir("partial");
    let err =
        write_per_layer_moe_per_expert(&MapSource::partial_layer(), &dir, NUM_LAYERS).unwrap_err();
    let s = err.to_string();
    assert!(s.contains("expert 1"), "{s}");
    assert!(s.contains("only 1 are present"), "{s}");
}

#[test]
fn a_packed_model_is_left_to_the_packed_writer() {
    // Gemma 4 is PackedBF16; this writer must decline it rather than race the
    // packed path to the same filenames.
    struct PackedSource(Box<dyn larql_models::ModelArchitecture>);
    impl WeightSource for PackedSource {
        fn get_tensor(&self, _k: &str) -> Option<(Vec<f32>, usize, usize)> {
            None
        }
        fn get_vector(&self, _k: &str) -> Option<Vec<f32>> {
            None
        }
        fn arch(&self) -> &dyn larql_models::ModelArchitecture {
            &*self.0
        }
        fn num_layers(&self) -> usize {
            NUM_LAYERS
        }
        fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
            None
        }
        fn vector_names(&self) -> Vec<String> {
            Vec::new()
        }
        fn get_packed_bf16(&self, _k: &str) -> Option<Vec<u8>> {
            None
        }
    }
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "gemma4_moe",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
    }));
    let dir = temp_dir("packed");
    let written = write_per_layer_moe_per_expert(&PackedSource(arch), &dir, NUM_LAYERS).unwrap();
    assert_eq!(written, 0, "packed models must not be handled here");
    assert!(!layer_file(&dir, 0).exists());
}

#[test]
fn a_dense_model_is_declined() {
    struct DenseSource(Box<dyn larql_models::ModelArchitecture>);
    impl WeightSource for DenseSource {
        fn get_tensor(&self, _k: &str) -> Option<(Vec<f32>, usize, usize)> {
            None
        }
        fn get_vector(&self, _k: &str) -> Option<Vec<f32>> {
            None
        }
        fn arch(&self) -> &dyn larql_models::ModelArchitecture {
            &*self.0
        }
        fn num_layers(&self) -> usize {
            NUM_LAYERS
        }
        fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
            None
        }
        fn vector_names(&self) -> Vec<String> {
            Vec::new()
        }
        fn get_packed_bf16(&self, _k: &str) -> Option<Vec<u8>> {
            None
        }
    }
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
    }));
    let dir = temp_dir("dense");
    assert_eq!(
        write_per_layer_moe_per_expert(&DenseSource(arch), &dir, NUM_LAYERS).unwrap(),
        0
    );
}

/// ROADMAP P5. GPT-OSS is `PackedMxfp4`: packed on disk, but the
/// safetensors loader dequantises and de-interleaves it, so by the time
/// a `WeightSource` is asked it answers `expert_ffn_*_key` exactly like a
/// separate-tensor model.
///
/// It used to fall through **both** writers — the packed one requires
/// `PackedBF16`, this one required `PerExpert` — and so wrote no expert
/// store at all. Extraction reported success, checksums verified, and the
/// model could not be served. That is the same silent-0-byte failure both
/// writers' doc comments were written about, arriving a third time through
/// a gate that tested an enum value instead of the capability.
#[test]
fn a_packed_mxfp4_model_gets_an_expert_store() {
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
    }));
    // Precondition: this really is the packed-MXFP4 arch, and it really
    // does advertise per-expert keys. Without both, the test below would
    // be exercising some other path.
    assert_eq!(
        arch.expert_format(),
        larql_models::ExpertFormat::PackedMxfp4,
        "fixture is not the packed-MXFP4 arch"
    );
    assert!(
        arch.expert_ffn_gate_key(0, 0).is_some(),
        "arch does not advertise per-expert keys; this writer cannot serve it"
    );

    let mut tensors = HashMap::new();
    for layer in 0..NUM_LAYERS {
        for expert in 0..NUM_EXPERTS {
            insert_expert(&mut tensors, &*arch, layer, expert);
        }
    }
    let source = MapSource { arch, tensors };

    let dir = temp_dir("packed_mxfp4");
    let written = write_per_layer_moe_per_expert(&source, &dir, NUM_LAYERS).unwrap();
    assert_eq!(
        written, NUM_LAYERS,
        "packed-MXFP4 model wrote no expert store — it cannot be served from a vindex"
    );
    for layer in 0..NUM_LAYERS {
        assert!(
            layer_file(&dir, layer).exists(),
            "layer {layer} store missing"
        );
    }
}

/// MXFP4 groups reconstruct at most 15 distinct values, all exactly
/// representable in Q6_K, so the transcode is lossless. Q4_K would
/// re-quantise an already-quantised tensor and discard the checkpoint's
/// own values for nothing — this pins the choice so a later "make every
/// expert store Q4_K for uniformity" cannot quietly undo it.
#[test]
fn packed_mxfp4_experts_are_stored_exactly_as_q6k() {
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
    }));
    let mut tensors = HashMap::new();
    for expert in 0..NUM_EXPERTS {
        insert_expert(&mut tensors, &*arch, 0, expert);
    }
    let source = MapSource { arch, tensors };
    let dir = temp_dir("mxfp4_q6k");
    write_per_layer_moe_per_expert(&source, &dir, NUM_LAYERS).unwrap();

    let bytes = std::fs::read(layer_file(&dir, 0)).unwrap();
    let (format, ..) = parse_layer_weights_header(&bytes).unwrap();
    assert_eq!(
        format,
        crate::format::weights::write_layers::LayerWeightFormat::Q6_K,
        "MXFP4 experts were not stored as Q6_K, so the transcode is lossy"
    );
}

/// A `WeightSource` shaped like `StreamingWeights` against a real GPT-OSS
/// checkpoint: the synthesised per-expert keys resolve to nothing (no shard
/// contains them) and only the raw packed `*_blocks`/`*_scales` answer.
struct RawPackedSource {
    arch: Box<dyn larql_models::ModelArchitecture>,
    raw: HashMap<String, (Vec<u8>, Vec<usize>)>,
}

impl WeightSource for RawPackedSource {
    fn get_tensor(&self, _key: &str) -> Option<(Vec<f32>, usize, usize)> {
        None
    }
    fn get_vector(&self, _key: &str) -> Option<Vec<f32>> {
        None
    }
    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        &*self.arch
    }
    fn num_layers(&self) -> usize {
        NUM_LAYERS
    }
    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        None
    }
    fn vector_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn get_packed_bf16(&self, _key: &str) -> Option<Vec<u8>> {
        None
    }
    fn get_raw_u8(&self, key: &str) -> Option<(Vec<u8>, Vec<usize>)> {
        self.raw.get(key).cloned()
    }
}

fn gpt_oss_arch() -> Box<dyn larql_models::ModelArchitecture> {
    larql_models::detect_from_json(&serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
    }))
}

/// Packed MXFP4 fixture tensors for one layer: fused gate_up
/// `[E, 2*INTER, G, 16]` and down `[E, HIDDEN, G, 16]`, plus their scales.
/// Nibble bytes vary with position so a permuted or mis-sliced decode
/// cannot reproduce them, and scale bytes vary per group so a scale/block
/// misalignment shows up in the values (a constant scale would forgive it).
fn packed_layer_fixture(
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
    raw: &mut HashMap<String, (Vec<u8>, Vec<usize>)>,
) {
    let groups = HIDDEN / larql_models::quant::mxfp4::MXFP4_GROUP_ELEMS;
    let gu_rows = 2 * INTER;
    let fill = |n: usize, seed: usize| -> Vec<u8> {
        (0..n).map(|i| ((i * 37 + seed * 11) % 251) as u8).collect()
    };
    // E8M0 scale bytes around 2^0 (127): keep within ±3 octaves so the
    // dequantised magnitudes stay ordinary f32.
    let scales = |n: usize, seed: usize| -> Vec<u8> {
        (0..n).map(|i| 124 + ((i + seed) % 7) as u8).collect()
    };
    raw.insert(
        arch.packed_gate_up_blocks_key(layer).unwrap(),
        (
            fill(
                NUM_EXPERTS * gu_rows * groups * larql_models::quant::mxfp4::MXFP4_GROUP_BYTES,
                layer,
            ),
            vec![
                NUM_EXPERTS,
                gu_rows,
                groups,
                larql_models::quant::mxfp4::MXFP4_GROUP_BYTES,
            ],
        ),
    );
    raw.insert(
        arch.packed_gate_up_scales_key(layer).unwrap(),
        (
            scales(NUM_EXPERTS * gu_rows * groups, layer),
            vec![NUM_EXPERTS, gu_rows, groups],
        ),
    );
    let dn_groups = INTER / larql_models::quant::mxfp4::MXFP4_GROUP_ELEMS;
    raw.insert(
        arch.packed_down_blocks_key(layer).unwrap(),
        (
            fill(
                NUM_EXPERTS * HIDDEN * dn_groups * larql_models::quant::mxfp4::MXFP4_GROUP_BYTES,
                layer + 100,
            ),
            vec![
                NUM_EXPERTS,
                HIDDEN,
                dn_groups,
                larql_models::quant::mxfp4::MXFP4_GROUP_BYTES,
            ],
        ),
    );
    raw.insert(
        arch.packed_down_scales_key(layer).unwrap(),
        (
            scales(NUM_EXPERTS * HIDDEN * dn_groups, layer + 100),
            vec![NUM_EXPERTS, HIDDEN, dn_groups],
        ),
    );
}

/// The defect this file exists for, fourth appearance: the capability gate
/// passed (GPT-OSS advertises per-expert keys) but the *streaming* source
/// cannot answer them — they name tensors synthesised by the in-RAM loader,
/// present in no shard. The writer skipped every layer and reported success,
/// so a real `larql extract` produced a vindex with no `layers/` directory
/// at all (observed 2026-08-09 on openai/gpt-oss-20b).
///
/// The store must instead be synthesised from the raw packed blocks — and
/// synthesised *identically* to what the per-expert-key path produces from
/// the same dequantised planes, so the two build paths cannot drift.
#[test]
fn a_streaming_shaped_source_synthesises_the_store_from_packed_blocks() {
    let arch = gpt_oss_arch();
    let mut raw = HashMap::new();
    for layer in 0..NUM_LAYERS {
        packed_layer_fixture(&*arch, layer, &mut raw);
    }
    let source = RawPackedSource { arch, raw };

    let dir = temp_dir("streaming_packed");
    let written = write_per_layer_moe_per_expert(&source, &dir, NUM_LAYERS).unwrap();
    assert_eq!(
        written, NUM_LAYERS,
        "raw packed blocks did not synthesise an expert store"
    );

    // Reference arm: dequantise the same packed bytes through the same
    // public helpers and feed the result through the per-expert-key path.
    let arch2 = gpt_oss_arch();
    let mut tensors = HashMap::new();
    for layer in 0..NUM_LAYERS {
        let (gu_blocks, _) = source
            .get_raw_u8(&arch2.packed_gate_up_blocks_key(layer).unwrap())
            .unwrap();
        let (gu_scales, _) = source
            .get_raw_u8(&arch2.packed_gate_up_scales_key(layer).unwrap())
            .unwrap();
        let (dn_blocks, _) = source
            .get_raw_u8(&arch2.packed_down_blocks_key(layer).unwrap())
            .unwrap();
        let (dn_scales, _) = source
            .get_raw_u8(&arch2.packed_down_scales_key(layer).unwrap())
            .unwrap();
        let (gates, ups) = larql_models::quant::mxfp4::split_gate_up_experts(
            &gu_blocks,
            &gu_scales,
            NUM_EXPERTS,
            2 * INTER,
            HIDDEN / 32,
        )
        .unwrap();
        let downs = larql_models::quant::mxfp4::dequantize_all_experts(
            &dn_blocks,
            &dn_scales,
            NUM_EXPERTS,
            HIDDEN,
            INTER / 32,
        )
        .unwrap();
        for e in 0..NUM_EXPERTS {
            let gate_key = arch2.expert_ffn_gate_key(layer, e).unwrap();
            let up_key = arch2.expert_ffn_up_key(layer, e).unwrap();
            let down_key = arch2.expert_ffn_down_key(layer, e).unwrap();
            tensors.insert(gate_key, (gates[e].clone(), INTER, HIDDEN));
            tensors.insert(up_key, (ups[e].clone(), INTER, HIDDEN));
            tensors.insert(down_key, (downs[e].clone(), HIDDEN, INTER));
        }
    }
    let reference = MapSource {
        arch: arch2,
        tensors,
    };
    let ref_dir = temp_dir("streaming_packed_ref");
    write_per_layer_moe_per_expert(&reference, &ref_dir, NUM_LAYERS).unwrap();

    for layer in 0..NUM_LAYERS {
        let a = std::fs::read(layer_file(&dir, layer)).unwrap();
        let b = std::fs::read(layer_file(&ref_dir, layer)).unwrap();
        assert_eq!(
            a, b,
            "layer {layer}: packed-fallback store differs from the \
             per-expert-key store built from the same dequantised planes"
        );
    }
}

/// The other half of the fix: an arch that passes the capability gate but a
/// source that can answer neither the per-expert keys nor the raw packed
/// tensors used to produce `Ok(0)` — success, no store, unservable model.
/// It must now be a build error that names the condition.
#[test]
fn a_capability_declaring_source_that_resolves_nothing_is_an_error() {
    let source = RawPackedSource {
        arch: gpt_oss_arch(),
        raw: HashMap::new(),
    };
    let dir = temp_dir("resolves_nothing");
    let err = write_per_layer_moe_per_expert(&source, &dir, NUM_LAYERS).unwrap_err();
    let s = err.to_string();
    assert!(
        s.contains("no layer yielded an expert store"),
        "error must name the silent-0-byte condition, got: {s}"
    );
    assert!(!layer_file(&dir, 0).exists());
}

/// Blocks present with scales absent is a malformed checkpoint, not a dense
/// layer — the exporter writes the pair together. Skipping it would be the
/// silent-0-byte failure again with a different cause.
#[test]
fn packed_blocks_without_scales_is_refused() {
    let arch = gpt_oss_arch();
    let mut raw = HashMap::new();
    packed_layer_fixture(&*arch, 0, &mut raw);
    let scales_key = arch.packed_gate_up_scales_key(0).unwrap();
    raw.remove(&scales_key);
    let source = RawPackedSource { arch, raw };
    let dir = temp_dir("blocks_no_scales");
    let err = write_per_layer_moe_per_expert(&source, &dir, NUM_LAYERS).unwrap_err();
    assert!(
        err.to_string().contains("scales"),
        "error should name the missing side: {err}"
    );
}
