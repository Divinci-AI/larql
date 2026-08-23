//! Representation compilation over a real encoded container.
//!
//! The gate under test is the one the module header states: compiled bytes
//! must equal, bit for bit, what the runtime would have produced by
//! quantising the same tensor at load. Everything else here guards the
//! properties that make that claim meaningful — the canonical bytes survive,
//! the pack is smaller, and a reader can find its way back to the source.

use super::*;
use crate::format::vindex3::fixtures::{
    dense_f32_model, encode_fixture_container, miniature_glimmer,
};
use crate::format::vindex3::opplan::exec::backend::WeightFormat;
use crate::format::vindex3::opplan::exec::weights::load_weight;
use nvfp4_pack::split;

/// Encode the miniature Glimmer fixture, then compile an NVFP4
/// representation of it.
fn compiled_pair(
    tmp: &tempfile::TempDir,
) -> (std::path::PathBuf, std::path::PathBuf, RepresentReport) {
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("nvfp4.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");
    let report = compile_representation(&src, &out, &RepresentSpec::nvfp4())
        .expect("the dense fixture is 16-aligned throughout");
    (src, out, report)
}

fn index_of(dir: &std::path::Path) -> Vindex3Index {
    serde_json::from_str(&std::fs::read_to_string(dir.join(INDEX_JSON)).unwrap()).unwrap()
}

#[test]
fn compiled_bytes_equal_what_the_loader_would_have_quantised() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);

    // The transient path: open the SOURCE container and quantise at load,
    // exactly as `--backend metal-nvfp4` does today.
    let src_inspection = inspect_container(&src, false).unwrap();
    let src_store = OperandStore::open(&src, &src_inspection).unwrap();
    let src_index = index_of(&src);

    let out_index = index_of(&out);
    let mut checked = 0usize;

    for (rep_id, entry) in &out_index.representations {
        if entry.encoding != DTYPE_NVFP4 {
            continue;
        }
        let (header, payload_start) = read_segment_header(&out.join(&entry.segment)).unwrap();
        let mut file = std::fs::File::open(out.join(&entry.segment)).unwrap();

        let from = entry
            .compiled_from
            .as_ref()
            .expect("compiled pack names its source");
        let source_entry = src_index
            .representations
            .get(from)
            .expect("the source representation survives compilation");
        let (src_header, _) = read_segment_header(&src.join(&source_entry.segment)).unwrap();

        // Byte-equality is only claimable against the encoder that wrote
        // the pack. This build wrote it, so the claim holds here — but the
        // precondition is asserted rather than assumed, so a future encoder
        // change turns this into a clear message instead of a diff.
        let recipe = entry
            .encoder
            .as_ref()
            .expect("a compiled pack names its encoder");
        assert!(
            recipe.is_reproducible_by_this_build(),
            "pack was compiled by {}; this build compiles with {} — compare \
             behaviour, not bytes",
            recipe.name(),
            nvfp4_pack::EncoderRecipe::current().name()
        );

        for t in &header.tensors {
            if t.dtype != DTYPE_NVFP4 {
                continue;
            }
            let layout = PackLayout::derive(&t.shape, &t.name).unwrap();

            // Stored.
            let mut payload = vec![0u8; t.len as usize];
            file.seek(SeekFrom::Start(payload_start + t.offset))
                .unwrap();
            file.read_exact(&mut payload).unwrap();
            let (packed, scales, tensor_scale) = split(&payload, &layout, &t.name).unwrap();

            // Transient, from the canonical source representation.
            let src_tensor = src_header
                .tensors
                .iter()
                .find(|s| s.name == t.name)
                .expect("the pack carries the same tensor names");
            let loaded = load_weight(
                (&src_store).into(),
                &OperandRef {
                    object: entry.object.clone(),
                    tensor: src_tensor.name.clone(),
                    dtype: src_tensor.dtype.clone(),
                    shape: src_tensor.shape.clone(),
                },
                WeightFormat::Nvfp4,
            )
            .unwrap();
            let LoadedWeight::Nvfp4 {
                packed: want_packed,
                scales: want_scales,
                tensor_scale: want_scale,
            } = &loaded
            else {
                panic!("asked for NVFP4, got another format");
            };

            // Compare the logical prefixes: `as_slice` exposes the
            // page-aligned allocation, and comparing padded-to-padded would
            // pass even if the pack persisted the padding.
            assert_eq!(
                packed,
                &want_packed.as_slice()[..want_packed.logical_len()],
                "{rep_id}/{}: stored codes differ from load-time codes",
                t.name
            );
            assert_eq!(
                scales,
                &want_scales.as_slice()[..want_scales.logical_len()],
                "{rep_id}/{}: stored group scales differ",
                t.name
            );
            assert_eq!(
                tensor_scale.to_bits(),
                want_scale.to_bits(),
                "{rep_id}/{}: stored tensor scale differs",
                t.name
            );
            checked += 1;
        }
    }

    assert!(
        checked > 0,
        "the fixture compiled no NVFP4 tensors to check"
    );
}

#[test]
fn the_canonical_representation_survives_byte_for_byte() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);
    let src_index = index_of(&src);
    let out_index = index_of(&out);

    for (id, src_entry) in &src_index.representations {
        let out_entry = out_index
            .representations
            .get(id)
            .unwrap_or_else(|| panic!("canonical representation {id} was dropped"));
        // Compiling an alternative must not touch what it was derived from:
        // the source stays bit-authoritative and its hashes prove it.
        assert_eq!(out_entry.payload_sha256, src_entry.payload_sha256, "{id}");
        assert_eq!(out_entry.segment_sha256, src_entry.segment_sha256, "{id}");
        assert_eq!(out_entry.encoding, src_entry.encoding, "{id}");
        assert!(out_entry.compiled_from.is_none(), "{id} is not compiled");
    }
}

#[test]
fn the_pack_is_smaller_and_names_its_provenance() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = compiled_pair(&tmp);
    let index = index_of(&out);

    assert!(!report.compiled_objects.is_empty());
    for compiled in &report.compiled_objects {
        assert!(
            compiled.compiled_bytes < compiled.source_bytes,
            "{}: {} compiled vs {} source",
            compiled.object,
            compiled.compiled_bytes,
            compiled.source_bytes
        );
        let entry = index
            .representations
            .get(&compiled.representation_id)
            .expect("compiled representation is in the index");
        assert_eq!(entry.encoding, DTYPE_NVFP4);
        // Provenance a derived pack cannot otherwise state.
        let from = entry.compiled_from.as_deref().expect("names its source");
        assert!(
            index.representations.contains_key(from),
            "provenance {from} must resolve inside the same container"
        );
    }
}

#[test]
fn the_graph_marks_the_compiled_representation_approximate() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = compiled_pair(&tmp);
    let graph: crate::format::vindex3::graph::SystemGraph =
        serde_json::from_str(&std::fs::read_to_string(out.join(SYSTEM_GRAPH_JSON)).unwrap())
            .unwrap();

    let compiled: BTreeSet<&str> = report
        .compiled_objects
        .iter()
        .map(|c| c.object.as_str())
        .collect();
    let mut seen = 0usize;
    for object in &graph.objects {
        if !compiled.contains(object.id.as_str()) {
            continue;
        }
        let rep = object
            .representations
            .iter()
            .find(|r| r.encoding == DTYPE_NVFP4)
            .expect("the graph learns the object gained a materialisation");
        // Canonical is what the release ships as ground truth; a 4-bit
        // re-encoding is not that, and mislabelling it would let a profile
        // select an approximation believing it authoritative.
        assert_eq!(rep.fidelity, Fidelity::Approximate);
        assert!(
            object
                .representations
                .iter()
                .any(|r| r.fidelity == Fidelity::Canonical),
            "{}: the canonical representation is still declared",
            object.id
        );
        seen += 1;
    }
    assert_eq!(seen, compiled.len());
}

#[test]
fn non_matrix_tensors_are_carried_verbatim() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = compiled_pair(&tmp);
    let index = index_of(&out);

    // Norms and biases are 1-D: NVFP4 does not apply, and dropping them
    // would leave a pack its consumers had to patch from elsewhere.
    let carried: usize = report
        .compiled_objects
        .iter()
        .map(|c| c.carried_tensors)
        .sum();
    assert!(carried > 0, "the fixture has 1-D tensors to carry");

    for compiled in &report.compiled_objects {
        let entry = index
            .representations
            .get(&compiled.representation_id)
            .unwrap();
        let (header, _) = read_segment_header(&out.join(&entry.segment)).unwrap();
        assert_eq!(
            header.tensors.len(),
            compiled.compiled_tensors + compiled.carried_tensors,
            "{}: pack is complete",
            compiled.object
        );
        for t in &header.tensors {
            if t.dtype != DTYPE_NVFP4 {
                assert_ne!(t.shape.len(), 2, "a 2-D tensor was carried, not compiled");
            }
        }
    }
}

#[test]
fn an_unknown_encoding_is_refused_before_anything_is_written() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("mxfp4.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let spec = RepresentSpec {
        encoding: "MXFP4".into(),
        objects: Vec::new(),
        roles: policy::RolePolicy::default(),
    };
    let err = compile_representation(&src, &out, &spec)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no representation compiler"), "{err}");
    assert!(
        !out.exists(),
        "a refused compile leaves no half-written container"
    );
}

#[test]
fn compiling_twice_does_not_quantise_a_quantised_pack() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, first) = compiled_pair(&tmp);
    let twice = tmp.path().join("twice.vindex3");
    let second = compile_representation(&out, &twice, &RepresentSpec::nvfp4()).unwrap_err();

    // Every object already carries the target encoding, so there is nothing
    // left to compile — and re-encoding an approximation would compound the
    // error silently.
    assert!(
        second.to_string().contains("nothing was compiled"),
        "{second}"
    );
    assert!(!first.compiled_objects.is_empty());
}

#[test]
fn an_object_filter_compiles_only_what_it_names() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("one.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let all = index_of(&src);
    let target = all
        .representations
        .values()
        .map(|e| e.object.clone())
        .find(|o| o.contains("decoder_stack"))
        .expect("the fixture has a decoder stack");

    let spec = RepresentSpec {
        encoding: DTYPE_NVFP4.to_string(),
        objects: vec![target.clone()],
        roles: policy::RolePolicy::default(),
    };
    let report = compile_representation(&src, &out, &spec).unwrap();
    assert_eq!(report.compiled_objects.len(), 1);
    assert_eq!(report.compiled_objects[0].object, target);
}

#[test]
fn a_model_whose_k_is_not_group_aligned_is_refused_whole() {
    // NVFP4 groups 16 elements. The miniature Glimmer fixture is hidden=12,
    // ffn=20 — nothing in it can be grouped, and the refusal names that
    // rather than compiling a container with an empty pack in it.
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("nvfp4.vindex3");
    encode_fixture_container(miniature_glimmer, &checkpoint, &src, "target");

    let err = compile_representation(&src, &out, &RepresentSpec::nvfp4())
        .unwrap_err()
        .to_string();
    assert!(err.contains("nothing was compiled"), "{err}");
    assert!(!out.join(INDEX_JSON).exists(), "no container was written");
}

#[test]
fn the_pack_lives_under_segments_and_is_declared() {
    // Two ways a pack goes missing: written outside `segments/`, or written
    // there but absent from `index.segments`, which
    // `Vindex3Container::segment` refuses to resolve.
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = compiled_pair(&tmp);
    let index = index_of(&out);

    for compiled in &report.compiled_objects {
        let entry = index
            .representations
            .get(&compiled.representation_id)
            .unwrap();
        assert!(
            entry.segment.starts_with("segments/"),
            "pack at {} is outside segments/",
            entry.segment
        );
        assert!(entry.segment.ends_with(".bin"), "{}", entry.segment);
        assert!(
            out.join(&entry.segment).is_file(),
            "{} does not exist on disk",
            entry.segment
        );

        let key = entry.segment.trim_end_matches(".bin");
        assert!(
            index.segments.contains_key(key),
            "segment key {key} is not declared; declared: {:?}",
            index.segments.keys().collect::<Vec<_>>()
        );
    }

    // Nothing stray at the container root.
    for e in std::fs::read_dir(&out).unwrap() {
        let name = e.unwrap().file_name().to_string_lossy().into_owned();
        assert!(
            !name.ends_with(".lyrw") && !name.contains("@"),
            "stray segment at the container root: {name}"
        );
    }
}

#[test]
fn the_default_policy_preserves_the_embedding_and_the_head() {
    // Regression for the shape-based default: an embedding table is a 2-D
    // matrix, and "matrix implies quantise" 4-bits one of the surfaces
    // where 4-bit is least safe. Role decides now.
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = compiled_pair(&tmp);
    let index = index_of(&out);

    let compiled: Vec<&str> = report
        .compiled_objects
        .iter()
        .map(|c| c.object.as_str())
        .collect();
    assert!(
        compiled.iter().any(|o| o.contains("decoder_stack")),
        "decoder linear weights are the point of the default policy"
    );

    // The embedding object may be compiled only if some tensor in it was
    // eligible — under the default, none is.
    for c in &report.compiled_objects {
        if c.object.contains("embedding") || c.object.contains("output_head") {
            panic!("{} was compiled under the conservative default", c.object);
        }
    }
    // And no NVFP4 representation exists for it in the index.
    for (id, e) in &index.representations {
        if e.encoding == DTYPE_NVFP4 {
            assert!(
                !id.contains("embedding") && !id.contains("output_head"),
                "{id} should not have a compiled pack by default"
            );
        }
    }
}

#[test]
fn opting_the_embedding_in_compiles_it() {
    // The escape hatch works, and only for the role named.
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join("aggressive.vindex3");
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");

    let mut spec = RepresentSpec::nvfp4();
    spec.roles = spec.roles.clone().including(policy::Role::Embedding);
    let report = compile_representation(&src, &out, &spec).unwrap();

    assert!(
        report
            .compiled_objects
            .iter()
            .any(|c| c.object.contains("embedding")),
        "explicit opt-in must compile the embedding"
    );
}

#[test]
fn the_report_names_what_the_policy_protected() {
    // A conservative default is only trustworthy if it says what it
    // conserved; a silent policy is indistinguishable from no policy.
    let tmp = tempfile::tempdir().unwrap();
    let (_, _, report) = compiled_pair(&tmp);
    let stack = report
        .compiled_objects
        .iter()
        .find(|c| c.object.contains("decoder_stack"))
        .expect("the stack is compiled");
    assert!(
        !stack.preserved.is_empty(),
        "the decoder stack carries norms the policy preserved"
    );
    let protected: usize = stack.preserved.values().sum();
    assert_eq!(protected, stack.carried_tensors);
}

#[test]
fn a_compiled_pack_records_the_abi_it_was_compiled_against() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, report) = compiled_pair(&tmp);
    let src_index = index_of(&src);
    let index = index_of(&out);

    for c in &report.compiled_objects {
        let entry = index.representations.get(&c.representation_id).unwrap();
        let codec = entry
            .codec
            .as_ref()
            .expect("a compiled pack states its ABI");
        assert_eq!(codec.family, "nvfp4");
        assert_eq!(codec.revision, nvfp4_pack::CodecIdentity::REVISION);
        assert_eq!(codec.group_elems, 16);
        codec.admit().expect("this build implements what it wrote");

        // Provenance survives being copied out of the container.
        let from = entry.compiled_from.as_deref().unwrap();
        let digest = entry
            .source_representation_digest
            .as_deref()
            .expect("a derived pack names the bytes it derives from");
        assert_eq!(digest, src_index.representations[from].payload_sha256);
    }

    // Source-encoded representations carry no ABI: their bytes are the
    // checkpoint's, not this compiler's.
    for (id, e) in &index.representations {
        if e.encoding != DTYPE_NVFP4 {
            assert!(e.codec.is_none(), "{id} should not claim a compiler ABI");
        }
    }
}

#[test]
fn a_future_abi_revision_is_refused_rather_than_decoded() {
    // The whole point: an improved `quantize_nvfp4` must not silently
    // redefine containers already on disk.
    let mut future = nvfp4_pack::CodecIdentity::nvfp4_v1();
    future.revision = nvfp4_pack::CodecIdentity::REVISION + 1;
    let err = future.admit().unwrap_err().to_string();
    assert!(err.contains("another build"), "{err}");
    assert!(err.contains("Recompile"), "{err}");

    let mut alien = nvfp4_pack::CodecIdentity::nvfp4_v1();
    alien.family = "mxfp4".into();
    assert!(alien.admit().unwrap_err().to_string().contains("is not"));

    // Same revision, disagreeing geometry: a corrupted or hand-edited
    // index, and named differently so it is not mistaken for version skew.
    let mut bad = nvfp4_pack::CodecIdentity::nvfp4_v1();
    bad.group_elems = 32;
    let err = bad.admit().unwrap_err().to_string();
    assert!(err.contains("disagrees with its own revision"), "{err}");
}

#[test]
fn the_encoder_recipe_is_recorded_apart_from_the_codec_abi() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, report) = compiled_pair(&tmp);
    let index = index_of(&out);

    for c in &report.compiled_objects {
        let e = index.representations.get(&c.representation_id).unwrap();
        let codec = e.codec.as_ref().unwrap();
        let encoder = e.encoder.as_ref().expect("a pack names its encoder");
        assert_eq!(encoder.name(), "nvfp4-nearest-v1");
        assert!(encoder.is_reproducible_by_this_build());
        // Two separate identities, not one: same decode contract, and a
        // recipe that may change under it.
        assert_eq!(codec.family, "nvfp4");
        assert_ne!(codec.family, encoder.algorithm);
    }
}

#[test]
fn a_different_encoder_is_not_a_refusal_only_a_weaker_claim() {
    // A GPTQ pack decodes through the same kernel and is entirely valid;
    // it simply is not byte-reproducible by a nearest-rounding build.
    // Treating that as corruption would make every encoder improvement a
    // breaking change.
    let gptq = nvfp4_pack::EncoderRecipe {
        algorithm: "nvfp4-gptq".into(),
        revision: 1,
    };
    assert!(!gptq.is_reproducible_by_this_build());
    assert_eq!(gptq.name(), "nvfp4-gptq-v1");

    // The decode contract is unaffected — that is the whole point of
    // keeping the two identities apart.
    nvfp4_pack::CodecIdentity::nvfp4_v1()
        .admit()
        .expect("codec admission does not depend on the encoder recipe");

    let newer = nvfp4_pack::EncoderRecipe {
        algorithm: "nvfp4-nearest".into(),
        revision: 2,
    };
    assert!(!newer.is_reproducible_by_this_build());
}

// ── The loader ladder ────────────────────────────────────────────────────
//
// Compiling the right bytes is worth nothing if execution does not read
// them. These pin the three claims that make a compiled pack real: it is
// selected, it is used *instead of* quantising, and using it changes no
// value.

use crate::format::vindex3::opplan::exec::operands::RepresentationSource;

/// Load one tensor through a store opened under `source`, returning the
/// bound weight and how many tensors the session quantised at load.
fn load_under(
    dir: &std::path::Path,
    source: RepresentationSource,
    object: &str,
    tensor: &str,
    dtype: &str,
    shape: &[usize],
) -> (LoadedWeight, u64) {
    let inspection = inspect_container(dir, false).unwrap();
    let store = OperandStore::open_for(dir, &inspection, Some(DTYPE_NVFP4), source).unwrap();
    let loaded = load_weight(
        (&store).into(),
        &OperandRef {
            object: object.to_string(),
            tensor: tensor.to_string(),
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
        },
        WeightFormat::Nvfp4,
    )
    .unwrap();
    let n = store.runtime_quantised();
    (loaded, n)
}

/// The first compiled tensor of the decoder stack, as (object, name, dtype,
/// shape) in the *source* container.
fn a_compiled_tensor(src: &std::path::Path) -> (String, String, String, Vec<usize>) {
    let index = index_of(src);
    let entry = index
        .representations
        .values()
        .find(|e| e.object.contains("decoder_stack"))
        .unwrap();
    let (header, _) = read_segment_header(&src.join(&entry.segment)).unwrap();
    let t = header
        .tensors
        .iter()
        .find(|t| t.shape.len() == 2 && t.name.contains("q_proj"))
        .expect("the stack has attention projections");
    (
        entry.object.clone(),
        t.name.clone(),
        t.dtype.clone(),
        t.shape.clone(),
    )
}

#[test]
fn stored_and_transient_bind_identical_weights() {
    // Same encoder recipe wrote the pack, so this is a byte claim, not a
    // KL claim. If it ever fails, something in selection, layout or
    // loading differs — and a small numerical difference would be the
    // *wrong* thing to accept here.
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);
    let (object, tensor, dtype, shape) = a_compiled_tensor(&src);

    let (stored, stored_n) = load_under(
        &out,
        RepresentationSource::Stored,
        &object,
        &tensor,
        &dtype,
        &shape,
    );
    let (transient, transient_n) = load_under(
        &out,
        RepresentationSource::Transient,
        &object,
        &tensor,
        &dtype,
        &shape,
    );

    let (
        LoadedWeight::Nvfp4 {
            packed: sp,
            scales: ss,
            tensor_scale: st,
        },
        LoadedWeight::Nvfp4 {
            packed: tp,
            scales: ts,
            tensor_scale: tt,
        },
    ) = (&stored, &transient)
    else {
        panic!("both arms must bind NVFP4");
    };
    assert_eq!(
        &sp.as_slice()[..sp.logical_len()],
        &tp.as_slice()[..tp.logical_len()],
        "codes differ between the stored pack and a fresh quantisation"
    );
    assert_eq!(
        &ss.as_slice()[..ss.logical_len()],
        &ts.as_slice()[..ts.logical_len()],
        "group scales differ"
    );
    assert_eq!(st.to_bits(), tt.to_bits(), "tensor scale differs");

    // And the counter proves the two arms got there by different routes —
    // otherwise this test would pass even if `stored` silently quantised.
    assert_eq!(stored_n, 0, "stored mode quantised at load");
    assert_eq!(transient_n, 1, "transient mode did not invoke the encoder");
}

#[test]
fn transient_ignores_a_present_pack_and_still_encodes() {
    // `transient` is the oracle the compiler is checked against. An arm
    // that fell through to a convenient pack would silently stop being
    // one, and every parity result after that would be vacuous.
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);
    let (object, tensor, dtype, shape) = a_compiled_tensor(&src);

    let (_, n) = load_under(
        &out,
        RepresentationSource::Transient,
        &object,
        &tensor,
        &dtype,
        &shape,
    );
    assert_eq!(n, 1, "the pack exists, and transient must encode anyway");
}

#[test]
fn stored_forbids_manufacturing_and_names_the_tensor() {
    // The invariant is about work, not coverage. Opening a container with
    // no pack is fine; being asked to quantise one of its tensors is not.
    let tmp = tempfile::tempdir().unwrap();
    let (src, _, _) = compiled_pair(&tmp);
    let (object, tensor, dtype, shape) = a_compiled_tensor(&src);

    let inspection = inspect_container(&src, false).unwrap();
    let store = OperandStore::open_for(
        &src,
        &inspection,
        Some(DTYPE_NVFP4),
        RepresentationSource::Stored,
    )
    .expect("opening a container without packs is not itself a violation");

    let err = match load_weight(
        (&store).into(),
        &OperandRef {
            object,
            tensor: tensor.clone(),
            dtype,
            shape,
        },
        WeightFormat::Nvfp4,
    ) {
        Ok(_) => panic!("stored mode quantised at load"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains(&tensor), "the refusal names the tensor: {err}");
    assert!(err.contains("forbids manufacturing"), "{err}");
    assert!(err.contains("represent"), "and says how to fix it: {err}");
    assert_eq!(
        store.runtime_quantised(),
        0,
        "a refused load is not a count"
    );
}

#[test]
fn stored_binds_a_policy_preserved_object_without_complaint() {
    // The embedding has no pack *by design*. Strict mode must not treat a
    // deliberate protection as a missing artifact, or a conservative role
    // policy and a strict source policy could never be used together.
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, _) = compiled_pair(&tmp);
    let inspection = inspect_container(&out, false).unwrap();
    let store = OperandStore::open_for(
        &out,
        &inspection,
        Some(DTYPE_NVFP4),
        RepresentationSource::Stored,
    )
    .expect("a preserved object is not a missing pack");

    let sel = store.selection();
    if let Some((_, embed)) = sel.iter().find(|(k, _)| k.contains("embedding")) {
        assert!(!embed.stored, "the embedding has no pack");
        assert_ne!(embed.encoding, DTYPE_NVFP4);
    }
    let stack = sel
        .iter()
        .find(|(k, _)| k.contains("decoder_stack"))
        .unwrap()
        .1;
    assert!(stack.stored, "the stack does have one and must use it");
}

#[test]
fn auto_prefers_the_pack_but_falls_back() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, out, _) = compiled_pair(&tmp);
    let (object, tensor, dtype, shape) = a_compiled_tensor(&src);

    // Pack present: used, nothing quantised.
    let (_, n) = load_under(
        &out,
        RepresentationSource::Auto,
        &object,
        &tensor,
        &dtype,
        &shape,
    );
    assert_eq!(n, 0, "auto did not use the available pack");

    // Pack absent: manufactured rather than refused.
    let (_, n) = load_under(
        &src,
        RepresentationSource::Auto,
        &object,
        &tensor,
        &dtype,
        &shape,
    );
    assert_eq!(
        n, 1,
        "auto must fall back to encoding when nothing is stored"
    );
}

#[test]
fn selection_reports_which_objects_came_from_a_pack() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, out, _) = compiled_pair(&tmp);
    let inspection = inspect_container(&out, false).unwrap();

    let stored = OperandStore::open_for(
        &out,
        &inspection,
        Some(DTYPE_NVFP4),
        RepresentationSource::Auto,
    )
    .unwrap();
    let sel = stored.selection();
    let stack = sel
        .iter()
        .find(|(k, _)| k.contains("decoder_stack"))
        .expect("the stack is bound")
        .1;
    assert!(
        stack.stored,
        "the decoder stack has a pack and should use it"
    );
    assert_eq!(stack.encoding, DTYPE_NVFP4);

    // The embedding is preserved by policy, so it has no pack and binds
    // canonically — the selection must say so rather than implying the
    // whole model is 4-bit.
    if let Some((_, embed)) = sel.iter().find(|(k, _)| k.contains("embedding")) {
        assert!(!embed.stored);
        assert_ne!(embed.encoding, DTYPE_NVFP4);
    }
}
