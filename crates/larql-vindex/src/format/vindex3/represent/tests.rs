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
