//! Representation compilation: add a compiled physical encoding of an
//! object to a container, without changing what the model *is*.
//!
//! ```text
//! source tensors → representation compiler → persisted representation pack
//!                                                      ↓
//!                                          a profile selects the pack
//! ```
//!
//! This is a third verb alongside the two that already exist, and the
//! distinction is the point:
//!
//! - **COMPILE** materialises overlay *meaning* into rewritten segments.
//! - **COMPACT** reorganises bytes while preserving meaning exactly
//!   (`SemanticDiff(input, output) == ∅`).
//! - **REPRESENT** adds a *lossy alternative encoding* beside the
//!   canonical bytes. It preserves neither byte-equality (the pack is new
//!   bytes) nor exact semantics (4-bit is an approximation), so it cannot
//!   hide behind either gate and carries its own: the compiled bytes must
//!   equal, bit for bit, what the runtime would have produced by
//!   quantising at load.
//!
//! That gate is what makes the operation safe, and it is met by
//! construction rather than by comparison: the compiler runs the *same two
//! steps* the load path runs — [`OperandSource::load`] then
//! [`quantize_nvfp4`] — so persisted and transient bytes cannot diverge
//! without one of those two changing for both.
//!
//! ## What it is for
//!
//! A 30B BF16 source is tens of gigabytes on disk and pays the
//! quantisation cost on every cold load. Neither is a property of the
//! model; both are properties of having only one stored representation.
//! Compiling one changes the artifact, not the semantics — and at K3
//! scale, where sparse expert fetches make bytes-per-expert an input to
//! the inference algorithm rather than a storage detail, it stops being a
//! convenience.
//!
//! ## What it does not do
//!
//! It does not replace the canonical representation. The source bytes stay
//! in the container and stay canonical; the compiled pack is added beside
//! them with [`Fidelity::Approximate`]. A profile then selects between
//! representations that *exist* — the rule
//! [`super::variants`] already states, and the reason a compiler is needed
//! at all: a profile cannot turn one encoding's bytes into another's.

pub mod nvfp4_pack;

use std::collections::BTreeSet;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use super::encode::segment::{read_segment_header, write_segment, PlannedTensor};
use super::encode::REPRESENTATION_ID_SEP;
use super::graph::object::{Fidelity, Representation};
use super::index::{RepresentationEntry, Vindex3Index};
use super::inspect::inspect_container;
use super::opplan::exec::operands::{OperandSource, OperandStore};
use super::opplan::exec::weights::{quantize_nvfp4, LoadedWeight};
use super::opplan::OperandRef;
use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;
use nvfp4_pack::{PackLayout, DTYPE_NVFP4};

/// Filename of the system graph, carried beside the index.
const SYSTEM_GRAPH_JSON: &str = "system_graph.json";

/// What one representation compilation produced.
#[derive(Debug, Clone)]
pub struct RepresentReport {
    /// Objects that gained a compiled representation.
    pub compiled_objects: Vec<CompiledObject>,
    /// Segments carried across untouched.
    pub linked_segments: usize,
}

/// One object's compiled pack.
#[derive(Debug, Clone)]
pub struct CompiledObject {
    pub object: String,
    pub representation_id: String,
    /// Tensors re-encoded into the pack.
    pub compiled_tensors: usize,
    /// Tensors copied verbatim because they are not matrices the encoding
    /// applies to (norms, biases, 1-D vectors).
    pub carried_tensors: usize,
    pub source_bytes: u64,
    pub compiled_bytes: u64,
}

impl CompiledObject {
    /// Compression of the compiled pack against the bytes it was compiled
    /// from. `0.0` when the pack is empty.
    pub fn compression(&self) -> f64 {
        if self.compiled_bytes == 0 {
            return 0.0;
        }
        self.source_bytes as f64 / self.compiled_bytes as f64
    }
}

/// Which objects to compile, and into what.
#[derive(Debug, Clone)]
pub struct RepresentSpec {
    /// Target encoding. Only [`DTYPE_NVFP4`] today; the match below is the
    /// single place a second encoding is added.
    pub encoding: String,
    /// Objects to compile. Empty means every object carrying a matrix the
    /// encoding applies to.
    pub objects: Vec<String>,
}

impl RepresentSpec {
    pub fn nvfp4() -> Self {
        Self {
            encoding: DTYPE_NVFP4.to_string(),
            objects: Vec::new(),
        }
    }

    fn wants(&self, object: &str) -> bool {
        self.objects.is_empty() || self.objects.iter().any(|o| o == object)
    }
}

/// Compile `spec`'s representations of the container at `src` into a new
/// container at `out`.
///
/// The output carries every original segment plus one compiled pack per
/// targeted object. Untouched segments are hard-linked where the
/// filesystem allows, so adding a representation costs the pack's bytes,
/// not the container's.
pub fn compile_representation(
    src: &Path,
    out: &Path,
    spec: &RepresentSpec,
) -> Result<RepresentReport, VindexError> {
    if spec.encoding != DTYPE_NVFP4 {
        return Err(VindexError::Parse(format!(
            "encoding `{}` has no representation compiler; known: {DTYPE_NVFP4}",
            spec.encoding
        )));
    }

    let raw_index = std::fs::read_to_string(src.join(INDEX_JSON))?;
    let mut index: Vindex3Index = serde_json::from_str(&raw_index)
        .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;

    let inspection = inspect_container(src, false)?;
    let store = OperandStore::open(src, &inspection)?;
    let source = OperandSource::from(&store);

    std::fs::create_dir_all(out)?;

    let mut report = RepresentReport {
        compiled_objects: Vec::new(),
        linked_segments: 0,
    };

    // Every existing representation travels unchanged: compiling an
    // alternative never removes the canonical bytes it was derived from.
    let existing: Vec<(String, RepresentationEntry)> = index
        .representations
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (_, entry) in &existing {
        let from = src.join(&entry.segment);
        let to = out.join(&entry.segment);
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if std::fs::hard_link(&from, &to).is_err() {
            std::fs::copy(&from, &to)?;
        }
        report.linked_segments += 1;
    }

    let mut added: Vec<(String, RepresentationEntry)> = Vec::new();
    let mut compiled_object_ids: BTreeSet<String> = BTreeSet::new();

    for (rep_id, entry) in &existing {
        if !spec.wants(&entry.object) {
            continue;
        }
        // One compiled pack per object. An object already carrying the
        // target encoding is left alone rather than re-encoded — a second
        // pass must not quantise a quantised pack.
        let target_id = format!("{}{REPRESENTATION_ID_SEP}{}", entry.object, spec.encoding);
        if index.representations.contains_key(&target_id)
            || compiled_object_ids.contains(&entry.object)
        {
            continue;
        }

        let src_segment = src.join(&entry.segment);
        let (header, payload_start) = read_segment_header(&src_segment)?;

        // Plan first: which tensors the encoding applies to, and how long
        // each becomes. Nothing is written until every length is known,
        // because the segment writer needs the table before the payload.
        let mut planned: Vec<PlannedTensor> = Vec::new();
        let mut layouts: Vec<(String, Option<PackLayout>)> = Vec::new();
        let mut compiled_tensors = 0usize;
        let mut carried_tensors = 0usize;
        let mut source_bytes = 0u64;

        for t in &header.tensors {
            match PackLayout::derive(&t.shape, &t.name) {
                Ok(layout) => {
                    source_bytes += t.len;
                    compiled_tensors += 1;
                    planned.push(PlannedTensor {
                        relative_name: t.name.clone(),
                        source_name: t.name.clone(),
                        dtype: DTYPE_NVFP4.to_string(),
                        shape: t.shape.clone(),
                        len: layout.total_len as u64,
                    });
                    layouts.push((t.name.clone(), Some(layout)));
                }
                Err(_) => {
                    // Not a matrix this encoding applies to — a norm, a
                    // bias, a 1-D vector. Carried verbatim so the pack is a
                    // complete object, not a partial one its consumers
                    // would have to patch from elsewhere.
                    carried_tensors += 1;
                    planned.push(PlannedTensor {
                        relative_name: t.name.clone(),
                        source_name: t.name.clone(),
                        dtype: t.dtype.clone(),
                        shape: t.shape.clone(),
                        len: t.len,
                    });
                    layouts.push((t.name.clone(), None));
                }
            }
        }

        if compiled_tensors == 0 {
            // Nothing to compile here; the object keeps its canonical
            // representation alone rather than gaining an identical copy
            // under a misleading name.
            continue;
        }

        let out_segment = super::write::segment_path(out, &target_id);
        let segment_rel = super::write::segment_path(Path::new(""), &target_id);
        if let Some(parent) = out_segment.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut src_file = std::fs::File::open(&src_segment)?;
        let written = write_segment(&out_segment, &target_id, planned, |name, w, tap| {
            let tensor = header
                .tensors
                .iter()
                .find(|t| t.name == name)
                .expect("planned from the same header");
            let layout = layouts
                .iter()
                .find(|(n, _)| n == name)
                .and_then(|(_, l)| *l);

            match layout {
                Some(layout) => {
                    // The load path is `OperandSource::load` then
                    // `quantize_nvfp4`. Running exactly those two here is
                    // what makes persisted bytes bit-identical to
                    // transient ones — see this module's header.
                    let values = source.load(&OperandRef {
                        object: entry.object.clone(),
                        tensor: tensor.name.clone(),
                        dtype: tensor.dtype.clone(),
                        shape: tensor.shape.clone(),
                    })?;
                    let quantised = quantize_nvfp4(&values, layout.rows, layout.k, &tensor.name)?;
                    let LoadedWeight::Nvfp4 {
                        packed,
                        scales,
                        tensor_scale,
                    } = &quantised
                    else {
                        return Err(VindexError::Parse(format!(
                            "tensor `{}`: NVFP4 quantiser returned another format",
                            tensor.name
                        )));
                    };
                    // `AlignedBytes::as_slice` exposes the whole
                    // page-aligned allocation; only `logical_len` bytes are
                    // the tensor. Persisting the padding would write 16 KB
                    // tails of zeros into the pack and put the file out of
                    // step with what `PackLayout` says it holds.
                    let mut bytes = Vec::with_capacity(layout.total_len);
                    bytes.extend_from_slice(&packed.as_slice()[..packed.logical_len()]);
                    bytes.extend_from_slice(&scales.as_slice()[..scales.logical_len()]);
                    bytes.extend_from_slice(&tensor_scale.to_le_bytes());
                    if bytes.len() != layout.total_len {
                        return Err(VindexError::Parse(format!(
                            "tensor `{}`: packed {} bytes, layout implies {}",
                            tensor.name,
                            bytes.len(),
                            layout.total_len
                        )));
                    }
                    w.write_all(&bytes)?;
                    tap(&bytes);
                    Ok(bytes.len() as u64)
                }
                None => {
                    src_file.seek(SeekFrom::Start(payload_start + tensor.offset))?;
                    let mut remaining = tensor.len;
                    let mut buf = vec![0u8; 1 << 20];
                    while remaining > 0 {
                        let take = remaining.min(buf.len() as u64) as usize;
                        src_file.read_exact(&mut buf[..take])?;
                        w.write_all(&buf[..take])?;
                        tap(&buf[..take]);
                        remaining -= take as u64;
                    }
                    Ok(tensor.len)
                }
            }
        })?;

        report.compiled_objects.push(CompiledObject {
            object: entry.object.clone(),
            representation_id: target_id.clone(),
            compiled_tensors,
            carried_tensors,
            source_bytes,
            compiled_bytes: written.payload_bytes,
        });
        compiled_object_ids.insert(entry.object.clone());

        added.push((
            target_id,
            RepresentationEntry {
                object: entry.object.clone(),
                encoding: spec.encoding.clone(),
                segment: segment_rel.to_string_lossy().into_owned(),
                tensor_count: written.tensor_count,
                payload_bytes: written.payload_bytes,
                payload_sha256: written.payload_sha256,
                segment_sha256: written.segment_sha256,
                compiled_from: Some(rep_id.clone()),
            },
        ));
    }

    if added.is_empty() {
        return Err(VindexError::Parse(format!(
            "no object in this container has a matrix `{}` applies to; \
             nothing was compiled and no container was written",
            spec.encoding
        )));
    }

    for (id, entry) in added {
        index.representations.insert(id, entry);
    }

    // The graph learns the object now has a second materialisation, marked
    // approximate: a profile may select it, and nothing may mistake it for
    // the bit-authoritative source.
    let graph_path = out.join(SYSTEM_GRAPH_JSON);
    let src_graph = src.join(SYSTEM_GRAPH_JSON);
    if src_graph.exists() {
        let graph_raw = std::fs::read_to_string(&src_graph)?;
        let mut graph: super::graph::SystemGraph = serde_json::from_str(&graph_raw)
            .map_err(|e| VindexError::Parse(format!("parse {SYSTEM_GRAPH_JSON}: {e}")))?;
        for object in &mut graph.objects {
            if compiled_object_ids.contains(&object.id)
                && !object
                    .representations
                    .iter()
                    .any(|r| r.encoding == spec.encoding)
            {
                object.representations.push(Representation {
                    encoding: spec.encoding.clone(),
                    fidelity: Fidelity::Approximate,
                });
            }
        }
        let serialised = serde_json::to_string_pretty(&graph)
            .map_err(|e| VindexError::Parse(format!("serialise {SYSTEM_GRAPH_JSON}: {e}")))?;
        std::fs::write(&graph_path, serialised)?;
    }

    for aux in ["moe_manifest.json", "tokenizer.json"] {
        let from = src.join(aux);
        if from.exists() {
            std::fs::copy(&from, out.join(aux))?;
        }
    }

    // Index last: a crash mid-compile leaves a directory that is not yet a
    // container, matching the encode writer's ordering contract.
    let serialised = serde_json::to_string_pretty(&index)
        .map_err(|e| VindexError::Parse(format!("serialise {INDEX_JSON}: {e}")))?;
    std::fs::write(out.join(INDEX_JSON), serialised)?;

    Ok(report)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
