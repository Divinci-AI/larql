//! Import a real MoE layer from a loaded model into a VINDEX3 container
//! (container ladder c8).
//!
//! # Why this is the rung after execution parity
//!
//! Every VINDEX3 parity result so far bound its operands out of a **VINDEX2**
//! file. That was deliberate — `examples/vindex3_gemma_layer_parity.rs` says
//! why: binding over the incumbent's own bytes leaves execution as the only
//! candidate cause of a mismatch. With execution closed, re-extraction becomes
//! the one remaining variable, and this is what introduces it.
//!
//! # Verbatim, or the comparison proves nothing
//!
//! Regions are copied **byte for byte** from the source. No transcoding, no
//! requantisation, no repacking — the same rule §9.1 puts on profiles, applied
//! to the importer. Two reasons, and the second is the load-bearing one:
//!
//! - A container that re-quantised on the way in could not be compared
//!   bit-identically against the layer bound over the source bytes, so the
//!   rung would have no gate.
//! - Anything this module *computed* would be a second implementation of the
//!   producer, and agreement between two implementations of the same idea is
//!   not evidence that either matches the model.
//!
//! So the only decisions here are *descriptive*: which role each byte range
//! plays, what shape it is stored in, and which programme interprets it.
//!
//! # Fused gate+up is described, not split
//!
//! Gemma stores gate and up as one fused range per expert. Splitting it would
//! be a transformation; instead the region is declared [`RegionRole::GateUpFused`]
//! and `gated-mlp-v1` — which admits both renderings — interprets it. The
//! decomposed case is equally importable by declaring two regions.

use crate::format::lyrw2::bank::{BankDescriptor, BankKind};
use crate::format::lyrw2::browse_mode::BrowseMode;
use crate::format::lyrw2::plan::Lyrw2Plan;
use crate::format::lyrw2::region_format::{Packing, RegionFormat};
use crate::format::lyrw2::region_role::RegionRole;
use crate::format::lyrw2::region_schema::RegionSchema;
use crate::format::lyrw2::write::Lyrw2Writer;
use crate::format::moe_manifest::bank_ref::{BankRef, ExpertDims};
use crate::format::moe_manifest::layer::{Combine, InputSpace, MoeLayer, Reduction};
use crate::format::moe_manifest::router::{
    Router, RouterPostProcessing, ScoreActivation, Selection,
};
use crate::format::moe_manifest::MoeManifest;
use crate::VindexError;

use super::write::{ContainerSpec, SegmentSource};

/// Schema indices within the routed bank. Two regions per expert.
const SCHEMA_GATE_UP: u16 = 0;
const SCHEMA_DOWN: u16 = 1;
const ROUTED_BANK_ID: u16 = 0;
const REGIONS_PER_EXPERT: u16 = 2;

/// The programme that interprets a gated MLP in either rendering.
pub const GATED_MLP_PROGRAMME: &str = "gated-mlp-v1";

/// One MoE layer's bytes and the shape they are stored in.
///
/// Borrowed, never owned: the caller already has these mapped, and copying a
/// multi-gigabyte layer to describe it would defeat the point.
pub struct MoeLayerSource<'a> {
    /// Which layer of the model this is.
    pub layer: u32,
    /// Per-expert fused gate+up bytes, exactly as stored.
    pub experts_gate_up: Vec<&'a [u8]>,
    /// Per-expert down bytes, exactly as stored.
    pub experts_down: Vec<&'a [u8]>,
    /// Stored format of both region kinds.
    pub format: RegionFormat,
    /// Residual width entering the layer.
    pub hidden_size: u32,
    /// Per-expert intermediate width, as **stored** — Gemma pads this, and the
    /// padded extent is what the bytes actually contain. The semantic width is
    /// a view over it, recorded by the manifest's `expert_dims`, not here.
    pub stored_intermediate: u32,
    /// Semantic intermediate width the operation means.
    pub semantic_intermediate: u32,
    /// Experts activated per token.
    pub top_k: u32,
}

impl MoeLayerSource<'_> {
    fn num_experts(&self) -> u32 {
        self.experts_gate_up.len() as u32
    }

    /// Refuse a source that cannot describe a bank, before any byte is written.
    fn check(&self) -> Result<(), VindexError> {
        let refuse = |why: String| Err(VindexError::Parse(format!("layer {}: {why}", self.layer)));
        if self.experts_gate_up.is_empty() {
            return refuse("no experts to import".into());
        }
        if self.experts_gate_up.len() != self.experts_down.len() {
            return refuse(format!(
                "{} gate_up experts but {} down experts — every expert needs both",
                self.experts_gate_up.len(),
                self.experts_down.len()
            ));
        }
        if self.top_k == 0 || self.top_k > self.num_experts() {
            return refuse(format!(
                "top_k {} is not selectable from {} experts",
                self.top_k,
                self.num_experts()
            ));
        }
        if self.semantic_intermediate > self.stored_intermediate {
            return refuse(format!(
                "semantic intermediate {} exceeds the stored {} — a view can \
                 narrow the stored extent, never widen it",
                self.semantic_intermediate, self.stored_intermediate
            ));
        }
        // Ragged expert slices mean the source's own layout is inconsistent;
        // importing them would bake that into a container that then fails at
        // bind with no trace back to here.
        let gu = self.experts_gate_up[0].len();
        if let Some(e) = self.experts_gate_up.iter().position(|s| s.len() != gu) {
            return refuse(format!(
                "expert {e}'s gate_up is {} bytes, expert 0's is {gu} — experts \
                 in one bank must share a layout",
                self.experts_gate_up[e].len()
            ));
        }
        let dn = self.experts_down[0].len();
        if let Some(e) = self.experts_down.iter().position(|s| s.len() != dn) {
            return refuse(format!(
                "expert {e}'s down is {} bytes, expert 0's is {dn} — experts \
                 in one bank must share a layout",
                self.experts_down[e].len()
            ));
        }
        Ok(())
    }
}

/// Write `source`'s regions into one LYRW v2 segment, verbatim.
///
/// `staging` is where the writer builds the file; the bytes are read back and
/// returned, so the caller decides where the container finally lives.
pub fn segment_bytes_for_layer(
    source: &MoeLayerSource<'_>,
    staging: &std::path::Path,
) -> Result<Vec<u8>, VindexError> {
    source.check()?;
    let experts = source.num_experts();

    let bank = BankDescriptor {
        bank_id: ROUTED_BANK_ID,
        kind: BankKind::Routed,
        num_entries: experts,
        input_dim: source.hidden_size,
        intermediate_dim: source.stored_intermediate,
        output_dim: source.hidden_size,
        region_schema_count: REGIONS_PER_EXPERT,
        browse: BrowseMode::None,
    };

    // Shapes are the **stored** ones. `gate_up` is one fused range of
    // 2 x intermediate rows; `down` contracts the intermediate back to hidden.
    let schemas = vec![
        RegionSchema::unpaired(
            SCHEMA_GATE_UP,
            RegionRole::GateUpFused,
            source.format,
            Packing::RowMajor,
            source.stored_intermediate * 2,
            source.hidden_size,
        ),
        RegionSchema::unpaired(
            SCHEMA_DOWN,
            RegionRole::Down,
            source.format,
            Packing::RowMajor,
            source.hidden_size,
            source.stored_intermediate,
        ),
    ];

    let plan = Lyrw2Plan::single_segment(source.layer, bank, schemas);
    let mut writer = Lyrw2Writer::create(staging, plan)
        .map_err(|e| VindexError::Parse(format!("create LYRW v2 writer: {e}")))?;

    // Entry order is (expert, then schema), matching the plan's declaration.
    for expert in 0..experts as usize {
        writer
            .write_region(source.experts_gate_up[expert])
            .map_err(|e| VindexError::Parse(format!("expert {expert} gate_up: {e}")))?;
        writer
            .write_region(source.experts_down[expert])
            .map_err(|e| VindexError::Parse(format!("expert {expert} down: {e}")))?;
    }
    writer
        .finish()
        .map_err(|e| VindexError::Parse(format!("finish LYRW v2 segment: {e}")))?;

    std::fs::read(staging).map_err(VindexError::Io)
}

/// The manifest entry describing `source`.
pub fn manifest_layer(source: &MoeLayerSource<'_>, storage: &str) -> MoeLayer {
    MoeLayer {
        layer: source.layer,
        input_space: InputSpace::Residual,
        router: Router {
            scores: format!("layers.{}.router.weight", source.layer),
            activation: ScoreActivation::Softmax,
            selection: Selection::TopK {
                k: source.top_k as usize,
            },
            post: RouterPostProcessing::default(),
            shared_expert_sink: false,
        },
        transforms: None,
        routed_bank: BankRef {
            experts: source.num_experts(),
            programme: GATED_MLP_PROGRAMME.to_string(),
            storage: storage.to_string(),
            expert_dims: Some(ExpertDims {
                input: source.hidden_size,
                // The **semantic** width: what the operation means, which is
                // what a kernel must contract over. The stored extent lives in
                // the bank descriptor above, and the two differ on Gemma.
                intermediate: source.semantic_intermediate,
                output: source.hidden_size,
            }),
        },
        shared_bank: None,
        reduction: Reduction::GateWeightedSum,
        routed_output_norm: None,
        combine: Combine::ResidualAdd,
    }
}

/// Build a one-layer VINDEX3 container specification from a real model layer.
///
/// The c8 rung: the first container whose bytes came out of a real checkpoint
/// rather than a fixture generator.
pub fn import_one_layer(
    source: &MoeLayerSource<'_>,
    model: &str,
    family: &str,
    num_layers: usize,
    staging: &std::path::Path,
) -> Result<ContainerSpec, VindexError> {
    let storage = routed_storage_key(source.layer);
    let bytes = segment_bytes_for_layer(source, staging)?;
    let manifest = MoeManifest::new(vec![manifest_layer(source, &storage)]);
    // Refuse here rather than at `write_container`: a manifest this module
    // built and cannot itself validate would push the failure into the
    // container it just wrote.
    let json = serde_json::to_string(&manifest)
        .map_err(|e| VindexError::Parse(format!("serialise manifest: {e}")))?;
    MoeManifest::parse(&json)?;

    Ok(ContainerSpec {
        model: model.to_string(),
        family: family.to_string(),
        hidden_size: source.hidden_size as usize,
        num_layers,
        manifest,
        segments: vec![SegmentSource {
            key: storage,
            bytes,
        }],
    })
}

/// Segment key for a routed layer. Composed, never globbed (§12.1).
pub fn routed_storage_key(layer: u32) -> String {
    format!("routed/layer_{layer:03}")
}

#[cfg(test)]
#[path = "import_tests.rs"]
mod tests;
