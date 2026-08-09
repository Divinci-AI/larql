//! MOSS-TTS-Realtime generation loop — text ids in, audio frames out.
//!
//! One outer step per 80 ms acoustic frame (`docs/tts-funnel.md` §1.3):
//!
//! ```text
//! [text id | previous frame's rvq ids]   (one row, 1+rvq columns)
//!         │  summed through 1+rvq embedding tables
//!         ▼
//! backbone decode_step_from_hidden (persistent KV)
//!         ▼  final-normed hidden
//! depth transformer: cached micro-steps over its own fresh KV
//!   position 0 = the hidden row, then one sampled codebook per position,
//!   head i read at position i
//!         ▼
//! frame [rvq ids]; codebook 0 == audio EOS stops the loop
//! ```
//!
//! The depth stage runs the dispatch helpers directly with a per-frame
//! handle set — the cache lives exactly one frame, matching the
//! reference's rebuilt `StaticCache` — and is *cached*, not
//! prefix-recomputed: micro-step `i` appends one position.
//!
//! No tokenizer exists anywhere in this loop. Text ids arrive from the
//! caller; audio ids leave to the caller. Greedy only for now — sampled
//! mode arrives with the reference's non-standard top-p replication.

use ndarray::Array2;

use larql_compute::forward::embed::embed_tables_sum;
use larql_compute::forward::ops::apply_norm;
use larql_models::speech::moss_tts_realtime::{DepthTransformerModel, MossTtsRealtimeConfig};
use larql_models::ModelWeights;

use crate::ffn::{FfnBackend, WeightFfn};
use crate::kv_dispatch::helpers::{
    kv_decode_step_from_hidden_via_dispatch, kv_prefill_from_hidden_via_dispatch,
};
use crate::kv_engine::EngineError;
use crate::KvEngine;

/// A completed (or frame-capped) generation.
#[derive(Debug)]
pub struct MossGeneration {
    /// Every generated frame, EOS frame included, `rvq` ids each.
    pub frames: Vec<Vec<u32>>,
    /// Index of the frame whose codebook 0 was audio EOS, if reached.
    pub eos_at: Option<usize>,
}

impl MossGeneration {
    /// Frames up to (excluding) the EOS frame — what a codec should
    /// decode, matching the reference's crop.
    pub fn emitted(&self) -> &[Vec<u32>] {
        match self.eos_at {
            Some(index) => &self.frames[..index],
            None => &self.frames,
        }
    }
}

/// Generate audio frames greedily until audio EOS or `max_frames`.
///
/// `prefill_matrix` is the `[T, 1+rvq]` prompt (system + voice-clone
/// splice + the text lead with audio BOS placed by the caller — temporal
/// protocol belongs to the caller, not this loop). `audio_tables` are the
/// backbone's per-codebook input tables
/// (`MossTtsRealtimeAuxWeights::audio_embed_tables`), summed with the
/// text embedding per position. `text_queue` holds the not-yet-consumed
/// text ids; once exhausted, `text_pad_id` is fed, per the reference.
/// The backbone `engine` must support multimodal decode
/// (`supports_multimodal`).
#[allow(clippy::too_many_arguments)]
pub fn generate_frames_greedy(
    engine: &mut dyn KvEngine,
    backbone: &ModelWeights,
    backbone_ffn: &dyn FfnBackend,
    audio_tables: &[Array2<f32>],
    depth: &DepthTransformerModel,
    config: &MossTtsRealtimeConfig,
    prefill_matrix: &Array2<u32>,
    text_queue: &[u32],
    text_pad_id: u32,
    max_frames: usize,
) -> Result<MossGeneration, EngineError> {
    generate_frames_greedy_streaming(
        engine,
        backbone,
        backbone_ffn,
        audio_tables,
        depth,
        config,
        prefill_matrix,
        text_queue,
        text_pad_id,
        max_frames,
        |_, _| {},
    )
}

/// [`generate_frames_greedy`] with a per-frame observer: `on_frame(index,
/// codes)` fires the moment each frame exists, before the next backbone
/// step — the streaming surface. Frames are the model's realtime unit
/// (80 ms each at 12.5 Hz), so a consumer feeding a codec + ring buffer
/// hangs directly off this callback.
#[allow(clippy::too_many_arguments)]
pub fn generate_frames_greedy_streaming(
    engine: &mut dyn KvEngine,
    backbone: &ModelWeights,
    backbone_ffn: &dyn FfnBackend,
    audio_tables: &[Array2<f32>],
    depth: &DepthTransformerModel,
    config: &MossTtsRealtimeConfig,
    prefill_matrix: &Array2<u32>,
    text_queue: &[u32],
    text_pad_id: u32,
    max_frames: usize,
    mut on_frame: impl FnMut(usize, &[u32]),
) -> Result<MossGeneration, EngineError> {
    let rvq = config.rvq;
    assert_eq!(
        prefill_matrix.ncols(),
        1 + rvq,
        "prefill matrix must carry one text column plus one per codebook"
    );
    assert_eq!(
        audio_tables.len(),
        rvq,
        "one backbone audio input table per codebook"
    );

    let backbone_tables: Vec<ndarray::ArrayView2<f32>> = std::iter::once(backbone.embed.view())
        .chain(audio_tables.iter().map(|table| table.view()))
        .collect();

    // ── Prefill, and the first frame off its last hidden ──
    let embeds = embed_tables_sum(&backbone_tables, prefill_matrix);
    let last = engine.prefill_from_hidden(backbone, backbone_ffn, &embeds)?;
    let mut frames: Vec<Vec<u32>> = Vec::new();
    let mut eos_at = None;
    let mut text_cursor = 0usize;

    let mut hidden = backbone_final_norm(backbone, &last);
    loop {
        let frame = depth_frame_greedy(depth, &hidden)?;
        let is_eos = frame[0] as usize == config.audio_eos_token();
        on_frame(frames.len(), &frame);
        frames.push(frame);
        if is_eos {
            eos_at = Some(frames.len() - 1);
            break;
        }
        if frames.len() >= max_frames {
            break;
        }

        // ── Next outer step: one text id + the frame just produced ──
        let text_id = match text_queue.get(text_cursor) {
            Some(&id) => id,
            None => text_pad_id,
        };
        text_cursor += 1;
        let mut step_ids = Array2::<u32>::zeros((1, 1 + rvq));
        step_ids[[0, 0]] = text_id;
        for (column, &code) in frames.last().expect("frame just pushed").iter().enumerate() {
            step_ids[[0, column + 1]] = code;
        }
        let step_embed = embed_tables_sum(&backbone_tables, &step_ids);
        let step_hidden = engine.decode_step_from_hidden(backbone, backbone_ffn, &step_embed)?;
        hidden = backbone_final_norm(backbone, &step_hidden);
    }

    Ok(MossGeneration { frames, eos_at })
}

/// One frame through the depth transformer: cached micro-steps over a
/// per-frame handle set, greedy per head.
fn depth_frame_greedy(
    depth: &DepthTransformerModel,
    backbone_hidden: &Array2<f32>,
) -> Result<Vec<u32>, EngineError> {
    let weights = &depth.weights;
    let ffn = WeightFfn { weights };
    let view = larql_models::WeightsView::dense(weights);
    let backend = crate::cpu_engine_backend();
    let rvq = depth.lm_heads.len();

    let mut frame = Vec::with_capacity(rvq);

    // Micro-step 0: the raw backbone hidden is the whole prefix.
    let (mut hidden, mut handles) = kv_prefill_from_hidden_via_dispatch(
        backend.as_ref(),
        view,
        &ffn,
        backbone_hidden,
        None,
        None,
        None,
    )
    .map_err(EngineError::Execution)?
    .ok_or_else(|| EngineError::BackendFailure {
        details: "depth prefill declined by backend".into(),
    })?;

    for micro in 0..rvq {
        let code = head_argmax(depth, micro, weights, &hidden);
        frame.push(code);
        if micro + 1 == rvq {
            break;
        }
        // Micro-step i+1: embed the code just sampled through table i.
        let embed_row = {
            let table = &depth.embed_tables[micro];
            let mut row = Array2::<f32>::zeros((1, table.ncols()));
            row.row_mut(0).assign(&table.row(code as usize));
            row
        };
        hidden = kv_decode_step_from_hidden_via_dispatch(
            backend.as_ref(),
            view,
            &ffn,
            &mut handles,
            &embed_row,
            micro + 1,
            None,
            None,
        )
        .map_err(EngineError::Execution)?
        .ok_or_else(|| EngineError::BackendFailure {
            details: "depth decode step declined by backend".into(),
        })?;
    }

    Ok(frame)
}

fn head_argmax(
    depth: &DepthTransformerModel,
    head: usize,
    weights: &ModelWeights,
    hidden: &Array2<f32>,
) -> u32 {
    let normed = backbone_final_norm(weights, hidden);
    let logits = normed.dot(&depth.lm_heads[head].t());
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in logits.row(0).iter().enumerate() {
        if value > best_value {
            best_value = value;
            best = index;
        }
    }
    best as u32
}

fn backbone_final_norm(weights: &ModelWeights, hidden: &Array2<f32>) -> Array2<f32> {
    apply_norm(
        weights,
        hidden,
        weights.arch.final_norm_key(),
        weights.arch.norm_weight_offset(),
    )
}
