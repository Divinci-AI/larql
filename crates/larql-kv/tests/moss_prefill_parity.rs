//! Funnel step 2 (`docs/tts-funnel.md` §3): MOSS-TTS-Realtime prefill
//! parity against the reference dump.
//!
//! Two comparisons, in causal order:
//!
//! 1. **The summed-embedding algebra** — `embed_tables_sum` over the
//!    `[T, 17]` prompt matrix vs the reference's per-position summed
//!    input embeddings. This isolates the new input primitive from the
//!    transformer entirely.
//! 2. **The backbone forward** — `StandardEngine::prefill_from_hidden`
//!    over those embeddings, final norm applied, vs the reference's
//!    `last_hidden_state` at the last position. This is the boundary
//!    crossing: LARQL's existing Qwen3 execution fed by a non-text input
//!    algebra it knows nothing about.
//!
//! NOT FOR CI: needs the ~4.3 GB checkpoint and the step-0 dump. Run:
//!
//! ```text
//! MOSS_TTS_REALTIME_DIR=<hf snapshot dir> \
//! MOSS_PARITY_BIN_DIR=<parity-dump/bin dir> \
//! cargo test -p larql-kv --test moss_prefill_parity -- --ignored --nocapture
//! ```
//!
//! The bin fixtures are flat little-endian arrays produced by
//! `moss_parity_dump.py --export-bin` (jarvis-voice), shapes in
//! `manifest.json` alongside them.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::Array2;

use larql_compute::forward::ops::apply_norm;
use larql_inference::ffn::WeightFfn;
use larql_inference::larql_models::loading::safetensors::load_model_dir;
use larql_inference::larql_models::speech::moss_tts_realtime::{
    load_moss_tts_aux_from_safetensors, MossTtsRealtimeConfig,
};
use larql_inference::KvEngine;
use larql_kv::engines::standard::StandardEngine;

const ENV_MODEL_DIR: &str = "MOSS_TTS_REALTIME_DIR";
const ENV_BIN_DIR: &str = "MOSS_PARITY_BIN_DIR";

/// Bit-exactness is not expected across engines (accumulation order in
/// fp32 GEMMs differs); these bounds are the "first gate" tolerances and
/// exist to be tightened, not relaxed.
const EMBED_MAX_ABS: f32 = 1e-5;
const HIDDEN_MAX_ABS: f32 = 1e-2;
const HIDDEN_MIN_COSINE: f64 = 0.999_99;

struct BinFixtures {
    shapes: HashMap<String, Vec<usize>>,
    dir: PathBuf,
}

impl BinFixtures {
    fn open(dir: &Path) -> Self {
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("manifest.json")).expect("bin manifest"),
        )
        .expect("bin manifest json");
        let shapes = manifest
            .as_object()
            .expect("manifest object")
            .iter()
            .map(|(key, meta)| {
                let shape = meta["shape"]
                    .as_array()
                    .expect("shape array")
                    .iter()
                    .map(|v| v.as_u64().expect("dim") as usize)
                    .collect();
                (key.clone(), shape)
            })
            .collect();
        Self {
            shapes,
            dir: dir.to_path_buf(),
        }
    }

    fn shape(&self, key: &str) -> &[usize] {
        self.shapes
            .get(key)
            .unwrap_or_else(|| panic!("no fixture {key}"))
    }

    fn f32_matrix(&self, key: &str) -> Array2<f32> {
        let shape = self.shape(key).to_vec();
        assert_eq!(shape.len(), 2, "{key} is not rank-2");
        let bytes = std::fs::read(self.dir.join(format!("{key}.bin"))).expect(key);
        let values: Vec<f32> = bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Array2::from_shape_vec((shape[0], shape[1]), values).expect(key)
    }

    fn i64_matrix_as_u32(&self, key: &str) -> Array2<u32> {
        let shape = self.shape(key).to_vec();
        assert_eq!(shape.len(), 2, "{key} is not rank-2");
        let bytes = std::fs::read(self.dir.join(format!("{key}.bin"))).expect(key);
        let values: Vec<u32> = bytes
            .chunks_exact(std::mem::size_of::<i64>())
            .map(|c| {
                let v = i64::from_le_bytes(c.try_into().unwrap());
                u32::try_from(v).expect("id fits in u32")
            })
            .collect();
        Array2::from_shape_vec((shape[0], shape[1]), values).expect(key)
    }
}

fn max_abs_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn row_cosine(a: &Array2<f32>, b: &Array2<f32>, row: usize) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.row(row).iter().zip(b.row(row).iter()) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[test]
#[ignore]
fn moss_prefill_matches_reference_dump() {
    let model_dir = std::env::var(ENV_MODEL_DIR)
        .unwrap_or_else(|_| panic!("set {ENV_MODEL_DIR} to the checkpoint snapshot"));
    let bin_dir = std::env::var(ENV_BIN_DIR)
        .unwrap_or_else(|_| panic!("set {ENV_BIN_DIR} to the parity-dump bin directory"));
    let fixtures = BinFixtures::open(Path::new(&bin_dir));

    // ── Load: backbone through the normal path, audio side through the
    // speech loader ──
    let weights = load_model_dir(&model_dir).expect("backbone load");
    assert_eq!(weights.arch.family(), "moss_tts_realtime");
    let config_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(&model_dir).join("config.json")).unwrap(),
    )
    .unwrap();
    let aux_config = MossTtsRealtimeConfig::from_config_json(&config_json).unwrap();
    let aux = load_moss_tts_aux_from_safetensors(&model_dir, weights.arch.as_ref(), aux_config)
        .expect("aux load");

    // ── The prefill input: prompt matrix + the 12-token text lead the
    // reference appended (prefill_ids is the exact [T, 17] the reference
    // forwarded, so use it directly) ──
    let prefill_ids = fixtures.i64_matrix_as_u32("prefill_ids");
    let reference_embeds = fixtures.f32_matrix("prefill_embeds");
    let reference_hidden = fixtures.f32_matrix("prefill_hidden");
    let rows = prefill_ids.nrows();
    assert_eq!(reference_embeds.nrows(), rows);
    assert_eq!(reference_hidden.nrows(), rows);

    // ── Gate 1: the 17-way summed embedding ──
    let mut tables: Vec<&Array2<f32>> = Vec::with_capacity(1 + aux.audio_embed_tables.len());
    let embed_owned = weights.embed.to_owned();
    tables.push(&embed_owned);
    tables.extend(aux.audio_embed_tables.iter());
    let embeds = larql_compute::forward::embed::embed_tables_sum(&tables, &prefill_ids);

    let embed_diff = max_abs_diff(&embeds, &reference_embeds);
    println!("gate 1 — summed embeddings: max |Δ| = {embed_diff:e} over {rows} rows");
    assert!(
        embed_diff <= EMBED_MAX_ABS,
        "summed-embedding mismatch: max |Δ| {embed_diff:e} > {EMBED_MAX_ABS:e}"
    );

    // ── Gate 2: backbone forward from those embeddings ──
    let mut engine = StandardEngine::new(None);
    let ffn = WeightFfn { weights: &weights };
    let last_hidden = engine
        .prefill_from_hidden(&weights, &ffn, &embeds)
        .expect("prefill_from_hidden");
    let normed = apply_norm(
        &weights,
        &last_hidden,
        weights.arch.final_norm_key(),
        weights.arch.norm_weight_offset(),
    );

    let reference_last = {
        let mut last = Array2::<f32>::zeros((1, reference_hidden.ncols()));
        last.row_mut(0).assign(&reference_hidden.row(rows - 1));
        last
    };
    let hidden_diff = max_abs_diff(&normed, &reference_last);
    let cosine = row_cosine(&normed, &reference_last, 0);
    println!("gate 2 — final hidden (last row): max |Δ| = {hidden_diff:e}, cosine = {cosine:.8}");
    assert!(
        cosine >= HIDDEN_MIN_COSINE,
        "backbone hidden diverged: cosine {cosine} < {HIDDEN_MIN_COSINE}"
    );
    assert!(
        hidden_diff <= HIDDEN_MAX_ABS,
        "backbone hidden diverged: max |Δ| {hidden_diff:e} > {HIDDEN_MAX_ABS:e}"
    );
}
