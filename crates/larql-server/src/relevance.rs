//! Relevance: does a feature fire *unusually* for this query, or for everything?
//!
//! # The problem
//!
//! DESCRIBE ranks by raw gate score, `dot(gate_f, query)`. Some features
//! respond to a direction that nearly every query shares, so they score high
//! for every entity: measured 2026-09-04 on production, `especially`, `either`,
//! `role`, `mode` sit in the top ten for Paris, France and Tokyo alike. They
//! are coherent (one concept each) so the coherence filter keeps them, and
//! they crowd out the entity's own features — Paris→French was #6 behind
//! four of them.
//!
//! # The term
//!
//! For each feature, its gate score is measured against a fixed **panel** of
//! unrelated queries, each built exactly as a DESCRIBE query is. That gives a
//! per-feature mean and standard deviation of "how much does this feature
//! fire for an arbitrary input". Relevance is the z-score of the actual
//! query's score against that background:
//!
//! ```text
//!   relevance_f = (score_f(query) − mean_f) / std_f
//! ```
//!
//! An always-on feature has a high mean and a large score for this query too,
//! so its z is ordinary. A feature specific to this entity scores far above
//! its background, so its z is large. Ranking by z is ranking by surprise.
//!
//! # Which background
//!
//! Two panels are kept, selectable per request (`background=`):
//!
//! * `entities` (default since 2026-09-04): [`ENTITY_PANEL`], a hundred or so
//!   real-world names — places, people, companies, works, foods, sciences —
//!   tokenised and averaged like any DESCRIBE entity. This is the population
//!   a browse is actually drawn from, so "surprising" means "surprising for
//!   an entity".
//! * `vocabulary`: token embeddings spread evenly through the vocabulary. The
//!   first background shipped. Measured 2026-09-04, it let language-specific
//!   function tokens (`kita`, `tôi`, `và`) read as surprising for Einstein:
//!   coherent features that no vocabulary-spread panel row excites, but that
//!   every proper-noun query does a little.
//!
//! # Cost
//!
//! The panel scores for a layer are one decode of that layer's gate matrix
//! plus one matmul `[features × hidden] · [hidden × panel]`, computed on
//! first use and cached for the life of the model. No per-request scan of the
//! whole layer, and nothing at load.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use larql_vindex::ndarray::{Array1, Array2, Axis};
use larql_vindex::PatchedVindex;

/// How many panel queries. Enough for a stable standard deviation, few
/// enough that the matmul is negligible next to the layer decode.
pub const PANEL_SIZE: usize = 64;

/// Which panel a request's relevance is measured against. See the module doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Background {
    Entities,
    Vocabulary,
}

impl Background {
    pub const DEFAULT: Background = Background::Entities;

    pub fn parse(s: &str) -> Option<Background> {
        match s {
            "entities" | "entity" => Some(Background::Entities),
            "vocabulary" | "vocab" => Some(Background::Vocabulary),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Background::Entities => "entities",
            Background::Vocabulary => "vocabulary",
        }
    }
}

/// The entity panel. Chosen for spread across kinds (place, person, company,
/// work, food, animal, science, sport, event, object) and across regions,
/// with a handful in non-Latin scripts so a script direction is part of the
/// background rather than a surprise. Deliberately *without* the entities
/// used to measure DESCRIBE quality (Paris, France, Tokyo, Einstein, Amazon,
/// Beethoven), so a measurement is never against its own background.
pub const ENTITY_PANEL: &[&str] = &[
    // places
    "London",
    "Berlin",
    "Madrid",
    "Rome",
    "Cairo",
    "Mumbai",
    "Beijing",
    "Seoul",
    "Sydney",
    "Toronto",
    "Chicago",
    "Mexico City",
    "São Paulo",
    "Nairobi",
    "Istanbul",
    "Moscow",
    "Stockholm",
    "Lagos",
    "Jakarta",
    "Bangkok",
    "Germany",
    "Spain",
    "Italy",
    "Egypt",
    "India",
    "China",
    "Brazil",
    "Canada",
    "Australia",
    "Nigeria",
    "Turkey",
    "Sweden",
    "Argentina",
    "Vietnam",
    "Poland",
    "the Nile",
    "Mount Everest",
    "the Sahara",
    "the Pacific Ocean",
    "the Alps",
    // people
    "Isaac Newton",
    "Marie Curie",
    "Napoleon",
    "Cleopatra",
    "Shakespeare",
    "Mozart",
    "Picasso",
    "Gandhi",
    "Nelson Mandela",
    "Abraham Lincoln",
    "Charles Darwin",
    "Ada Lovelace",
    "Frida Kahlo",
    "Confucius",
    "Leonardo da Vinci",
    "Galileo",
    // organisations, companies, products
    "Google",
    "Microsoft",
    "Toyota",
    "Samsung",
    "Nike",
    "Coca-Cola",
    "IKEA",
    "NASA",
    "the United Nations",
    "Harvard",
    "Oxford",
    "the Red Cross",
    "Wikipedia",
    "iPhone",
    "Linux",
    "Python",
    // works, culture
    "Hamlet",
    "the Mona Lisa",
    "Star Wars",
    "the Beatles",
    "Harry Potter",
    "the Odyssey",
    "jazz",
    "sushi",
    "pizza",
    "chocolate",
    "coffee",
    "tea",
    // science, nature, sport, ideas
    "photosynthesis",
    "gravity",
    "DNA",
    "electricity",
    "the Moon",
    "Jupiter",
    "oxygen",
    "carbon",
    "elephant",
    "dolphin",
    "eagle",
    "oak tree",
    "rose",
    "football",
    "chess",
    "tennis",
    "the Olympics",
    "democracy",
    "capitalism",
    "Buddhism",
    "the Renaissance",
    "the Industrial Revolution",
    "the Second World War",
    "the Internet",
    // other scripts
    "東京",
    "北京",
    "Москва",
    "القاهرة",
    "서울",
    "मुंबई",
];

/// The DESCRIBE query for `text`: its token embeddings scaled by
/// `embed_scale` and averaged. One place, so the panel and the request are
/// scored on the same footing; a background built one way and a query built
/// another would measure the difference between the two constructions.
///
/// `None` when the text tokenises to nothing.
pub fn entity_query(
    embeddings: &Array2<f32>,
    embed_scale: f32,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    text: &str,
) -> Option<Array1<f32>> {
    let encoding = tokenizer.encode(text, false).ok()?;
    let ids = encoding.get_ids();
    if ids.is_empty() {
        return None;
    }
    let hidden = embeddings.ncols();
    let mut avg = Array1::<f32>::zeros(hidden);
    for &tok in ids {
        let row = embeddings.row(tok as usize);
        avg += &row.mapv(|v| v * embed_scale);
    }
    avg /= ids.len() as f32;
    Some(avg)
}

/// Below this a feature's spread is treated as this floor, so a feature that
/// is nearly constant over the panel cannot turn a tiny excursion into a huge z.
const STD_FLOOR: f32 = 1e-3;

/// Per-feature background for one layer.
#[derive(Debug)]
pub struct LayerStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl LayerStats {
    /// The relevance of `score` for `feature`, or `None` when the feature is
    /// outside this layer's known range (an overlay-added feature, say).
    pub fn z(&self, feature: usize, score: f32) -> Option<f32> {
        let m = *self.mean.get(feature)?;
        let s = *self.std.get(feature)?;
        Some((score - m) / s.max(STD_FLOOR))
    }

    /// Compute from a gate matrix `[features × hidden]` and a panel
    /// `[panel × hidden]`.
    pub fn from_gates(gates: &Array2<f32>, panel: &Array2<f32>) -> LayerStats {
        // [features × panel]
        let scores = gates.dot(&panel.t());
        let n = scores.ncols().max(1) as f32;
        let mean: Vec<f32> = scores
            .mean_axis(Axis(1))
            .map(|m| m.to_vec())
            .unwrap_or_default();
        let std: Vec<f32> = scores
            .axis_iter(Axis(0))
            .zip(mean.iter())
            .map(|(row, m)| (row.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / n).sqrt())
            .collect();
        LayerStats { mean, std }
    }
}

/// Lazily computed per-layer backgrounds for one model, one set per panel.
pub struct RelevanceStats {
    /// `[panel × hidden]` per background, each row already built like a
    /// DESCRIBE query. A background whose panel could not be built (no
    /// tokenizer, no embeddings) has an empty matrix and yields no relevance.
    panels: HashMap<Background, Array2<f32>>,
    per_layer: RwLock<HashMap<(Background, usize), Arc<LayerStats>>>,
}

fn stack(rows: Vec<Array1<f32>>, hidden: usize) -> Array2<f32> {
    let mut panel = Array2::<f32>::zeros((rows.len(), hidden));
    for (i, r) in rows.iter().enumerate() {
        panel.row_mut(i).assign(r);
    }
    panel
}

/// Token ids taken evenly through the vocabulary so the panel spans scripts
/// and frequencies rather than one region of it; rows with no direction
/// (zero or non-finite) are skipped. A background, so junk sub-word tokens
/// are fine members of it.
fn vocabulary_panel(embeddings: &Array2<f32>, embed_scale: f32) -> Array2<f32> {
    let vocab = embeddings.nrows();
    let mut rows: Vec<Array1<f32>> = Vec::with_capacity(PANEL_SIZE);
    if vocab > 0 {
        let step = (vocab / PANEL_SIZE).max(1);
        let mut id = step / 2;
        while id < vocab && rows.len() < PANEL_SIZE {
            let r = embeddings.row(id).mapv(|v| v * embed_scale);
            let norm = r.dot(&r).sqrt();
            if norm.is_finite() && norm > f32::EPSILON {
                rows.push(r);
            }
            id += step;
        }
    }
    stack(rows, embeddings.ncols())
}

/// Every name in `names` that tokenises to a finite, non-zero query.
fn entity_panel(
    embeddings: &Array2<f32>,
    embed_scale: f32,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    names: &[&str],
) -> Array2<f32> {
    let vocab = embeddings.nrows();
    let rows: Vec<Array1<f32>> = names
        .iter()
        .filter_map(|n| {
            let enc = tokenizer.encode(*n, false).ok()?;
            // A token id outside the embedding table means this tokenizer
            // does not belong to these embeddings; skip rather than panic.
            if enc.get_ids().iter().any(|&t| t as usize >= vocab) {
                return None;
            }
            entity_query(embeddings, embed_scale, tokenizer, n)
        })
        .filter(|r| {
            let norm = r.dot(r).sqrt();
            norm.is_finite() && norm > f32::EPSILON
        })
        .collect();
    stack(rows, embeddings.ncols())
}

impl RelevanceStats {
    /// Only the vocabulary panel: for fixtures and callers with no tokenizer.
    /// `Background::Entities` yields no relevance on such a model.
    pub fn from_embeddings(embeddings: &Array2<f32>, embed_scale: f32) -> RelevanceStats {
        let mut panels = HashMap::new();
        panels.insert(
            Background::Vocabulary,
            vocabulary_panel(embeddings, embed_scale),
        );
        RelevanceStats {
            panels,
            per_layer: RwLock::new(HashMap::new()),
        }
    }

    /// Both panels: the entity panel from `names` through `tokenizer`, and
    /// the vocabulary panel from the embedding table.
    pub fn from_entities(
        embeddings: &Array2<f32>,
        embed_scale: f32,
        tokenizer: &larql_vindex::tokenizers::Tokenizer,
        names: &[&str],
    ) -> RelevanceStats {
        let mut r = RelevanceStats::from_embeddings(embeddings, embed_scale);
        r.panels.insert(
            Background::Entities,
            entity_panel(embeddings, embed_scale, tokenizer, names),
        );
        r
    }

    pub fn panel_size(&self, background: Background) -> usize {
        self.panels.get(&background).map_or(0, |p| p.nrows())
    }

    /// The `background` for `layer`, computing it on first use.
    ///
    /// `None` when that panel is empty (a model with no usable embeddings, or
    /// no tokenizer for the entity panel) or the layer has no gates — in
    /// which case there is nothing to rank by and the caller should fall back
    /// to the raw score rather than invent one.
    pub fn layer(
        &self,
        patched: &PatchedVindex,
        background: Background,
        layer: usize,
    ) -> Option<Arc<LayerStats>> {
        let panel = self.panels.get(&background)?;
        if panel.nrows() < 2 {
            return None;
        }
        let key = (background, layer);
        if let Some(s) = self.per_layer.read().ok()?.get(&key) {
            return Some(Arc::clone(s));
        }
        let gates = patched.base_gate_matrix(layer)?;
        let stats = Arc::new(LayerStats::from_gates(&gates, panel));
        // Two requests racing here both compute the same thing; the second
        // insert is harmless and cheaper than a lock held across the decode.
        self.per_layer.write().ok()?.insert(key, Arc::clone(&stats));
        Some(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::ndarray::arr2;

    #[test]
    fn an_always_on_feature_is_not_surprising_and_a_specific_one_is() {
        // Panel of four queries that all share direction e0 with some spread,
        // and vary in e1/e2. Feature A points along e0 — it fires for every
        // query. Feature B points along e2 — it fires only when e2 is present.
        let panel = arr2(&[
            [1.0, 0.2, 0.0],
            [1.0, -0.2, 0.1],
            [0.9, 0.1, -0.1],
            [1.1, 0.0, 0.0],
        ]);
        let gates = arr2(&[
            [1.0, 0.0, 0.0], // A: always on
            [0.0, 0.0, 1.0], // B: specific
        ]);
        let stats = LayerStats::from_gates(&gates, &panel);

        // A query that looks like the panel plus a strong e2 component.
        let q = [1.0f32, 0.0, 1.0];
        let score = |f: usize| gates.row(f).dot(&Array1::from(q.to_vec()));
        let za = stats.z(0, score(0)).unwrap();
        let zb = stats.z(1, score(1)).unwrap();

        // A scores 1.0 against a background mean of 1.0: not surprising.
        assert!(
            za.abs() < 1.0,
            "always-on feature should be ordinary, z={za}"
        );
        // B scores 1.0 against a background of ~0 with tiny spread: surprising.
        assert!(
            zb > 5.0,
            "specific feature should be far above background, z={zb}"
        );
        assert!(zb > za);
    }

    #[test]
    fn a_feature_constant_over_the_panel_uses_the_floor_not_infinity() {
        let panel = arr2(&[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]);
        let gates = arr2(&[[0.0, 1.0]]); // orthogonal to every panel row: std = 0
        let stats = LayerStats::from_gates(&gates, &panel);
        let z = stats.z(0, 0.5).unwrap();
        assert!(z.is_finite(), "z must be finite, got {z}");
        assert!(z > 0.0);
    }

    #[test]
    fn a_feature_outside_the_layer_has_no_relevance() {
        let panel = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let gates = arr2(&[[1.0, 0.0]]);
        let stats = LayerStats::from_gates(&gates, &panel);
        assert!(stats.z(7, 1.0).is_none());
    }

    #[test]
    fn the_panel_skips_rows_with_no_direction() {
        let mut e = Array2::<f32>::zeros((PANEL_SIZE * 2, 3));
        for i in 0..e.nrows() {
            if i % 4 != 0 {
                e[[i, 0]] = 1.0; // every 4th row stays all-zero
            }
        }
        let r = RelevanceStats::from_embeddings(&e, 1.0);
        let n = r.panel_size(Background::Vocabulary);
        assert!(n > 0 && n <= PANEL_SIZE);
        for row in r.panels[&Background::Vocabulary].axis_iter(Axis(0)) {
            assert!(row.dot(&row) > 0.0, "a zero row reached the panel");
        }
        // No tokenizer, so no entity panel — and no invented relevance.
        assert_eq!(r.panel_size(Background::Entities), 0);
    }

    #[test]
    fn the_entity_panel_never_contains_a_measurement_entity() {
        for held_out in [
            "Paris",
            "France",
            "Tokyo",
            "Einstein",
            "Amazon",
            "Beethoven",
        ] {
            assert!(
                !ENTITY_PANEL
                    .iter()
                    .any(|n| n.eq_ignore_ascii_case(held_out)),
                "{held_out} is in the panel it is measured against"
            );
        }
        assert!(
            ENTITY_PANEL.len() >= PANEL_SIZE,
            "panel too small for a stable std"
        );
    }

    #[test]
    fn background_names_round_trip() {
        for b in [Background::Entities, Background::Vocabulary] {
            assert_eq!(Background::parse(b.as_str()), Some(b));
        }
        assert_eq!(Background::parse("corpus"), None);
        assert_eq!(Background::DEFAULT, Background::Entities);
    }
}
