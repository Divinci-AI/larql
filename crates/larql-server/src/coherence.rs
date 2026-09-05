//! Query-time feature coherence, and a label chosen from the whole cluster.
//!
//! # The problem this exists for
//!
//! A feature's display label is `top_token`, which is built as pure argmax of
//! the logit lens (`extract/streaming/stages/down_meta.rs`: `top_k[0]`). Argmax
//! over an unembedding favours rare, long-tail tokens, so a feature whose top-k
//! is `[capital, capitals, 首都, ...]` can be labelled with whichever rare token
//! happened to edge out `capital`. Measured against production on 2026-09-01
//! over four entities and 80 associations: 33% of labels were not usable words,
//! and 26% had a better candidate already inside the same response.
//!
//! Two separate things are wrong, and they need opposite treatment:
//!
//!   * a COHERENT feature with a bad label — `[서울, capital, capitals, 首都]`
//!     is unambiguously "capital city" and only the label is wrong. Relabel it.
//!   * an INCOHERENT feature — `[Gates, RICS, latino, ستم]` means nothing.
//!     Relabelling makes it *worse*: picking the first latin-looking token turns
//!     `그러` into `uckoo`. These should be suppressed, not renamed.
//!
//! Nothing in the artifact distinguishes them today. `c_score` looks like it
//! would, but it is assigned `top_k[0].logit` at build time — the top logit
//! again, carrying no independent information.
//!
//! # What this computes
//!
//! Coherence is the mean pairwise cosine similarity of the feature's top-k token
//! embeddings. A feature whose top-k all point the same way in embedding space
//! is about one thing; a feature whose top-k are mutually unrelated is not.
//!
//! For unit vectors this has a closed form. With `c` the mean of `n` unit
//! vectors, the mean of the `n(n-1)` ordered off-diagonal cosines is
//!
//! ```text
//!   (n * ||c||^2 - 1) / (n - 1)
//! ```
//!
//! so the whole thing is O(n·d) rather than the O(n²·d) of summing every pair.
//! The same centroid then picks the label: the top-k token whose embedding is
//! nearest the centroid is the cluster's most representative member, which is a
//! different question from which token has the largest logit.
//!
//! Everything here reads data the artifact already carries — `top_k` stores 10
//! entries with token ids, and the embedding matrix is already resident — so
//! this needs no rebuild of the vindex.

use larql_vindex::ndarray::Array2;

/// The writing system a token is mostly made of.
///
/// Used only to break ties in label choice. A multilingual feature such as
/// `[일본, Jepang, 在日本, Japan]` has one meaning and four spellings; the
/// centroid-nearest token is the most *central* spelling, which for a cluster
/// that straddles scripts is routinely not the one the caller can read. The
/// caller asked in some script; a label in that script is the useful one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    Latin,
    Cyrillic,
    Greek,
    Arabic,
    Hebrew,
    Devanagari,
    Thai,
    Hangul,
    /// Han, Hiragana, Katakana — grouped, since features do not separate them.
    Cjk,
    /// Digits, punctuation, symbols, or nothing classifiable.
    Other,
}

fn script_of_char(c: char) -> Script {
    let u = c as u32;
    match u {
        0x0041..=0x024F | 0x1E00..=0x1EFF => Script::Latin,
        0x0370..=0x03FF => Script::Greek,
        0x0400..=0x052F => Script::Cyrillic,
        0x0590..=0x05FF => Script::Hebrew,
        0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => Script::Arabic,
        0x0900..=0x097F => Script::Devanagari,
        0x0E00..=0x0E7F => Script::Thai,
        0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7AF => Script::Hangul,
        0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => Script::Cjk,
        _ => Script::Other,
    }
}

/// The dominant script of a token, by letter count. Tokens with no letters
/// classify as `Other`, which never matches a preference — a label made of
/// punctuation is never the useful one.
pub fn script_of(token: &str) -> Script {
    let mut counts = [0usize; 10];
    let idx = |s: Script| match s {
        Script::Latin => 0,
        Script::Cyrillic => 1,
        Script::Greek => 2,
        Script::Arabic => 3,
        Script::Hebrew => 4,
        Script::Devanagari => 5,
        Script::Thai => 6,
        Script::Hangul => 7,
        Script::Cjk => 8,
        Script::Other => 9,
    };
    for c in token.chars() {
        if c.is_alphabetic() {
            counts[idx(script_of_char(c))] += 1;
        }
    }
    let order = [
        Script::Latin,
        Script::Cyrillic,
        Script::Greek,
        Script::Arabic,
        Script::Hebrew,
        Script::Devanagari,
        Script::Thai,
        Script::Hangul,
        Script::Cjk,
    ];
    let mut best = Script::Other;
    let mut best_n = 0;
    for s in order {
        let n = counts[idx(s)];
        if n > best_n {
            best_n = n;
            best = s;
        }
    }
    best
}

/// A token considered as a label candidate.
#[derive(Debug, Clone, Copy)]
pub struct Candidate<'a> {
    pub token: &'a str,
    pub token_id: u32,
}

/// What the coherence pass concluded about one feature.
#[derive(Debug, Clone)]
pub struct FeatureVerdict {
    /// Mean pairwise cosine over the top-k embeddings, in [-1, 1].
    pub coherence: f32,
    /// The top-k token nearest the centroid, when one could be chosen.
    pub label: Option<String>,
    /// How many candidates actually had an embedding row.
    pub support: usize,
}

/// L2-normalize a row, returning `None` for a zero (or non-finite) vector.
///
/// A zero row is not a direction, and normalizing it would produce NaNs that
/// silently poison the centroid and every cosine taken against it.
fn unit_row(embeddings: &Array2<f32>, token_id: u32) -> Option<Vec<f32>> {
    let idx = token_id as usize;
    if idx >= embeddings.nrows() {
        return None;
    }
    let row = embeddings.row(idx);
    let norm = row.dot(&row).sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return None;
    }
    Some(row.iter().map(|v| v / norm).collect())
}

/// Score a feature from its top-k tokens.
///
/// Returns `None` when fewer than two candidates have embeddings: coherence is a
/// statement about how a set of directions agree, and a single direction always
/// agrees with itself. Reporting 1.0 there would mark the least-supported
/// features as the most trustworthy — exactly backwards.
pub fn score_feature(
    embeddings: &Array2<f32>,
    candidates: &[Candidate<'_>],
    prefer: Option<Script>,
) -> Option<FeatureVerdict> {
    let dim = embeddings.ncols();
    let mut units: Vec<(usize, Vec<f32>)> = Vec::with_capacity(candidates.len());
    for (i, c) in candidates.iter().enumerate() {
        if let Some(u) = unit_row(embeddings, c.token_id) {
            units.push((i, u));
        }
    }
    let n = units.len();
    if n < 2 {
        return None;
    }

    // Centroid of the unit vectors.
    let mut centroid = vec![0.0f32; dim];
    for (_, u) in &units {
        for (c, v) in centroid.iter_mut().zip(u.iter()) {
            *c += *v;
        }
    }
    let inv = 1.0 / n as f32;
    for c in centroid.iter_mut() {
        *c *= inv;
    }

    // Mean pairwise cosine, via ||centroid||^2 rather than every pair.
    let sq = centroid.iter().map(|v| v * v).sum::<f32>();
    let coherence = ((n as f32 * sq) - 1.0) / (n as f32 - 1.0);
    let coherence = coherence.clamp(-1.0, 1.0);

    // Label: the candidate nearest the centroid, restricted to the preferred
    // script when any candidate is written in it.
    //
    // Measured on production 2026-09-01: Tokyo's top feature is unambiguously
    // "Japan" — top-k `[일본, Jepang, 在日本, Japan]`, coherence 0.61 — and the
    // centroid-nearest token was 일본, which a caller who typed "Tokyo" cannot
    // read. The cluster is right; the spelling chosen from it was not. The
    // preference is a filter over the candidates, not a different score, so a
    // feature's coherence is unchanged by it.
    //
    // Ties resolve to the earlier candidate, which keeps the choice
    // deterministic across requests — a label that flickered between equally
    // central tokens would make the same edit look like two different ones.
    let cnorm = sq.sqrt();
    let label = if cnorm <= f32::EPSILON {
        None
    } else {
        let in_script = |i: usize| prefer.is_none_or(|p| script_of(candidates[i].token) == p);
        let any_in_script = units.iter().any(|(i, _)| in_script(*i));
        units
            .iter()
            .filter(|(i, _)| !any_in_script || in_script(*i))
            .map(|(i, u)| {
                let dot: f32 = u.iter().zip(centroid.iter()).map(|(a, b)| a * b).sum();
                (*i, dot / cnorm)
            })
            .fold(None::<(usize, f32)>, |best, (i, s)| match best {
                Some((_, bs)) if bs >= s => best,
                _ => Some((i, s)),
            })
            .map(|(i, _)| candidates[i].token.to_string())
    };

    Some(FeatureVerdict {
        coherence,
        label,
        support: n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::ndarray::arr2;

    /// Rows: 0 and 1 near-identical, 2 orthogonal to them, 3 opposite to 0.
    fn embeddings() -> Array2<f32> {
        arr2(&[
            [1.0, 0.0, 0.0],   // 0
            [0.99, 0.14, 0.0], // 1  ~8 degrees from 0
            [0.0, 0.0, 1.0],   // 2  orthogonal
            [-1.0, 0.0, 0.0],  // 3  opposite
            [0.0, 0.0, 0.0],   // 4  zero row — no direction
        ])
    }

    /// A cluster with an unambiguous centre: row 0 lies on the centroid, rows 1
    /// and 2 sit symmetrically 37° either side of it. Every row is a unit
    /// vector, so dot products are cosines and the ordering is exact, not a
    /// floating-point coincidence.
    fn multilingual() -> Array2<f32> {
        arr2(&[
            [1.0, 0.0, 0.0],  // 0 — the centre
            [0.8, 0.6, 0.0],  // 1
            [0.8, -0.6, 0.0], // 2
        ])
    }

    fn cands<'a>(ids: &[(u32, &'a str)]) -> Vec<Candidate<'a>> {
        ids.iter()
            .map(|(id, t)| Candidate {
                token: t,
                token_id: *id,
            })
            .collect()
    }

    #[test]
    fn agreeing_tokens_score_high_and_disagreeing_ones_low() {
        let e = embeddings();
        let tight = score_feature(&e, &cands(&[(0, "a"), (1, "b")]), None).unwrap();
        let split = score_feature(&e, &cands(&[(0, "a"), (2, "c")]), None).unwrap();
        let opposed = score_feature(&e, &cands(&[(0, "a"), (3, "d")]), None).unwrap();

        assert!(tight.coherence > 0.95, "tight was {}", tight.coherence);
        assert!(
            split.coherence.abs() < 0.01,
            "orthogonal was {}",
            split.coherence
        );
        assert!(
            opposed.coherence < -0.95,
            "opposed was {}",
            opposed.coherence
        );

        // The ordering is the whole point: it must separate the populations.
        assert!(tight.coherence > split.coherence);
        assert!(split.coherence > opposed.coherence);
    }

    #[test]
    fn the_label_is_the_centroid_nearest_token_not_the_first() {
        // Two tokens agree and one is an outlier. The label must come from the
        // agreeing pair even when the outlier is listed first — that is exactly
        // the `[서울, capital, capitals]` case, where the argmax token is the
        // one that does not belong.
        let e = embeddings();
        let v = score_feature(
            &e,
            &cands(&[(2, "outlier"), (0, "core"), (1, "core2")]),
            None,
        )
        .unwrap();
        assert_ne!(v.label.as_deref(), Some("outlier"), "picked the outlier");
        assert!(matches!(v.label.as_deref(), Some("core") | Some("core2")));
    }

    #[test]
    fn a_single_usable_candidate_is_not_coherent_by_default() {
        // One direction trivially agrees with itself. Scoring that 1.0 would
        // rank the least-supported features as the most trustworthy.
        let e = embeddings();
        assert!(score_feature(&e, &cands(&[(0, "only")]), None).is_none());
        assert!(score_feature(&e, &[], None).is_none());
    }

    #[test]
    fn unusable_rows_are_skipped_rather_than_poisoning_the_score() {
        let e = embeddings();
        // id 4 is a zero row and id 99 is out of range; neither is a direction.
        let v = score_feature(
            &e,
            &cands(&[(0, "a"), (1, "b"), (4, "zero"), (99, "oob")]),
            None,
        )
        .unwrap();
        assert_eq!(v.support, 2, "only two rows were usable");
        assert!(v.coherence.is_finite(), "coherence went non-finite");
        assert!(v.coherence > 0.95);
    }

    #[test]
    fn scoring_is_deterministic() {
        // Two identical calls must agree, or the same edit would report two
        // different labels on consecutive reads.
        let e = embeddings();
        let c = cands(&[(0, "a"), (1, "b"), (2, "c")]);
        let x = score_feature(&e, &c, None).unwrap();
        let y = score_feature(&e, &c, None).unwrap();
        assert_eq!(x.label, y.label);
        assert_eq!(x.coherence.to_bits(), y.coherence.to_bits());
    }

    #[test]
    fn script_detection_reads_the_letters_and_ignores_the_rest() {
        assert_eq!(script_of("Japan"), Script::Latin);
        assert_eq!(script_of("Jepang"), Script::Latin);
        assert_eq!(script_of("일본"), Script::Hangul);
        assert_eq!(script_of("在日本"), Script::Cjk);
        assert_eq!(script_of("दिल्ली"), Script::Devanagari);
        assert_eq!(script_of("для"), Script::Cyrillic);
        // Mixed: the letters decide, not the punctuation.
        assert_eq!(script_of("Japan's"), Script::Latin);
        // No letters at all is not a script and never matches a preference.
        assert_eq!(script_of("(){"), Script::Other);
        assert_eq!(script_of("42"), Script::Other);
    }

    #[test]
    fn the_preferred_script_wins_even_when_it_is_not_the_most_central() {
        // The production case: the most central spelling is Hangul, a Latin
        // spelling exists, the caller asked in Latin. Row 0 is the centre of
        // this cluster; the Latin token is off-centre on row 1.
        let e = multilingual();
        let c = cands(&[(0, "일본"), (1, "Japan"), (2, "在日本")]);

        let neutral = score_feature(&e, &c, None).unwrap();
        assert_eq!(
            neutral.label.as_deref(),
            Some("일본"),
            "row 0 is the most central"
        );

        let latin = score_feature(&e, &c, Some(Script::Latin)).unwrap();
        assert_eq!(latin.label.as_deref(), Some("Japan"));

        // The preference is a filter over the label, not a change to the score.
        assert_eq!(neutral.coherence.to_bits(), latin.coherence.to_bits());
    }

    #[test]
    fn with_no_candidate_in_the_preferred_script_the_centroid_still_decides() {
        // Asking for Latin when the cluster has none must not produce "no
        // label": the most central spelling is still the best available one.
        let e = multilingual();
        let c = cands(&[(0, "일본"), (1, "日本"), (2, "在日本")]);
        let v = score_feature(&e, &c, Some(Script::Latin)).unwrap();
        assert_eq!(v.label.as_deref(), Some("일본"));
    }

    #[test]
    fn punctuation_never_satisfies_a_preference() {
        // `(){` is in no script. With Other never matching, the preference falls
        // through to the centroid rather than labelling a feature with braces.
        let e = multilingual();
        let c = cands(&[(0, "(){"), (1, "일본"), (2, "日本")]);
        let v = score_feature(&e, &c, Some(Script::Latin)).unwrap();
        assert_eq!(
            v.label.as_deref(),
            Some("(){"),
            "row 0 is most central; no Latin present"
        );
        // And when a real Latin token exists, braces lose to it.
        let c2 = cands(&[(0, "(){"), (1, "Japan"), (2, "日本")]);
        let v2 = score_feature(&e, &c2, Some(Script::Latin)).unwrap();
        assert_eq!(v2.label.as_deref(), Some("Japan"));
    }
}
