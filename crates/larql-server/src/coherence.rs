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

    // Label: the candidate nearest the centroid. Ties resolve to the earlier
    // candidate, which keeps the choice deterministic across requests — a label
    // that flickered between equally-central tokens would make the same edit
    // look like two different ones.
    let cnorm = sq.sqrt();
    let label = if cnorm <= f32::EPSILON {
        None
    } else {
        units
            .iter()
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
        let tight = score_feature(&e, &cands(&[(0, "a"), (1, "b")])).unwrap();
        let split = score_feature(&e, &cands(&[(0, "a"), (2, "c")])).unwrap();
        let opposed = score_feature(&e, &cands(&[(0, "a"), (3, "d")])).unwrap();

        assert!(tight.coherence > 0.95, "tight was {}", tight.coherence);
        assert!(split.coherence.abs() < 0.01, "orthogonal was {}", split.coherence);
        assert!(opposed.coherence < -0.95, "opposed was {}", opposed.coherence);

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
        let v = score_feature(&e, &cands(&[(2, "outlier"), (0, "core"), (1, "core2")])).unwrap();
        assert_ne!(v.label.as_deref(), Some("outlier"), "picked the outlier");
        assert!(matches!(v.label.as_deref(), Some("core") | Some("core2")));
    }

    #[test]
    fn a_single_usable_candidate_is_not_coherent_by_default() {
        // One direction trivially agrees with itself. Scoring that 1.0 would
        // rank the least-supported features as the most trustworthy.
        let e = embeddings();
        assert!(score_feature(&e, &cands(&[(0, "only")])).is_none());
        assert!(score_feature(&e, &[]).is_none());
    }

    #[test]
    fn unusable_rows_are_skipped_rather_than_poisoning_the_score() {
        let e = embeddings();
        // id 4 is a zero row and id 99 is out of range; neither is a direction.
        let v = score_feature(&e, &cands(&[(0, "a"), (1, "b"), (4, "zero"), (99, "oob")])).unwrap();
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
        let x = score_feature(&e, &c).unwrap();
        let y = score_feature(&e, &c).unwrap();
        assert_eq!(x.label, y.label);
        assert_eq!(x.coherence.to_bits(), y.coherence.to_bits());
    }
}
