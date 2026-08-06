//! The key range a query position is allowed to attend over.
//!
//! Causal masking says a position may not see the future. A *sliding-window*
//! layer additionally may not see the distant past: it attends to the last
//! `window` positions inclusive of itself, and nothing before them.
//!
//! **This existed nowhere in the dense f32 path.** `gqa.rs` sliced
//! `k[0..qi + 1]` and the word "window" did not appear in `block.rs`, so every
//! hybrid model ran full attention on layers the reference windows — 12 of
//! GPT-OSS-20B's 24. It is invisible below the window, because there a sliding
//! layer and a full layer compute the identical thing, which is why an
//! 85-token Gate-B diff passed while half the model's attention was wrong
//! (`docs/k3-funnel.md` §4.9.1).
//!
//! One type, used by prefill and decode alike, so the two cannot drift apart.

use std::ops::Range;

/// How far back a query position may attend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionSpan {
    /// Causal only — every position attends to the whole prefix.
    #[default]
    Full,
    /// Causal *and* bounded: a position attends to at most `window` keys,
    /// ending at itself.
    Sliding {
        /// Total keys visible, including the query's own position.
        window: usize,
    },
}

impl AttentionSpan {
    /// A window of zero or one would leave a position attending to nothing or
    /// to itself alone; neither is a configuration any checkpoint expresses,
    /// and both would make the softmax degenerate. Treat them as unwindowed
    /// rather than silently producing an empty score range.
    const MIN_MEANINGFUL_WINDOW: usize = 2;

    /// Build from an architecture's per-layer decision.
    ///
    /// `window` is `sliding_window_size()`, `sliding` is
    /// `is_sliding_window_layer(layer)`. Both must agree before a window
    /// applies: a layer marked sliding on a model that declares no window
    /// size has no window to apply, and a declared window on a full-attention
    /// layer must not be used.
    pub fn for_layer(sliding: bool, window: Option<usize>) -> Self {
        match (sliding, window) {
            (true, Some(w)) if w >= Self::MIN_MEANINGFUL_WINDOW => Self::Sliding { window: w },
            _ => Self::Full,
        }
    }

    /// The key range position `qi` attends over, given `qi` keys precede it.
    ///
    /// Always ends at `qi + 1` (causal, inclusive of self). Only the start
    /// moves.
    pub fn range(self, qi: usize) -> Range<usize> {
        let end = qi + 1;
        let start = match self {
            Self::Full => 0,
            Self::Sliding { window } => end.saturating_sub(window),
        };
        start..end
    }

    /// The key range a *decode* step attends over, given a cache of
    /// `total_len` keys with the new token last.
    ///
    /// Decode is `range` at `qi = total_len - 1`, written separately because
    /// the decode kernels hold `total_len` rather than a query index and the
    /// off-by-one between them is exactly the kind of thing that makes
    /// prefill and decode disagree.
    pub fn decode_range(self, total_len: usize) -> Range<usize> {
        if total_len == 0 {
            return 0..0;
        }
        self.range(total_len - 1)
    }

    /// Whether this span ever excludes a key. Lets a caller skip windowing
    /// work entirely on the common path.
    pub fn is_full(self) -> bool {
        matches!(self, Self::Full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GPT-OSS-20B's window.
    const GPT_OSS_WINDOW: usize = 128;

    #[test]
    fn full_span_attends_to_the_whole_prefix() {
        assert_eq!(AttentionSpan::Full.range(0), 0..1);
        assert_eq!(AttentionSpan::Full.range(500), 0..501);
    }

    /// The defining property: a windowed position drops the distant past but
    /// still ends at itself.
    #[test]
    fn sliding_span_drops_the_distant_past() {
        let s = AttentionSpan::Sliding {
            window: GPT_OSS_WINDOW,
        };
        assert_eq!(s.range(500), 373..501);
        assert_eq!(s.range(500).len(), GPT_OSS_WINDOW);
    }

    /// **The reason the 85-token fixture passed.** Below the window a sliding
    /// span and a full span are the same function, so a test shorter than the
    /// window cannot distinguish them. Pinned so nobody re-derives a "passing"
    /// short fixture and believes it.
    #[test]
    fn below_the_window_sliding_is_indistinguishable_from_full() {
        let s = AttentionSpan::Sliding {
            window: GPT_OSS_WINDOW,
        };
        for qi in 0..GPT_OSS_WINDOW {
            assert_eq!(
                s.range(qi),
                AttentionSpan::Full.range(qi),
                "qi={qi} must be identical below the window"
            );
        }
        // And the first position that is *not* identical is exactly the window.
        assert_ne!(
            s.range(GPT_OSS_WINDOW),
            AttentionSpan::Full.range(GPT_OSS_WINDOW)
        );
    }

    #[test]
    fn span_never_exceeds_the_window() {
        let s = AttentionSpan::Sliding { window: 4 };
        for qi in 0..50 {
            assert!(s.range(qi).len() <= 4, "qi={qi} exceeded the window");
            assert_eq!(s.range(qi).end, qi + 1, "qi={qi} must stay causal");
        }
    }

    #[test]
    fn range_is_never_empty() {
        for s in [
            AttentionSpan::Full,
            AttentionSpan::Sliding { window: 2 },
            AttentionSpan::Sliding {
                window: GPT_OSS_WINDOW,
            },
        ] {
            for qi in 0..300 {
                assert!(!s.range(qi).is_empty(), "{s:?} qi={qi} attends to nothing");
            }
        }
    }

    #[test]
    fn for_layer_needs_both_the_flag_and_the_size() {
        assert_eq!(
            AttentionSpan::for_layer(true, Some(GPT_OSS_WINDOW)),
            AttentionSpan::Sliding {
                window: GPT_OSS_WINDOW
            }
        );
        // A full-attention layer on a model that declares a window.
        assert_eq!(
            AttentionSpan::for_layer(false, Some(GPT_OSS_WINDOW)),
            AttentionSpan::Full
        );
        // A layer marked sliding on a model that declares no size.
        assert_eq!(AttentionSpan::for_layer(true, None), AttentionSpan::Full);
    }

    /// A degenerate window must not produce a position that sees only itself
    /// (or nothing) — that is a config error, not an attention pattern.
    #[test]
    fn degenerate_windows_fall_back_to_full() {
        assert_eq!(AttentionSpan::for_layer(true, Some(0)), AttentionSpan::Full);
        assert_eq!(AttentionSpan::for_layer(true, Some(1)), AttentionSpan::Full);
    }

    /// Decode must agree with prefill at the same absolute position, or the
    /// two paths diverge exactly where a KV cache makes it hardest to see.
    #[test]
    fn decode_range_matches_prefill_at_the_same_position() {
        for s in [
            AttentionSpan::Full,
            AttentionSpan::Sliding {
                window: GPT_OSS_WINDOW,
            },
        ] {
            for total_len in 1..300 {
                assert_eq!(
                    s.decode_range(total_len),
                    s.range(total_len - 1),
                    "{s:?} disagrees at total_len={total_len}"
                );
            }
        }
    }

    #[test]
    fn decode_range_on_an_empty_cache_is_empty() {
        assert_eq!(AttentionSpan::Full.decode_range(0), 0..0);
    }

    #[test]
    fn is_full_distinguishes_the_variants() {
        assert!(AttentionSpan::Full.is_full());
        assert!(!AttentionSpan::Sliding { window: 8 }.is_full());
        assert!(AttentionSpan::default().is_full());
    }
}
