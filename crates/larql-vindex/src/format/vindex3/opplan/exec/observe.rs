//! Observation points on the canonical decode step (LQL-2 TRACE).
//!
//! There is exactly one semantic execution path; TRACE and any other
//! observer **subscribe to it** — nothing re-enacts the plan to emit
//! events. [`DecodeSession::step_observed`] fires these events at the
//! step's existing operation boundaries and computes exactly what
//! [`step`] computes: the parity gate demands the observed and
//! unobserved paths stay bit-identical, so an observer can never
//! change arithmetic or execution order.
//!
//! Deliberately coarse at this rung: layer and sublayer boundaries and
//! the head's logits — structure, not tensors. Finer taps (operand
//! reads, attention state, residual values) are later detail levels
//! and must arrive the same way: more events on the one executor,
//! never a second traversal.
//!
//! [`DecodeSession::step_observed`]: super::decode::DecodeSession::step_observed
//! [`step`]: super::decode::DecodeSession::step

/// One decode step's observation events, in execution order.
#[derive(Debug, Clone, PartialEq)]
pub enum StepEvent {
    /// The token was embedded at this absolute position.
    Embedded { position: usize },
    /// A layer's attention sublayer completed (residual add included).
    AttentionDone { layer: usize },
    /// A layer's FFN sublayer completed (residual add and any layer
    /// scale included) — the layer boundary.
    FfnDone { layer: usize },
    /// The output head priced the vocabulary for this position.
    Logits { vocab: usize },
}

/// A subscriber to the canonical step's observation points.
pub trait StepObserver {
    fn event(&mut self, event: StepEvent);
}

/// The default subscriber: observes nothing. [`DecodeSession::step`]
/// is `step_observed` with this observer, so the unobserved path is
/// the observed path by construction.
///
/// [`DecodeSession::step`]: super::decode::DecodeSession::step
pub struct NoopObserver;

impl StepObserver for NoopObserver {
    fn event(&mut self, _event: StepEvent) {}
}

/// Convenience subscriber: records every event, for tests and for
/// consumers that render after the step completes.
#[derive(Default)]
pub struct RecordingObserver {
    pub events: Vec<StepEvent>,
}

impl StepObserver for RecordingObserver {
    fn event(&mut self, event: StepEvent) {
        self.events.push(event);
    }
}
