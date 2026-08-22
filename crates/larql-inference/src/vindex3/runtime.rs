//! Opening a VINDEX3 container as an inference runtime.

use std::path::Path;

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::kv::KvState;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prefill_plan;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ClosureDefect, ComponentOpPlan};

use crate::error::InferenceError;

use super::session::Vindex3Session;

/// Inspect the container, plan `component`'s operations, and open the
/// operand store — solely from the container's own contents. Kept
/// outside the generic impl so the whole opening path (and its
/// refusals) is one instantiation regardless of backend.
fn open_component(
    container: &Path,
    component: &str,
) -> Result<(ComponentOpPlan, OperandStore), InferenceError> {
    let inspection = inspect_container(container, false)?;
    let outcome = plan_component_ops(&inspection, container, component)?;
    if !outcome.closed() {
        return Err(unclosed_component(component, &outcome.defects));
    }
    let plan = outcome.plan.ok_or_else(|| {
        InferenceError::Parse(format!("component `{component}` produced no plan"))
    })?;
    let store = OperandStore::open(container, &inspection)?;
    Ok((plan, store))
}

/// Built outside the generic impl so the refusal exists (and is
/// counted) once, not once per backend instantiation. Defensive:
/// unreachable while every encoded text component carries a head,
/// kept so a headless component fails closed instead of panicking.
pub(super) fn headless_prefill_error() -> InferenceError {
    InferenceError::Parse(
        "prefill produced no logits — the component carries no output head".to_string(),
    )
}

/// A component whose stack does not fully classify into the declared
/// operations refuses to open, with the defects in the error. An
/// unclosed program must not be "best-effort" executed.
fn unclosed_component(component: &str, defects: &[ClosureDefect]) -> InferenceError {
    let listed: Vec<String> = defects.iter().map(|d| d.to_string()).collect();
    InferenceError::Parse(format!(
        "component `{component}` does not close: {}",
        listed.join("; ")
    ))
}

/// One opened container component: the executable plan, its operand
/// store, and the arithmetic backend. Owns what a [`Vindex3Session`]
/// borrows, so sessions can be created, dropped, and re-created (fresh
/// conversations) without re-planning the container.
pub struct Vindex3Runtime<B: PlanBackend> {
    plan: ComponentOpPlan,
    store: OperandStore,
    backend: B,
}

impl<B: PlanBackend> Vindex3Runtime<B> {
    /// Open `component` from the container, refusing any closure
    /// defect (see [`unclosed_component`]'s doc).
    pub fn open(container: &Path, component: &str, backend: B) -> Result<Self, InferenceError> {
        let (plan, store) = open_component(container, component)?;
        Ok(Self {
            plan,
            store,
            backend,
        })
    }

    /// Open an incremental session at position zero. Each call loads
    /// the operands in the backend's declared weight format.
    pub fn session(&self) -> Result<Vindex3Session<'_, B>, InferenceError> {
        Vindex3Session::new(&self.plan, &self.store, &self.backend)
    }

    /// Open a session whose continuation state lives in — and outlives
    /// the session as — the caller's [`KvState`] provider (VI3-INF-2).
    /// The session continues from `kv.position()`, so this is also the
    /// resume path after [`prefill_into`](Self::prefill_into). See
    /// [`Vindex3Session::with_kv_state`] for the provider contract.
    pub fn session_with_kv<'a>(
        &'a self,
        kv: &'a mut dyn KvState,
    ) -> Result<Vindex3Session<'a, B>, InferenceError> {
        Vindex3Session::with_kv_state(&self.plan, &self.store, &self.backend, kv)
    }

    /// Batch-prefill `tokens` into the caller's provider (VI3-INF-3)
    /// and return the last position's logits, so generation can sample
    /// the first continuation token before resuming decode over the
    /// **same** provider via [`session_with_kv`](Self::session_with_kv).
    /// A provider already holding state is extended from its logical
    /// position — a long prompt can prefill in chunks.
    pub fn prefill_into(
        &self,
        tokens: &[u32],
        kv: &mut dyn KvState,
    ) -> Result<Vec<f32>, InferenceError> {
        let out = prefill_plan(&self.plan, &self.store, tokens, &self.backend, kv)?;
        out.logits.ok_or_else(headless_prefill_error)
    }

    /// The component's executable plan — the model-meaning authority.
    pub fn plan(&self) -> &ComponentOpPlan {
        &self.plan
    }

    /// The arithmetic backend this runtime executes with.
    pub fn backend(&self) -> &B {
        &self.backend
    }
}
