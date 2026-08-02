//! A bound expert bank — the population an operation can route into.
//!
//! The bank owns the expert list and the shapes they share. Selection happens
//! upstream in the router; the bank's job is to hand back the expert a
//! selected id names, and to refuse when the id has no expert behind it.
//!
//! That refusal matters more than it looks. A router and a bank that disagree
//! about the population is a binding fault, and the natural coding of it —
//! skip the expert, carry on — produces a token that is quietly missing a
//! fraction of its FFN contribution and looks entirely reasonable.

use crate::format::capability::coordinate::BankCoordinate;
use larql_compute::Activation;

use super::error::ExecutionError;
use super::projection::BoundProjection;
use super::tensor::BoundTensor;

/// One expert: its gated projection and its down projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundExpert<'a> {
    /// Id within the bank's population, as the router names it.
    pub expert_id: u32,
    pub projection: BoundProjection<'a>,
    /// `[hidden, intermediate]` — contracts the intermediate axis away.
    pub down: BoundTensor<'a>,
}

impl BoundExpert<'_> {
    pub fn validate(&self, intermediate: usize, hidden: usize) -> Result<(), ExecutionError> {
        self.projection.validate(intermediate, hidden)?;
        self.down.require_matrix(hidden, intermediate)
    }
}

/// A population of experts sharing one shape and one activation.
///
/// `PartialEq` only: `Activation` is not `Eq`, which is the correct choice for
/// a type that names float behaviour rather than a discrete tag.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundBankOperation<'a> {
    /// Which bank this is. Diagnostics only — execution never branches on it.
    pub bank: BankCoordinate,
    pub experts: Vec<BoundExpert<'a>>,
    /// Intermediate width of every expert in the bank.
    pub intermediate_dim: usize,
    /// Input/output width — the residual width for a direct bank, the latent
    /// width for a bank sitting behind routed-input/output transforms.
    pub hidden_dim: usize,
    pub activation: Activation,
}

impl<'a> BoundBankOperation<'a> {
    pub fn population(&self) -> usize {
        self.experts.len()
    }

    /// The expert an id names, or a refusal.
    ///
    /// Deliberately not `Option`: a selected-but-absent expert is never a
    /// condition to handle locally, it is a binding fault that must reach the
    /// caller intact.
    pub fn expert(&self, expert_id: u32) -> Result<&BoundExpert<'a>, ExecutionError> {
        self.experts
            .iter()
            .find(|e| e.expert_id == expert_id)
            .ok_or(ExecutionError::ExpertOutOfRange {
                expert: expert_id,
                population: self.population(),
            })
    }

    /// Check every expert against the bank's declared shape.
    pub fn validate(&self) -> Result<(), ExecutionError> {
        for expert in &self.experts {
            expert.validate(self.intermediate_dim, self.hidden_dim)?;
        }
        Ok(())
    }

    pub fn describe(&self) -> String {
        format!(
            "{} — {} experts, {}×{}",
            self.bank.describe(),
            self.population(),
            self.intermediate_dim,
            self.hidden_dim
        )
    }
}
