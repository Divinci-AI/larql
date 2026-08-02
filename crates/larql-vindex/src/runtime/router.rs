//! The bound router — scores, selects and weights.
//!
//! The user-facing sketch of a bound operation carried the router as a bare
//! tensor. A tensor alone cannot express the decision, though: top-k depth,
//! whether selected probabilities are renormalised, and whether learned
//! per-expert scales apply are all part of *what routing means* for a given
//! model, and all three change which experts run and how much each
//! contributes.
//!
//! They are bound properties rather than execution-time inference, and they
//! reuse `larql-compute`'s policy vocabulary so that the reference path and
//! the incumbent path are describing the same thing in the same words.

use larql_compute::{MoeExpertScalePolicy, MoeTopKWeightPolicy};

use super::error::ExecutionError;
use super::tensor::BoundTensor;

/// One expert selected for a token, with the weight it contributes at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedExpert {
    pub expert_id: u32,
    /// Weight after every policy has been applied — what the reduction uses.
    pub weight: f32,
    /// Probability before renormalisation and per-expert scaling. Kept because
    /// a routing disagreement shows up here first, while the final weight can
    /// still coincide after renormalisation hides it.
    pub raw_score: f32,
}

/// Scoring and selection for one bank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundRouter<'a> {
    /// `[num_experts, hidden]`.
    pub weight: BoundTensor<'a>,
    pub top_k: usize,
    pub selected_weight: MoeTopKWeightPolicy,
    pub expert_scale: MoeExpertScalePolicy,
    /// Learned per-expert output scale, when the policy uses one.
    pub per_expert_scale: Option<BoundTensor<'a>>,
}

impl BoundRouter<'_> {
    /// Experts this router can address.
    pub fn population(&self) -> usize {
        self.weight.rows()
    }

    /// Input width the router contracts over.
    pub fn hidden_dim(&self) -> usize {
        self.weight.cols()
    }

    /// Check the router's shape against the population it routes into.
    pub fn validate(&self, population: usize, hidden: usize) -> Result<(), ExecutionError> {
        self.weight.require_matrix(population, hidden)?;
        if let Some(scale) = &self.per_expert_scale {
            scale.require_vector(population)?;
        }
        Ok(())
    }

    pub fn describe(&self) -> String {
        format!(
            "router {} — top-{} of {}",
            self.weight.describe(),
            self.top_k,
            self.population()
        )
    }
}
