//! The VINDEX3-only registry/resolver — the `vindex3-registry` initiative.
//!
//! Turns one of four public reference forms into a resolved VINDEX3
//! artifact, structurally incapable of resolving to VINDEX2:
//!
//! ```text
//! qwen3.8
//!      |
//!      v
//! official VINDEX3 registry entry        (manifest.rs)
//!      |
//!      v
//! pinned Hugging Face VINDEX3 artifact    (resolver.rs -> ArtifactRef)
//!      |
//!      v
//! local VINDEX3 container                 (a later rung fetches this)
//! ```
//!
//! Scope for this rung (`docs/vindex3-registry-design.md`, architecture
//! confirmed 2026-08-23): the manifest schema, the name/variant grammar,
//! this resolver, and a tiny static test registry ([`fixtures`]). **Not**
//! wired into `larql run`/`serve`/`pull`, and not consolidated with the
//! three existing resolvers yet (design doc §1/§8) — that is the
//! follow-up "resolver convergence" rung, once this contract is proven.
//! Not a generic model registry, and VINDEX2 has no representation in
//! this schema at all — see [`manifest`] and [`resolver`] for where that
//! is enforced structurally rather than by convention.

mod abi;
mod error;
pub mod fixtures;
mod manifest;
mod reference;
mod resolver;

#[cfg(test)]
mod abi_tests;
#[cfg(test)]
mod manifest_tests;
#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod resolver_tests;

pub use abi::{Vindex3Abi, CURRENT_VINDEX3_ABI};
pub use error::RegistryError;
pub use manifest::{
    Provenance, RegistryArtifactRef, RegistryManifest, RegistryModel, RegistryVariant,
    REGISTRY_MANIFEST_SCHEMA_VERSION,
};
pub use reference::{ExplicitReference, ModelName, ModelReference, VariantName};
pub use resolver::{resolve, ArtifactRef, ResolvedVindex3, Vindex3Resolution};
