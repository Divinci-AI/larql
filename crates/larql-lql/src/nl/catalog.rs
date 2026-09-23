//! The closed set of LQL statements English can route to, and the access
//! class of each.
//!
//! Access is derived from the PARSED statement, never from what the router
//! claimed it picked. A router that says "describe" and somehow yields a
//! `DELETE` is still classified as a write, because the classification reads
//! the AST that would actually execute.

use crate::ast::Statement;

/// A routable statement kind. `Pipe` is deliberately absent: composition is
/// written in LQL, not asked for in English.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Extract,
    Compile,
    Diff,
    Use,
    Walk,
    Infer,
    Select,
    Describe,
    Explain,
    Insert,
    Delete,
    Update,
    Merge,
    Rebalance,
    ShowRelations,
    ShowLayers,
    ShowFeatures,
    ShowEntities,
    ShowModels,
    Stats,
    ShowCompactStatus,
    CompactInto,
    CompactMinor,
    CompactMajor,
    BeginPatch,
    SavePatch,
    ApplyPatch,
    ShowPatches,
    RemovePatch,
    Trace,
}

/// Every routable kind, in the order offered to the router.
pub const ALL: [Kind; 30] = [
    Kind::Extract,
    Kind::Compile,
    Kind::Diff,
    Kind::Use,
    Kind::Walk,
    Kind::Infer,
    Kind::Select,
    Kind::Describe,
    Kind::Explain,
    Kind::Insert,
    Kind::Delete,
    Kind::Update,
    Kind::Merge,
    Kind::Rebalance,
    Kind::ShowRelations,
    Kind::ShowLayers,
    Kind::ShowFeatures,
    Kind::ShowEntities,
    Kind::ShowModels,
    Kind::Stats,
    Kind::ShowCompactStatus,
    Kind::CompactInto,
    Kind::CompactMinor,
    Kind::CompactMajor,
    Kind::BeginPatch,
    Kind::SavePatch,
    Kind::ApplyPatch,
    Kind::ShowPatches,
    Kind::RemovePatch,
    Kind::Trace,
];

/// The option key used for "no LQL statement serves this request".
pub const NONE_KEY: &str = "none";

impl Kind {
    /// Stable option key sent to the router and read back from it.
    pub fn key(self) -> &'static str {
        match self {
            Kind::Extract => "extract",
            Kind::Compile => "compile",
            Kind::Diff => "diff",
            Kind::Use => "use",
            Kind::Walk => "walk",
            Kind::Infer => "infer",
            Kind::Select => "select",
            Kind::Describe => "describe",
            Kind::Explain => "explain",
            Kind::Insert => "insert",
            Kind::Delete => "delete",
            Kind::Update => "update",
            Kind::Merge => "merge",
            Kind::Rebalance => "rebalance",
            Kind::ShowRelations => "show_relations",
            Kind::ShowLayers => "show_layers",
            Kind::ShowFeatures => "show_features",
            Kind::ShowEntities => "show_entities",
            Kind::ShowModels => "show_models",
            Kind::Stats => "stats",
            Kind::ShowCompactStatus => "show_compact_status",
            Kind::CompactInto => "compact_into",
            Kind::CompactMinor => "compact_minor",
            Kind::CompactMajor => "compact_major",
            Kind::BeginPatch => "begin_patch",
            Kind::SavePatch => "save_patch",
            Kind::ApplyPatch => "apply_patch",
            Kind::ShowPatches => "show_patches",
            Kind::RemovePatch => "remove_patch",
            Kind::Trace => "trace",
        }
    }

    pub fn from_key(key: &str) -> Option<Kind> {
        ALL.iter().copied().find(|k| k.key() == key)
    }

    /// What the statement does, in the words the router is shown.
    pub fn description(self) -> &'static str {
        match self {
            Kind::Extract => "Extract a model's weights from a checkpoint or model id into a new vindex on disk.",
            Kind::Compile => "Compile the current edited vindex back into a standalone vindex or model checkpoint file.",
            Kind::Diff => "Compare two vindexes, or a vindex against the current session, and list what differs.",
            Kind::Use => "Open or switch to a vindex or model so later queries run against it.",
            Kind::Walk => "Show which FFN features fire for a prompt, without attention (a feature scan).",
            Kind::Infer => "Run the model on a prompt and show its predicted next tokens.",
            Kind::Select => "Query stored edges as rows, e.g. every entity with a given relation, SQL-style.",
            Kind::Describe => "Describe everything the model knows about one entity: its relations and targets.",
            Kind::Explain => "Explain, layer by layer, how the model arrives at its prediction for a prompt.",
            Kind::Insert => "Add a new fact (entity, relation, target) to the model's knowledge.",
            Kind::Delete => "Remove or erase a fact or entity from the model's knowledge.",
            Kind::Update => "Change an existing fact, e.g. give an entity a different target for a relation.",
            Kind::Merge => "Merge another vindex's knowledge into the current one.",
            Kind::Rebalance => "Rebalance installed facts so each lands in the target probability band.",
            Kind::ShowRelations => "List the relation types the model has learned.",
            Kind::ShowLayers => "List the model's layers and what each band of layers encodes.",
            Kind::ShowFeatures => "List the FFN features stored at one specific layer.",
            Kind::ShowEntities => "List the named entities the model stores knowledge about.",
            Kind::ShowModels => "List the models or vindexes that are available.",
            Kind::Stats => "Show summary statistics about the loaded vindex: size, layers, features.",
            Kind::ShowCompactStatus => "Show whether the vindex needs compaction and its storage status.",
            Kind::CompactInto => "Write a compacted, physically reorganised copy of the vindex to a new path.",
            Kind::CompactMinor => "Run a minor compaction of the vindex in place.",
            Kind::CompactMajor => "Run a major compaction of the vindex in place.",
            Kind::BeginPatch => "Start recording edits into a new named patch file.",
            Kind::SavePatch => "Save the edits made so far into the current patch file.",
            Kind::ApplyPatch => "Apply an existing patch file's edits to the current vindex.",
            Kind::ShowPatches => "List the patches applied to the current vindex.",
            Kind::RemovePatch => "Remove a previously applied patch from the current vindex.",
            Kind::Trace => "Trace the residual stream for a prompt, showing how a token's probability builds across layers.",
        }
    }
}

/// Which routable kind a parsed statement is. `None` for `Pipe`.
///
/// Exhaustive on purpose: adding a variant to `Statement` without deciding
/// how English reaches it fails to compile here.
pub fn kind_of(stmt: &Statement) -> Option<Kind> {
    Some(match stmt {
        Statement::Extract { .. } => Kind::Extract,
        Statement::Compile { .. } => Kind::Compile,
        Statement::Diff { .. } => Kind::Diff,
        Statement::Use { .. } => Kind::Use,
        Statement::Walk { .. } => Kind::Walk,
        Statement::Infer { .. } => Kind::Infer,
        Statement::Select { .. } => Kind::Select,
        Statement::Describe { .. } => Kind::Describe,
        Statement::Explain { .. } => Kind::Explain,
        Statement::Insert { .. } => Kind::Insert,
        Statement::Delete { .. } => Kind::Delete,
        Statement::Update { .. } => Kind::Update,
        Statement::Merge { .. } => Kind::Merge,
        Statement::Rebalance { .. } => Kind::Rebalance,
        Statement::ShowRelations { .. } => Kind::ShowRelations,
        Statement::ShowLayers { .. } => Kind::ShowLayers,
        Statement::ShowFeatures { .. } => Kind::ShowFeatures,
        Statement::ShowEntities { .. } => Kind::ShowEntities,
        Statement::ShowModels => Kind::ShowModels,
        Statement::Stats { .. } => Kind::Stats,
        Statement::ShowCompactStatus => Kind::ShowCompactStatus,
        Statement::CompactInto { .. } => Kind::CompactInto,
        Statement::CompactMinor => Kind::CompactMinor,
        Statement::CompactMajor { .. } => Kind::CompactMajor,
        Statement::BeginPatch { .. } => Kind::BeginPatch,
        Statement::SavePatch => Kind::SavePatch,
        Statement::ApplyPatch { .. } => Kind::ApplyPatch,
        Statement::ShowPatches => Kind::ShowPatches,
        Statement::RemovePatch { .. } => Kind::RemovePatch,
        Statement::Trace { .. } => Kind::Trace,
        Statement::Pipe { .. } => return None,
    })
}

/// What executing a statement can do. Ordered: a pipe is as strong as its
/// strongest side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Access {
    /// Answers a question. A misroute costs a wrong answer and nothing else.
    Read,
    /// Changes which vindex or patch the session points at. Reversible.
    Session,
    /// Changes knowledge, or writes a file to disk. Needs confirmation.
    Write,
}

/// The access class of a parsed statement.
///
/// Exhaustive on purpose, like [`kind_of`]. Two statements are WRITE only in
/// some forms, and that is decided from their fields, not their name:
/// `DIFF … INTO PATCH` writes a patch file, and `TRACE … SAVE` writes a trace.
pub fn access(stmt: &Statement) -> Access {
    match stmt {
        Statement::Walk { .. }
        | Statement::Infer { .. }
        | Statement::Select { .. }
        | Statement::Describe { .. }
        | Statement::Explain { .. }
        | Statement::ShowRelations { .. }
        | Statement::ShowLayers { .. }
        | Statement::ShowFeatures { .. }
        | Statement::ShowEntities { .. }
        | Statement::ShowModels
        | Statement::Stats { .. }
        | Statement::ShowCompactStatus
        | Statement::ShowPatches => Access::Read,

        Statement::Diff { into_patch, .. } => {
            if into_patch.is_some() {
                Access::Write
            } else {
                Access::Read
            }
        }
        Statement::Trace { save, .. } => {
            if save.is_some() {
                Access::Write
            } else {
                Access::Read
            }
        }

        Statement::Use { .. } | Statement::BeginPatch { .. } => Access::Session,

        Statement::Extract { .. }
        | Statement::Compile { .. }
        | Statement::Insert { .. }
        | Statement::Delete { .. }
        | Statement::Update { .. }
        | Statement::Merge { .. }
        | Statement::Rebalance { .. }
        | Statement::CompactInto { .. }
        | Statement::CompactMinor
        | Statement::CompactMajor { .. }
        | Statement::SavePatch
        | Statement::ApplyPatch { .. }
        | Statement::RemovePatch { .. } => Access::Write,

        Statement::Pipe { left, right } => access(left).max(access(right)),
    }
}
