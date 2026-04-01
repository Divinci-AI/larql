# larql-inference

Inference engine for transformer models. Forward pass, attention, FFN routing, and the WalkFfn sparse inference backend.

## Overview

This crate handles running transformer models — forward passes, attention computation, and FFN evaluation. It uses `larql-vindex` for gate KNN (sparse feature selection) and `larql-models` for weight loading and architecture definitions.

```rust
use larql_inference::InferenceModel;

// Load a model
let model = InferenceModel::load("google/gemma-3-4b-it")?;

// Run inference
let result = larql_inference::predict(
    model.weights(), model.tokenizer(), &token_ids, 5,
);
println!("Top prediction: {} ({:.1}%)", result.predictions[0].0, result.predictions[0].1 * 100.0);
```

## Key Components

| Module | Purpose |
|--------|---------|
| `forward.rs` | Forward pass: `predict()`, `predict_with_ffn()`, `forward_to_layer()` |
| `attention.rs` | Multi-head attention with GQA, RoPE, sliding window |
| `ffn/` | FFN evaluation: dense, sparse, gated, cached |
| `vindex/walk_ffn.rs` | WalkFfn — sparse FFN using vindex gate KNN (8092 features, lossless) |
| `capture.rs` | Residual stream vector capture for probing |
| `model.rs` | Model loading (re-exports from larql-models) |
| `walker/` | Weight-level graph walkers (no forward pass) |

## WalkFfn

The WalkFfn replaces the dense FFN with a sparse version that uses the vindex gate KNN:

1. Gate KNN selects top-8092 features (from gate_vectors.bin)
2. Only selected features go through up/down projections
3. Result is identical to dense FFN (97.91% on France→Paris)
4. Enables interpretable inference — see which features fired

## Crate Dependencies

```
larql-models      ModelWeights, architecture traits, quant
    ↓
larql-vindex      VectorIndex, gate KNN, patches
    ↓
larql-inference   Forward pass, attention, WalkFfn
```

## License

Apache-2.0
