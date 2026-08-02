//! Shared builders for runtime tests.
//!
//! `BoundTensor` borrows its bytes, so a helper that returns one has to keep
//! the buffer alive. These leak it: the buffers are a few dozen bytes, the
//! count is bounded by the test suite, and the alternative — threading an
//! owner struct through every test — obscures what each test is about.

use crate::format::capability::binding::RepresentationIdentity;
use crate::format::capability::component::ComponentContract;
use crate::format::lyrw2::region_format::RegionFormat;

use super::tensor::BoundTensor;

/// Variant these test operands claim to come from.
pub const TEST_VARIANT: &str = "test";

fn bind(region_set: &str, values: &[f32], contract: ComponentContract) -> BoundTensor<'static> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    BoundTensor::direct(
        RepresentationIdentity::new(region_set, TEST_VARIANT),
        leaked,
        RegionFormat::F32,
        contract,
    )
    .expect("test operands are well-formed")
}

/// An f32 matrix in row-major order.
pub fn matrix(region_set: &str, rows: u32, cols: u32, values: &[f32]) -> BoundTensor<'static> {
    assert_eq!(
        values.len(),
        (rows * cols) as usize,
        "{region_set}: {rows}×{cols} needs {} values",
        rows * cols
    );
    bind(region_set, values, ComponentContract::matrix(rows, cols))
}

/// An f32 vector.
pub fn vector(region_set: &str, values: &[f32]) -> BoundTensor<'static> {
    bind(
        region_set,
        values,
        ComponentContract::vector(values.len() as u32),
    )
}

/// A matrix whose values ascend from 1.0, for tests that only need distinct
/// numbers in a known order.
pub fn ascending(region_set: &str, rows: u32, cols: u32) -> BoundTensor<'static> {
    let values: Vec<f32> = (0..rows * cols).map(|i| (i + 1) as f32).collect();
    matrix(region_set, rows, cols, &values)
}
