//! What one compiled overlay costs, measured rather than assumed.
//!
//! `#[ignore]`d because it prints a measurement instead of asserting a
//! threshold: RSS on a shared laptop is too noisy to gate a suite on, and a
//! number baked into an assertion would be wrong on the next machine.
//!
//!     cargo test -p larql-server --test mem_probe -- --ignored --nocapture
//!
//! Run it before changing DEFAULT_OVERLAY_CACHE_ENTRIES. That constant was 8 by
//! judgement until this probe was written; the two numbers it prints are what
//! moved it to 4.
//!
//! The structural number is the important one. If HEAP-RESIDENT gate layers is
//! 0, gate data is mmap-backed and an overlay shares the artifact by refcount —
//! so overlays are cheap to hold. If it is ever non-zero, every overlay
//! deep-copies that much gate data and the cache must shrink accordingly.
//!
//! The RSS delta measures a COLD overlay. The warm caches
//! (`f16_decode_cache`, `warmed_gates`) refill per layer touched and dominate in
//! production; see the arithmetic on DEFAULT_OVERLAY_CACHE_ENTRIES.
use std::path::Path;
fn rss_kb() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .unwrap_or(0)
}

#[test]
#[ignore]
fn measure_overlay_footprint() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-vindex");
    let opts = larql_server::bootstrap::load::LoadVindexOptions {
        no_infer: true,
        ..Default::default()
    };
    let m =
        larql_server::bootstrap::load::load_single_vindex(root.to_str().unwrap(), opts).unwrap();

    let base = m.patched.blocking_read();
    let heap_layers = base
        .base
        .gate
        .gate_vectors
        .iter()
        .filter(|g| g.is_some())
        .count();
    let total_layers = base.base.gate.gate_vectors.len();
    println!("HEAP-RESIDENT gate layers: {heap_layers}/{total_layers}");
    println!("  (0 => gate data is mmap-backed and shared by refcount on clone;");
    println!("   >0 => every overlay deep-copies that much gate data)");

    let before = rss_kb();
    let mut held = Vec::new();
    for _ in 0..8 {
        held.push(larql_vindex::PatchedVindex::new(base.base().clone()));
    }
    let after = rss_kb();
    println!("RSS: {before} KB -> {after} KB for 8 overlays");
    println!("per-overlay: {} KB", (after.saturating_sub(before)) / 8);
    drop(held);
}
