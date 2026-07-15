//! Minimal OCR pipeline using the `auto-download` feature.
//!
//! Demonstrates how the high-level builder transparently fetches missing model
//! files from ModelScope. Any model path that isn't an on-disk file but
//! matches an entry in [`oar_ocr::download::REGISTRY`] is downloaded into
//! `$OAR_HOME` (default `~/.oar`) and verified against its SHA-256.
//!
//! # Build
//!
//! ```bash
//! cargo run --features auto-download --example auto_download -- <image.jpg>
//! ```
//!
//! Without the `auto-download` feature this example refuses to compile so the
//! intent is explicit.

#[cfg(not(feature = "auto-download"))]
fn main() {
    eprintln!(
        "This example requires the `auto-download` feature.\n\
         Re-run with: cargo run --features auto-download --example auto_download -- <image.jpg>"
    );
    std::process::exit(2);
}

#[cfg(feature = "auto-download")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use oar_ocr::oarocr::OAROCRBuilder;
    use oar_ocr::utils::load_image;
    use std::path::PathBuf;
    use tracing_subscriber::EnvFilter;

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let image: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: auto_download <image>")?;

    println!("OAR cache: {}", oar_ocr::download::cache_dir().display());

    // Bare file names are resolved through the registry on `build()`.
    // The first run downloads to ~/.oar (or $OAR_HOME); subsequent runs reuse
    // the cached copies after verifying their SHA-256.
    let ocr = OAROCRBuilder::new(
        "pp-ocrv5_mobile_det.onnx",
        "pp-ocrv5_mobile_rec.onnx",
        "ppocrv5_dict.txt",
    )
    .with_text_line_orientation_classification("pp-lcnet_x1_0_textline_ori.onnx")
    .build()?;

    let img = load_image(&image)?;
    let results = ocr.predict(vec![img])?;
    for (i, page) in results.iter().enumerate() {
        println!("--- page {i} ---");
        for region in &page.text_regions {
            if let Some(ref text) = region.text {
                println!("{text}");
            }
        }
    }
    Ok(())
}
