//! Opt-in Metal/CPU parity check for PP-DocLayout (regression guard for #177,
//! where the detector returned garbage boxes on Metal).
//!
//! Needs a checkpoint, so it is skipped unless both env vars are set:
//!
//! ```sh
//! LAYOUT_DIR=models/PP-DocLayoutV3_safetensors IMG=page.png \
//!   cargo test -p oar-ocr-vl --features metal --release --test layout_device_parity -- --nocapture
//! ```
#![cfg(all(feature = "metal", target_os = "macos"))]

use candle_core::Device;

#[test]
fn layout_metal_matches_cpu() {
    let (Ok(dir), Ok(img_path)) = (std::env::var("LAYOUT_DIR"), std::env::var("IMG")) else {
        eprintln!("skipping: set LAYOUT_DIR and IMG to run this parity check");
        return;
    };
    let Ok(metal_device) = Device::new_metal(0) else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let img = image::open(&img_path).unwrap().to_rgb8();

    let cpu = oar_ocr_vl::PpDocLayout::from_dir(&dir, Device::Cpu).unwrap();
    let metal = oar_ocr_vl::PpDocLayout::from_dir(&dir, metal_device).unwrap();
    let expected = cpu.detect(&img).unwrap();
    let actual = metal.detect(&img).unwrap();

    assert_eq!(
        expected.len(),
        actual.len(),
        "detection count differs between CPU and Metal"
    );
    let mut worst_box = 0f32;
    let mut worst_score = 0f32;
    for (i, (want, got)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(want.class_id, got.class_id, "class differs at index {i}");
        for axis in 0..4 {
            worst_box = worst_box.max((want.bbox[axis] - got.bbox[axis]).abs());
        }
        worst_score = worst_score.max((want.score - got.score).abs());
    }
    println!(
        "{} detections; worst |Δbox| = {worst_box:.4} px, worst |Δscore| = {worst_score:.6}",
        expected.len()
    );
    assert!(worst_box < 1.0, "boxes diverge by {worst_box} px");
    assert!(worst_score < 0.01, "scores diverge by {worst_score}");
}
