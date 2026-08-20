//! MinerU2.5 / MinerU2.5-Pro model-native two-stage document parser.

mod utils;

use clap::Parser;
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{MinerU, MinerUParseOptions, PageParser};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};

#[derive(Parser)]
#[command(name = "mineru")]
#[command(about = "MinerU2.5 / MinerU2.5-Pro two-stage document parser")]
struct Args {
    /// Path to a MinerU2.5 or MinerU2.5-Pro model directory.
    #[arg(short, long)]
    model_dir: PathBuf,
    /// Paths to input images.
    #[arg(required = true)]
    images: Vec<PathBuf>,
    /// Device: cpu, cuda, cuda:N, or metal.
    #[arg(short, long, default_value = "cpu")]
    device: String,
    /// Maximum generated tokens per pass.
    #[arg(long, default_value = "4096")]
    max_tokens: usize,
    /// Number of cropped regions per recognition batch.
    #[arg(long, default_value = "2")]
    region_batch_size: usize,
    /// Minimum edge length for cropped blocks.
    #[arg(long, default_value = "28")]
    min_image_edge: u32,
    /// Maximum crop edge ratio before padding.
    #[arg(long, default_value = "50")]
    max_image_edge_ratio: f32,
    /// Print the raw layout protocol output.
    #[arg(long, default_value_t = false)]
    dump_layout: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();
    if !args.model_dir.exists() {
        return Err(format!("model directory not found: {}", args.model_dir.display()).into());
    }

    let device = parse_device(&args.device)?;
    info!("Using device: {device:?}");
    let load_start = Instant::now();
    let model = MinerU::from_dir(&args.model_dir, device)?;
    info!("Model loaded in {:.2?}", load_start.elapsed());
    let options = MinerUParseOptions {
        max_tokens: args.max_tokens,
        region_batch_size: args.region_batch_size,
        min_image_edge: args.min_image_edge,
        max_image_edge_ratio: args.max_image_edge_ratio,
    };

    for image_path in &args.images {
        let image = match load_image(image_path) {
            Ok(image) => image,
            Err(error) => {
                error!("Failed to load {}: {error}", image_path.display());
                continue;
            }
        };
        let start = Instant::now();
        match model.parse_page(&image, &options) {
            Ok(document) => {
                info!("Inference time: {:.2?}", start.elapsed());
                if args.dump_layout
                    && let Some(raw) = &document.raw_output
                {
                    info!("Layout raw output:\n{raw}");
                }
                for diagnostic in &document.diagnostics {
                    warn!(
                        "{} block {:?}: {}",
                        diagnostic.stage, diagnostic.block_index, diagnostic.message
                    );
                }
                println!("{}", serde_json::to_string_pretty(&document.blocks)?);
            }
            Err(error) => error!("Inference failed for {}: {error}", image_path.display()),
        }
    }
    Ok(())
}
