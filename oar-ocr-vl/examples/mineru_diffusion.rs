//! MinerU-Diffusion single-pass and model-native two-stage parsing example.

use clap::Parser;
use oar_ocr_vl::mineru_diffusion::DEFAULT_PROMPT;
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{
    DiffusionGenerationConfig, MinerUDiffusion, MinerUDiffusionParseOptions, PageParser,
};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};

#[derive(Parser)]
#[command(name = "mineru_diffusion")]
#[command(about = "MinerU-Diffusion block-diffusion document parser")]
struct Args {
    #[arg(short, long)]
    model_dir: PathBuf,
    #[arg(required = true)]
    images: Vec<PathBuf>,
    #[arg(short, long, default_value = "cpu")]
    device: String,
    /// Run flat full-page recognition instead of two-stage parsing.
    #[arg(long, default_value_t = false)]
    single_pass: bool,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, default_value_t = 1024)]
    gen_length: usize,
    #[arg(long, default_value_t = 32)]
    block_length: usize,
    #[arg(long, default_value_t = 32)]
    denoising_steps: usize,
    #[arg(long, default_value_t = 0.95)]
    dynamic_threshold: f32,
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 28)]
    min_image_edge: u32,
    #[arg(long, default_value_t = 50.0)]
    max_image_edge_ratio: f32,
    #[arg(long, default_value_t = false)]
    dump_layout: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    let device = parse_device(&args.device)?;
    info!("Using device: {device:?}");
    let load_start = Instant::now();
    let model = MinerUDiffusion::from_dir(&args.model_dir, device)?;
    info!("Model loaded in {:.2?}", load_start.elapsed());

    let generation = DiffusionGenerationConfig {
        gen_length: args.gen_length,
        block_length: args.block_length,
        denoising_steps: args.denoising_steps,
        dynamic_threshold: args.dynamic_threshold,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    };
    let parse_options = MinerUDiffusionParseOptions {
        generation: generation.clone(),
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
        if args.single_pass {
            let prompt = args.prompt.as_deref().unwrap_or(DEFAULT_PROMPT);
            match model.generate_one(&image, prompt, &generation) {
                Ok(text) => println!("{}", text.trim()),
                Err(error) => error!("Generation failed for {}: {error}", image_path.display()),
            }
            continue;
        }

        match model.parse_page(&image, &parse_options) {
            Ok(document) => {
                info!("Done in {:.2?}", start.elapsed());
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
            Err(error) => error!("Parsing failed for {}: {error}", image_path.display()),
        }
    }
    Ok(())
}
