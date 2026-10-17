//! Youtu-Parsing-Omni native image-to-JSON document parsing.
//!
//! ```bash
//! cargo run --release -p oar-ocr-vl --example youtu_parsing_omni -- \
//!     --model-dir tencent/Youtu-Parsing-Omni --device cpu document.jpg
//! ```

use clap::Parser;
use oar_ocr_vl::utils::{image::load_image, parse_device};
use oar_ocr_vl::youtu_parsing_omni::DEFAULT_MAX_NEW_TOKENS;
use oar_ocr_vl::{PageParser, YoutuParsingOmni, YoutuParsingOmniParseOptions};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Youtu-Parsing-Omni image/document parsing to native JSON")]
struct Args {
    /// Checkpoint directory.
    #[arg(long)]
    model_dir: PathBuf,
    /// Input page images.
    #[arg(required = true)]
    images: Vec<PathBuf>,
    /// Device: auto, cpu, cuda:N, or metal.
    #[arg(long, default_value = "auto")]
    device: String,
    /// Maximum generated tokens per page.
    #[arg(long, default_value_t = DEFAULT_MAX_NEW_TOKENS)]
    max_tokens: usize,
    /// Override the checkpoint's padded image pixel ceiling.
    #[arg(long)]
    max_pixels: Option<usize>,
    /// Override the official document instruction.
    #[arg(long)]
    prompt: Option<String>,
    /// Emit Markdown instead of native OmniSchema JSON.
    #[arg(long)]
    markdown: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model = YoutuParsingOmni::from_dir(&args.model_dir, parse_device(&args.device)?)?;
    let mut options = YoutuParsingOmniParseOptions::default().with_max_new_tokens(args.max_tokens);
    if let Some(pixels) = args.max_pixels {
        options = options.with_max_pixels(pixels);
    }
    if let Some(prompt) = args.prompt {
        options = options.with_prompt(prompt);
    }
    for path in args.images {
        let image = load_image(&path)?;
        let page = model.parse_page(&image, &options)?;
        for diagnostic in &page.diagnostics {
            eprintln!("{}: {}", path.display(), diagnostic.message);
        }
        let output = if args.markdown {
            page.markdown
        } else {
            page.raw_output
        };
        println!("{}", output.unwrap_or_default());
    }
    Ok(())
}
