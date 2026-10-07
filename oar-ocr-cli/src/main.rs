mod args;
mod pdf;

use anyhow::{Context, Result, bail, ensure};
use args::{Cli, Command, Common, OcrFormat, PageFormat};
use clap::Parser;
use image::RgbImage;
use oar_ocr::{
    core::config::{OrtExecutionProvider, OrtSessionConfig},
    oarocr::{OAROCRBuilder, OARStructureBuilder},
};
use oar_ocr_vl::{
    AnyPageParser, AnyPageParserLoadOptions, AnyPageParserModel, AnyPageParserOptions, PageParser,
};
use pdf::Input;
use serde_json::Value;
use std::{
    collections::BTreeSet,
    fs,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

const PARSERS: &[AnyPageParserModel] = &[
    AnyPageParserModel::HpdParsing,
    AnyPageParserModel::HunyuanOcr,
    AnyPageParserModel::JinaOcr,
    AnyPageParserModel::MinerU2509,
    AnyPageParserModel::MinerUPro,
    AnyPageParserModel::MinerUDiffusion,
    AnyPageParserModel::MonkeyOcrV2S,
    AnyPageParserModel::MonkeyOcrV2B,
    AnyPageParserModel::OvisOcr2,
    AnyPageParserModel::WeVisDoc2B,
    AnyPageParserModel::WeVisDoc4B,
    AnyPageParserModel::XiaomiOcr,
    AnyPageParserModel::PaddleOcrVl,
    AnyPageParserModel::PaddleOcrVl1_5,
    AnyPageParserModel::PaddleOcrVl1_6,
    AnyPageParserModel::GlmOcr,
    AnyPageParserModel::TeleOcr,
];

// Match the largest default image batch while bounding rendered page memory.
const PAGE_BATCH_SIZE: usize = 8;

struct Document {
    text: String,
    json: Value,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_max_level(if cli.verbose {
            tracing::Level::INFO
        } else {
            tracing::Level::WARN
        })
        .with_writer(io::stderr)
        .without_time()
        .init();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    if matches!(&cli.command, Command::Parse(args) if args.list_models) {
        for model in PARSERS {
            println!("{model}");
        }
        return Ok(());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()?;
    let (paths, json) = match &cli.command {
        Command::Ocr(args) => (&args.images, args.format == OcrFormat::Json),
        Command::Structure(args) => (&args.images, args.format == PageFormat::Json),
        Command::Parse(args) => (&args.images, args.format == PageFormat::Json),
    };
    let inputs = paths
        .iter()
        .map(|path| {
            Input::open(path, cli.common.pages.as_ref())
                .with_context(|| format!("could not open {}", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let destinations = output_paths(&inputs, &cli.common, json)?;
    let total_pages: usize = inputs.iter().map(|input| input.pages.len()).sum();
    match &cli.command {
        Command::Ocr(args) => {
            let builder = match (&args.models.det, &args.models.rec, &args.models.dict) {
                (Some(det), Some(rec), Some(dict)) => OAROCRBuilder::new(det, rec, dict),
                (None, None, None) => OAROCRBuilder::pp_ocrv6(args.size.into()),
                _ => bail!("custom OCR models require --det, --rec, and --dict together"),
            };
            let model = builder
                .ort_session(classic_device(&cli.common.device, cli.verbose)?)
                .build().context("could not load OCR models; check the network or provide local --det, --rec, and --dict files")?;
            tracing::info!("OCR models loaded; recognizing {total_pages} page(s)");
            process_inputs(
                &inputs,
                &cli.common,
                destinations,
                json,
                PAGE_BATCH_SIZE,
                |images, pages, first_index| {
                    model
                        .predict(images)?
                        .into_iter()
                        .zip(pages)
                        .enumerate()
                        .map(|(offset, (mut page, (input, _)))| {
                            page.input_path = input.path.to_string_lossy().into_owned().into();
                            page.index = first_index + offset;
                            Ok(Document {
                                text: page.concatenated_text("\n"),
                                json: serde_json::to_value(page)?,
                            })
                        })
                        .collect()
                },
            )
        }
        Command::Structure(args) => {
            let builder = OARStructureBuilder::pp_structurev3();
            let builder = match (&args.models.det, &args.models.rec, &args.models.dict) {
                (Some(det), Some(rec), Some(dict)) => builder.with_ocr(det, rec, dict),
                (None, None, None) => builder,
                _ => bail!("custom OCR models require --det, --rec, and --dict together"),
            };
            let model = builder
                .ort_session(classic_device(&cli.common.device, cli.verbose)?)
                .build().context("could not load structure models; check model-download connectivity and try again")?;
            tracing::info!("Structure models loaded; parsing {total_pages} page(s)");
            process_inputs(
                &inputs,
                &cli.common,
                destinations,
                json,
                PAGE_BATCH_SIZE,
                |images, pages, first_index| {
                    let dimensions = images.iter().map(RgbImage::dimensions).collect::<Vec<_>>();
                    model
                        .predict_images(images)
                        .into_iter()
                        .zip(pages)
                        .enumerate()
                        .map(|(offset, (page, (input, _)))| {
                            let mut page = page?;
                            page.input_path = input.path.to_string_lossy().into_owned().into();
                            page.index = first_index + offset;
                            let (width, height) = dimensions[offset];
                            Ok(Document {
                                text: page.to_markdown(),
                                json: page.to_json(width, height),
                            })
                        })
                        .collect()
                },
            )
        }
        Command::Parse(args) => {
            let parser = load_parser(args, &cli.common.device)?;
            tracing::info!("Page parser loaded; parsing {total_pages} page(s)");
            let mut options = AnyPageParserOptions::default();
            if let Some(tokens) = args.max_tokens {
                options = options.with_max_new_tokens(tokens);
            }
            process_inputs(
                &inputs,
                &cli.common,
                destinations,
                json,
                1,
                |images, pages, _| {
                    images
                        .into_iter()
                        .zip(pages)
                        .map(|(image, (input, _))| {
                            let path = &input.path;
                            let page = parser
                                .parse_page(&image, &options)
                                .with_context(|| format!("could not parse {}", path.display()))?;
                            for diagnostic in &page.diagnostics {
                                tracing::warn!("{}: {}", path.display(), diagnostic.message);
                            }
                            let text = page.markdown.clone().unwrap_or_else(|| {
                                let blocks = page
                                    .blocks
                                    .iter()
                                    .filter_map(|b| b.content.as_deref())
                                    .collect::<Vec<_>>()
                                    .join("\n\n");
                                if blocks.is_empty() {
                                    page.raw_output.clone().unwrap_or_default()
                                } else {
                                    blocks
                                }
                            });
                            Ok(Document {
                                text,
                                json: page.to_json(image.width(), image.height()),
                            })
                        })
                        .collect()
                },
            )
        }
    }
}

fn classic_device(device: &str, verbose: bool) -> Result<OrtSessionConfig> {
    let config = match device {
        "auto" => OrtSessionConfig::auto().resolve_auto(),
        "cpu" => OrtSessionConfig::new().with_execution_providers(vec![OrtExecutionProvider::CPU]),
        "metal" => {
            ensure!(
                cfg!(all(
                    target_os = "macos",
                    any(feature = "metal", feature = "coreml")
                )),
                "Metal/CoreML requires macOS and a build with --features metal; use --device cpu instead"
            );
            OrtSessionConfig::new().with_execution_providers(vec![
                OrtExecutionProvider::CoreML {
                    ane_only: None,
                    subgraphs: None,
                },
                OrtExecutionProvider::CPU,
            ])
        }
        cuda if cuda.starts_with("cuda:") => {
            ensure!(
                cfg!(feature = "cuda"),
                "CUDA support is missing; reinstall with `cargo install oar-ocr-cli --features cuda --force`, or use --device cpu"
            );
            OrtSessionConfig::new().with_execution_providers(vec![
                OrtExecutionProvider::CUDA {
                    device_id: Some(cuda[5..].parse()?),
                    gpu_mem_limit: None,
                    arena_extend_strategy: None,
                    cudnn_conv_algo_search: None,
                    cudnn_conv_use_max_workspace: None,
                },
                OrtExecutionProvider::CPU,
            ])
        }
        _ => bail!("unsupported device; use auto, cpu, cuda:N, or metal"),
    };
    let config = config.with_intra_threads(4);
    #[cfg(any(feature = "cuda", feature = "metal", feature = "coreml"))]
    if device != "auto" && device != "cpu" {
        probe_classic_device(device)
            .context("could not initialize the requested accelerator; verify its driver/runtime installation or use --device cpu")?;
    }
    Ok(if verbose {
        config.with_log_severity_level(1)
    } else {
        config
    })
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "coreml"))]
fn probe_classic_device(device: &str) -> Result<()> {
    oar_ocr::core::inference::initialize_ort_environment()?;
    let provider = match device {
        #[cfg(feature = "cuda")]
        cuda if cuda.starts_with("cuda:") => ort::ep::CUDA::default()
            .with_device_id(cuda[5..].parse()?)
            .build()
            .error_on_failure(),
        #[cfg(any(feature = "metal", feature = "coreml"))]
        "metal" => ort::ep::CoreML::default().build().error_on_failure(),
        _ => bail!("device requires its corresponding accelerator feature"),
    };
    let builder = ort::session::Session::builder()?
        .with_intra_threads(1)
        .map_err(ort::Error::<()>::from)?;
    builder
        .with_execution_providers([provider])
        .map_err(ort::Error::<()>::from)?;
    Ok(())
}

fn load_parser(args: &args::Parse, device: &str) -> Result<AnyPageParser> {
    if device.starts_with("cuda:") {
        ensure!(
            cfg!(feature = "cuda"),
            "CUDA support is missing; reinstall with `cargo install oar-ocr-cli --features cuda --force`, or use --device cpu"
        );
    }
    if device == "metal" {
        ensure!(
            cfg!(all(target_os = "macos", feature = "metal")),
            "Metal requires macOS and a build with --features metal; use --device cpu instead"
        );
    }
    let model = args
        .model
        .context("specify --model, or use --list-models")?;
    if let Some(dir) = &args.layout_dir {
        ensure!(
            dir.is_dir(),
            "layout directory {} does not exist; point --layout-dir at a PP-DocLayout checkpoint",
            dir.display()
        );
    }
    if let Some(dir) = &args.model_dir {
        ensure!(
            dir.is_dir(),
            "model directory {} does not exist; provide a checkpoint directory or omit --model-dir to download it",
            dir.display()
        );
        let device = oar_ocr_vl::utils::parse_device(device)
            .context("could not create the requested device; try --device cpu")?;
        let mut options = AnyPageParserLoadOptions::default();
        if let Some(layout) = &args.layout_dir {
            options = options.with_layout_dir(layout);
        }
        return AnyPageParser::from_dir_with_options(model, dir, device, &options)
            .context("could not load the local parser; check that the checkpoint matches --model, and provide --layout-dir for PaddleOCR-VL, GLM-OCR, or TeleOCR");
    }
    #[cfg(feature = "auto-download")]
    {
        use oar_ocr_vl::{AnyPageParserPretrainedOptions, DownloadSource};
        let source = match args.source {
            args::Source::Modelscope => DownloadSource::ModelScope,
            args::Source::Huggingface => DownloadSource::HuggingFace,
        };
        let mut options = AnyPageParserPretrainedOptions::default().with_source(source);
        if let Some(layout) = &args.layout_dir {
            options = options.with_layout_dir(layout);
        }
        let device = oar_ocr_vl::utils::parse_device(device)
            .context("could not create the requested device; try --device cpu")?;
        AnyPageParser::from_pretrained(model, device, &options)
            .context("could not download or load the parser; check connectivity, try --source huggingface, or provide --model-dir and --layout-dir for offline loading")
    }
    #[cfg(not(feature = "auto-download"))]
    bail!(
        "model downloads are disabled; reinstall with --features auto-download or provide --model-dir and, when needed, --layout-dir"
    )
}

fn output_paths(inputs: &[Input], common: &Common, json: bool) -> Result<Option<Vec<PathBuf>>> {
    let Some(directory) = &common.output else {
        return Ok(None);
    };
    let outputs = inputs
        .iter()
        .map(|input| {
            let name = input.path.file_name().context("input has no filename")?;
            let stem = input.path.file_stem().context("input has no stem")?;
            let extension = if json { "json" } else { "md" };
            Ok(input
                .pages
                .iter()
                .map(|page| {
                    if input.is_pdf() {
                        let mut name = stem.to_os_string();
                        name.push(format!("_p{page}.{extension}"));
                        directory.join(name)
                    } else {
                        directory.join(name).with_extension(extension)
                    }
                })
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    // Compare case-insensitively so names that only differ in case are caught
    // before inference on case-insensitive filesystems too.
    ensure!(
        outputs
            .iter()
            .map(|path| path.to_string_lossy().to_lowercase())
            .collect::<BTreeSet<_>>()
            .len()
            == outputs.len(),
        "input filenames collide; use unique image stems or run them separately"
    );
    for path in &outputs {
        ensure!(
            !path.exists(),
            "output {} already exists; choose a fresh --output directory",
            path.display()
        );
    }
    fs::create_dir_all(directory)
        .with_context(|| format!("could not create output directory {}", directory.display()))?;
    Ok(Some(outputs))
}

fn process_inputs(
    inputs: &[Input],
    common: &Common,
    destinations: Option<Vec<PathBuf>>,
    json: bool,
    batch_size: usize,
    mut process: impl FnMut(Vec<RgbImage>, &[(&Input, usize)], usize) -> Result<Vec<Document>>,
) -> Result<()> {
    let mut out = io::stdout().lock();
    let array = inputs.iter().any(Input::is_pdf)
        || inputs.iter().map(|input| input.pages.len()).sum::<usize>() != 1;
    if destinations.is_none() && json && array {
        writeln!(out, "[")?;
    }
    let mut index = 0;
    let mut pages = inputs
        .iter()
        .flat_map(|input| input.pages.iter().map(move |number| (input, *number)));
    loop {
        let chunk = pages.by_ref().take(batch_size).collect::<Vec<_>>();
        if chunk.is_empty() {
            break;
        }
        let images = chunk
            .iter()
            .map(|(input, number)| {
                input.render(*number, common.dpi).with_context(|| {
                    format!("could not render {} page {number}", input.path.display())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let documents = process(images, &chunk, index).with_context(|| {
            let (input, number) = chunk[0];
            format!(
                "could not process batch starting at {} page {number}",
                input.path.display()
            )
        })?;
        ensure!(
            documents.len() == chunk.len(),
            "pipeline returned an unexpected number of pages"
        );
        for ((input, number), mut document) in chunk.into_iter().zip(documents) {
            if let Some(page) = document.json.get_mut("page") {
                page["index"] = (number - 1).into();
                document.json["source"] = input.path.to_string_lossy().into_owned().into();
            } else if input.is_pdf() {
                document.json["page_number"] = number.into();
                document.json["input_path"] = input.path.to_string_lossy().into_owned().into();
            }
            if let Some(paths) = &destinations {
                let path = &paths[index];
                let text = if json {
                    serde_json::to_string_pretty(&document.json)?
                } else {
                    document.text
                };
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(path)
                    .with_context(|| {
                        format!(
                            "could not create {}; choose a fresh output directory",
                            path.display()
                        )
                    })?;
                file.write_all(text.as_bytes())?;
            } else if json {
                if array && index > 0 {
                    writeln!(out, ",")?;
                }
                serde_json::to_writer_pretty(&mut out, &document.json)?;
            } else {
                if input.is_pdf() {
                    writeln!(out, "\n<!-- {}: page {number} -->\n", input.path.display())?;
                }
                writeln!(out, "{}", document.text)?;
            }
            index += 1;
        }
    }
    if destinations.is_none() && json {
        if array {
            write!(out, "\n]")?;
        }
        writeln!(out)?;
    }
    Ok(())
}
