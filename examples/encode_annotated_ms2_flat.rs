//! Embed every spectrum in the Zenodo `annotated-ms2-top-128-peaks` dataset
//! and write a TSV with metadata columns (SMILES, taxonomy, retention time,
//! etc.) followed by the latent dimensions.
//!
//! This example demonstrates wiring [`spectral_autoencoder::SpectrumEmbedder`]
//! against a mascot-rs dataset loader. The embedder handles vectorisation,
//! batching, and forward passes. The example only does:
//!
//! 1. Resolve the Zenodo dataset via `AnnotatedMs2Builder`.
//! 2. Open the streaming MGF iterator.
//! 3. Drive the iterator in chunks of `--batch-size`, extracting metadata
//!    on the way in and pairing each emitted [`EmbeddingRow`] with its
//!    source record's annotations on the way out.
//!
//! The pairing is 1:1 because the embedder is built with the default
//! `skip_errors = false`. Any failure aborts the whole run.
//!
//! ```bash
//! cargo run --release --example encode_annotated_ms2_flat \
//!     --no-default-features --features cuda,embed,embed-mgf -- \
//!     --checkpoint runs/gems-a10-top128-flat-cache15-bs2048-80epoch \
//!     --output runs/annotated-ms2-top128-flat/embeddings.tsv
//! ```

#[cfg(all(feature = "cuda", feature = "embed", feature = "embed-mgf"))]
mod app {
    use std::{
        fs,
        io::{BufWriter, Write},
        path::PathBuf,
    };

    use clap::Parser;
    use indicatif::{ProgressBar, ProgressStyle};
    use mascot_rs::prelude::{
        ANNOTATED_MS2_TOP_128_SPECTRA_COUNT, AnnotatedMs2Builder, Dataset, MGFVec,
        MascotGenericFormat,
    };
    use mass_spectrometry::prelude::{Spectrum, SpectrumFloat};
    use spectral_autoencoder::{DEFAULT_EMBED_BATCH_SIZE, SpectrumEmbedder};

    type Backend = burn::backend::Cuda<f32, i32>;

    const DEFAULT_DATASET_DIR: &str = "datasets/annotated-ms2-top-128-peaks";

    const OUTPUT_COLUMNS: &[&str] = &[
        "index",
        "feature_id",
        "scans",
        "spectrum_id",
        "name",
        "smiles",
        "inchikey",
        "formula",
        "npc_pathway",
        "npc_superclass",
        "npc_class",
        "classyfire_kingdom",
        "classyfire_superclass",
        "classyfire_class",
        "classyfire_subclass",
        "classyfire_direct_parent",
        "source",
        "retention_time",
        "precursor_mz",
        "charge",
        "ion_mode",
        "instrument",
    ];

    const SPECTRUM_ID_KEY: &str = "SPECTRUMID";
    const NAME_KEY: &str = "NAME";
    const INCHIKEY_KEY: &str = "INCHIKEY";
    const SOURCE_KEY: &str = "SOURCE_DATASET";
    const NPC_PATHWAY_KEY: &str = "NPC_PATHWAYS";
    const NPC_SUPERCLASS_KEY: &str = "NPC_SUPERCLASSES";
    const NPC_CLASS_KEY: &str = "NPC_CLASSES";
    const CLASSYFIRE_KINGDOM_KEY: &str = "CHEMONT_KINGDOM";
    const CLASSYFIRE_SUPERCLASS_KEY: &str = "CHEMONT_SUPERCLASS";
    const CLASSYFIRE_CLASS_KEY: &str = "CHEMONT_CLASS";
    const CLASSYFIRE_SUBCLASS_KEY: &str = "CHEMONT_SUBCLASS";
    const CLASSYFIRE_DIRECT_PARENT_KEY: &str = "CHEMONT_DIRECT_PARENT";

    #[derive(Debug, Parser)]
    #[command(
        name = "encode_annotated_ms2_flat",
        about = "Embed the Zenodo annotated-ms2-top-128-peaks dataset"
    )]
    struct Args {
        /// Training run directory containing `model-config.json` and `model.mpk`.
        #[arg(long)]
        checkpoint: PathBuf,
        /// Output TSV path.
        #[arg(long)]
        output: PathBuf,
        /// Local dataset cache directory.
        #[arg(long, default_value = DEFAULT_DATASET_DIR)]
        dataset_dir: PathBuf,
        /// Spectra processed per forward pass.
        #[arg(long, default_value_t = DEFAULT_EMBED_BATCH_SIZE)]
        batch_size: usize,
        /// CUDA device ordinal.
        #[arg(long, default_value_t = 0)]
        cuda_device: usize,
        /// Stop after this many spectra (useful for smoke tests).
        #[arg(long)]
        limit: Option<usize>,
        /// Force-redownload the dataset.
        #[arg(long, default_value_t = false)]
        force_download: bool,
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let args = Args::parse();
        if let Some(parent) = args.output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir_all(&args.dataset_dir)?;

        let runtime = tokio::runtime::Runtime::new()?;
        let builder = MGFVec::<f32>::annotated_ms2_top_128_peaks()
            .target_directory(&args.dataset_dir)
            .force_download(args.force_download)
            .verbose();
        let dataset_path = builder.path();
        let mut records =
            runtime.block_on(<AnnotatedMs2Builder<f32> as Dataset>::mgf_iter(builder))?;

        println!("annotated MS2 MGF: {}", dataset_path.display());
        println!("checkpoint: {}", args.checkpoint.display());
        println!("output: {}", args.output.display());
        println!("device: cuda:{}", args.cuda_device);
        println!("batch size: {}", args.batch_size);
        if let Some(limit) = args.limit {
            println!("limit: {limit}");
        }

        let device = burn::backend::cuda::CudaDevice::new(args.cuda_device);
        let mut embedder = SpectrumEmbedder::<Backend>::builder(args.checkpoint.clone(), device)
            .with_batch_size(args.batch_size)
            .build()?;
        let latent_width = embedder.latent_width();

        let mut writer = BufWriter::new(fs::File::create(&args.output)?);
        write_header(&mut writer, latent_width)?;

        let total_expected = args.limit.unwrap_or(ANNOTATED_MS2_TOP_128_SPECTRA_COUNT);
        let bar = progress_bar(total_expected as u64);

        let mut spectra: Vec<MascotGenericFormat<f32>> = Vec::with_capacity(args.batch_size);
        let mut metadata: Vec<Vec<String>> = Vec::with_capacity(args.batch_size);
        let mut emitted = 0usize;

        while let Some(record) = records.by_ref().next() {
            let record = record?;
            metadata.push(annotation_fields(emitted + spectra.len(), &record));
            spectra.push(record);

            let reached_limit = args.limit.is_some_and(|cap| emitted + spectra.len() >= cap);
            if spectra.len() >= args.batch_size || reached_limit {
                let rows = embedder.embed(&spectra)?;
                debug_assert_eq!(
                    rows.len(),
                    metadata.len(),
                    "skip_errors=false should be 1:1"
                );
                for (fields, row) in metadata.drain(..).zip(rows) {
                    write_row(&mut writer, &fields, &row.latent)?;
                }
                emitted += spectra.len();
                spectra.clear();
                bar.set_position(emitted as u64);
                if reached_limit {
                    break;
                }
            }
        }

        if !spectra.is_empty() {
            let rows = embedder.embed(&spectra)?;
            for (fields, row) in metadata.drain(..).zip(rows) {
                write_row(&mut writer, &fields, &row.latent)?;
            }
            emitted += spectra.len();
            bar.set_position(emitted as u64);
        }

        writer.flush()?;
        bar.finish_with_message(format!("encoded {emitted} spectra"));
        println!("wrote embeddings: {}", args.output.display());
        Ok(())
    }

    fn write_header<W: Write>(writer: &mut W, latent_width: usize) -> std::io::Result<()> {
        for (index, column) in OUTPUT_COLUMNS.iter().enumerate() {
            if index > 0 {
                writer.write_all(b"\t")?;
            }
            writer.write_all(column.as_bytes())?;
        }
        for index in 0..latent_width {
            write!(writer, "\tz{index}")?;
        }
        writer.write_all(b"\n")
    }

    fn write_row<W: Write>(
        writer: &mut W,
        fields: &[String],
        latent: &[f32],
    ) -> std::io::Result<()> {
        for (index, field) in fields.iter().enumerate() {
            if index > 0 {
                writer.write_all(b"\t")?;
            }
            write_tsv_field(writer, field)?;
        }
        for value in latent {
            writer.write_all(b"\t")?;
            write!(writer, "{value:.8}")?;
        }
        writer.write_all(b"\n")
    }

    fn annotation_fields(index: usize, record: &MascotGenericFormat<f32>) -> Vec<String> {
        let metadata = record.metadata();
        let retention_time = metadata
            .retention_time()
            .filter(|value| value.is_finite())
            .map(format_f64);
        let precursor_mz = Some(format_f64(record.precursor_mz().to_f64()));
        let charge = record.charge().map(|value| value.to_string());
        let ion_mode = record.ion_mode().map(|value| value.to_string());
        let instrument = record.source_instrument().map(|value| value.to_string());
        let smiles = metadata.smiles().map(ToString::to_string);
        let formula = record.formula().map(|value| value.to_string());

        vec![
            index.to_string(),
            clean_option(record.feature_id()),
            clean_option(record.scans()),
            metadata_value(record, SPECTRUM_ID_KEY),
            metadata_value(record, NAME_KEY),
            clean_owned(smiles),
            metadata_value(record, INCHIKEY_KEY),
            clean_owned(formula),
            metadata_value(record, NPC_PATHWAY_KEY),
            metadata_value(record, NPC_SUPERCLASS_KEY),
            metadata_value(record, NPC_CLASS_KEY),
            metadata_value(record, CLASSYFIRE_KINGDOM_KEY),
            metadata_value(record, CLASSYFIRE_SUPERCLASS_KEY),
            metadata_value(record, CLASSYFIRE_CLASS_KEY),
            metadata_value(record, CLASSYFIRE_SUBCLASS_KEY),
            metadata_value(record, CLASSYFIRE_DIRECT_PARENT_KEY),
            metadata_value(record, SOURCE_KEY),
            clean_owned(retention_time),
            clean_owned(precursor_mz),
            clean_owned(charge),
            clean_owned(ion_mode),
            clean_owned(instrument),
        ]
    }

    fn metadata_value(record: &MascotGenericFormat<f32>, key: &str) -> String {
        record
            .metadata()
            .arbitrary_metadata_value(key)
            .map(clean_value)
            .unwrap_or_default()
    }

    fn clean_option(value: Option<&str>) -> String {
        value.map(clean_value).unwrap_or_default()
    }

    fn clean_owned(value: Option<String>) -> String {
        value.as_deref().map(clean_value).unwrap_or_default()
    }

    fn clean_value(value: &str) -> String {
        value.trim().to_owned()
    }

    fn write_tsv_field<W: Write>(writer: &mut W, field: &str) -> std::io::Result<()> {
        for byte in field.bytes() {
            match byte {
                b'\t' | b'\r' | b'\n' => writer.write_all(b" ")?,
                _ => writer.write_all(&[byte])?,
            }
        }
        Ok(())
    }

    fn progress_bar(total: u64) -> ProgressBar {
        let bar = ProgressBar::new(total);
        if let Ok(style) = ProgressStyle::with_template(
            "{msg} [{elapsed_precise}] {wide_bar} {pos}/{len} spectra {per_sec}",
        ) {
            bar.set_style(style);
        }
        bar.set_message("encoding annotated MS2");
        bar
    }

    fn format_f64(value: f64) -> String {
        format!("{value:.8}")
    }
}

#[cfg(all(feature = "cuda", feature = "embed", feature = "embed-mgf"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::main()
}

#[cfg(not(all(feature = "cuda", feature = "embed", feature = "embed-mgf")))]
fn main() {
    eprintln!(
        "encode_annotated_ms2_flat requires --no-default-features \
         --features cuda,embed,embed-mgf"
    );
}
