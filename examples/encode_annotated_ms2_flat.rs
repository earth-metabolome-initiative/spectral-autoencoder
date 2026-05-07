#[cfg(all(feature = "cuda-fusion", feature = "std"))]
mod app {
    use std::{
        env, fs, io,
        io::{BufWriter, Write},
        path::{Path, PathBuf},
    };

    use burn::{
        backend::{Cuda, cuda::CudaDevice},
        module::Module,
        record::{CompactRecorder, Recorder},
        tensor::{Tensor, TensorData},
    };
    use indicatif::{ProgressBar, ProgressStyle};
    use mascot_rs::prelude::{
        ANNOTATED_MS2_TOP_128_SPECTRA_COUNT, MGFIter, MGFVec, MascotGenericFormat,
    };
    use mass_spectrometry::prelude::{Spectrum, SpectrumFloat};
    use spectral_autoencoder::{
        ConditioningEncoder, SpectralAutoencoderConfig, SpectrumVectorizer,
        SpectrumVectorizerConfig,
    };

    type Backend = Cuda<f32, i32>;

    const MAX_PEAKS: usize = 128;
    const DEFAULT_MODEL: &str = "runs/gems-a10-top128-flat-cache15-bs2048-80epoch-latent96-rtgap30-rankw5-ret005/flat_vector_model";
    const DEFAULT_OUTPUT: &str =
        "runs/annotated-ms2-top128-flat-latent96-rtgap30-80epoch/embeddings.tsv";
    const DEFAULT_DATASET_DIR: &str = "datasets/annotated-ms2-top-128-peaks";
    const DEFAULT_BATCH_SIZE: usize = 8192;

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

    const SPECTRUM_ID_KEYS: &[&str] = &["SPECTRUMID", "SPECTRUM_ID", "USI", "IDENTIFIER"];
    const NAME_KEYS: &[&str] = &["NAME", "COMPOUND_NAME", "COMPOUND", "TITLE"];
    const INCHIKEY_KEYS: &[&str] = &["INCHIKEY", "INCHI_KEY", "INCHIKEY2D"];
    const SOURCE_KEYS: &[&str] = &[
        "SOURCE_DATASET",
        "SOURCE",
        "DATASET",
        "ORIGIN",
        "LIBRARY",
        "PROVENANCE",
    ];
    const NPC_PATHWAY_KEYS: &[&str] = &[
        "NPC_PATHWAY",
        "NPC_PATHWAYS",
        "NPCLASSIFIER_PATHWAY",
        "NP_CLASSIFIER_PATHWAY",
        "NPCPathway",
        "npc_pathway",
        "pathway",
    ];
    const NPC_SUPERCLASS_KEYS: &[&str] = &[
        "NPC_SUPERCLASS",
        "NPCLASSIFIER_SUPERCLASS",
        "NP_CLASSIFIER_SUPERCLASS",
        "NPCSuperclass",
        "npc_superclass",
    ];
    const NPC_CLASS_KEYS: &[&str] = &[
        "NPC_CLASS",
        "NPCLASSIFIER_CLASS",
        "NP_CLASSIFIER_CLASS",
        "NPCClass",
        "npc_class",
    ];
    const CLASSYFIRE_KINGDOM_KEYS: &[&str] = &[
        "CHEMONT_KINGDOM",
        "CLASSYFIRE_KINGDOM",
        "CLASSYFIRE_KINGDOM_NAME",
        "CF_KINGDOM",
        "kingdom",
    ];
    const CLASSYFIRE_SUPERCLASS_KEYS: &[&str] = &[
        "CHEMONT_SUPERCLASS",
        "CLASSYFIRE_SUPERCLASS",
        "CLASSYFIRE_SUPERCLASS_NAME",
        "CF_SUPERCLASS",
        "superclass",
    ];
    const CLASSYFIRE_CLASS_KEYS: &[&str] = &[
        "CHEMONT_CLASS",
        "CLASSYFIRE_CLASS",
        "CLASSYFIRE_CLASS_NAME",
        "CF_CLASS",
        "class",
    ];
    const CLASSYFIRE_SUBCLASS_KEYS: &[&str] = &[
        "CHEMONT_SUBCLASS",
        "CLASSYFIRE_SUBCLASS",
        "CLASSYFIRE_SUBCLASS_NAME",
        "CF_SUBCLASS",
        "subclass",
    ];
    const CLASSYFIRE_DIRECT_PARENT_KEYS: &[&str] = &[
        "CHEMONT_DIRECT_PARENT",
        "CLASSYFIRE_DIRECT_PARENT",
        "CLASSYFIRE_DIRECT_PARENT_NAME",
        "CF_DIRECT_PARENT",
        "direct_parent",
    ];

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let args = Args::from_env()?;
        if let Some(parent) = args.output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir_all(&args.dataset_dir)?;

        let runtime = tokio::runtime::Runtime::new()?;
        let download = runtime.block_on(
            MGFVec::<f64>::annotated_ms2_top_128_peaks()
                .target_directory(&args.dataset_dir)
                .force_download(args.force_download)
                .verbose()
                .download(),
        )?;

        println!("annotated MS2 MGF: {}", download.path().display());
        println!("model: {}", args.model.display());
        println!("output: {}", args.output.display());
        println!("device: cuda:{}", args.device);
        println!("batch size: {}", args.batch_size);
        if let Some(limit) = args.limit {
            println!("limit: {limit}");
        }

        let device = CudaDevice::new(args.device);
        let config = SpectralAutoencoderConfig::twenty_million_run_with_peaks(MAX_PEAKS);
        let model = config.init::<Backend>(&device);
        let record = CompactRecorder::new().load(record_base_path(&args.model), &device)?;
        let model = model.load_record(record);

        let vectorizer = SpectrumVectorizer::new(SpectrumVectorizerConfig {
            max_peaks: MAX_PEAKS,
            ..SpectrumVectorizerConfig::default()
        });
        let conditioning = ConditioningEncoder::default();
        let mut records = MGFIter::<f64, _>::from_path(download.path())?.skipping_invalid_records();
        let mut writer = BufWriter::new(fs::File::create(&args.output)?);
        write_header(&mut writer, config.encoder.latent_width)?;

        let total = args.limit.unwrap_or(ANNOTATED_MS2_TOP_128_SPECTRA_COUNT);
        let bar = progress_bar(total as u64);
        let mut spectra = Vec::with_capacity(args.batch_size * vectorizer.vector_width());
        let mut conditions = Vec::with_capacity(args.batch_size * conditioning.vector_width());
        let mut annotations = Vec::with_capacity(args.batch_size);
        let mut written = 0usize;

        for record in records.by_ref() {
            let record = record?;
            let annotation = annotation_fields(written + annotations.len(), &record);
            spectra.extend(vectorizer.encode(&record)?.values);
            conditions.extend(conditioning.encode(&record));
            annotations.push(annotation);

            if annotations.len() == args.batch_size {
                write_batch(
                    &model,
                    &device,
                    &mut writer,
                    &mut spectra,
                    &mut conditions,
                    &mut annotations,
                    config.encoder.spectrum_width,
                    config.encoder.condition_width,
                    config.encoder.latent_width,
                )?;
                written += args.batch_size;
                bar.set_position(written as u64);
                if args.limit.is_some_and(|limit| written >= limit) {
                    break;
                }
            }

            if args
                .limit
                .is_some_and(|limit| written + annotations.len() >= limit)
            {
                break;
            }
        }

        if !annotations.is_empty() {
            let batch_items = annotations.len();
            write_batch(
                &model,
                &device,
                &mut writer,
                &mut spectra,
                &mut conditions,
                &mut annotations,
                config.encoder.spectrum_width,
                config.encoder.condition_width,
                config.encoder.latent_width,
            )?;
            written += batch_items;
            bar.set_position(written as u64);
        }

        writer.flush()?;
        bar.finish_with_message(format!(
            "encoded {written} spectra, skipped {} malformed records",
            records.skipped_records()
        ));
        println!("wrote embeddings: {}", args.output.display());
        Ok(())
    }

    struct Args {
        model: PathBuf,
        output: PathBuf,
        dataset_dir: PathBuf,
        batch_size: usize,
        device: usize,
        limit: Option<usize>,
        force_download: bool,
    }

    impl Args {
        fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
            Ok(Self {
                model: path_var("ANNOTATED_MS2_MODEL", DEFAULT_MODEL),
                output: path_var("ANNOTATED_MS2_OUT", DEFAULT_OUTPUT),
                dataset_dir: path_var("ANNOTATED_MS2_DIR", DEFAULT_DATASET_DIR),
                batch_size: usize_var("ANNOTATED_MS2_BATCH_SIZE", DEFAULT_BATCH_SIZE)?,
                device: usize_var("ANNOTATED_MS2_CUDA_DEVICE", 0)?,
                limit: optional_usize_var("ANNOTATED_MS2_LIMIT")?,
                force_download: bool_var("ANNOTATED_MS2_FORCE_DOWNLOAD", false)?,
            })
        }
    }

    fn write_batch<W: Write>(
        model: &spectral_autoencoder::SpectralAutoencoder<Backend>,
        device: &CudaDevice,
        writer: &mut W,
        spectra: &mut Vec<f32>,
        conditions: &mut Vec<f32>,
        annotations: &mut Vec<Vec<String>>,
        spectrum_width: usize,
        condition_width: usize,
        latent_width: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let batch_size = annotations.len();
        let spectra_tensor = Tensor::<Backend, 2>::from_data(
            TensorData::new(core::mem::take(spectra), [batch_size, spectrum_width]),
            device,
        );
        let condition_tensor = Tensor::<Backend, 2>::from_data(
            TensorData::new(core::mem::take(conditions), [batch_size, condition_width]),
            device,
        );
        let latent = model.encoder.forward(spectra_tensor, condition_tensor);
        let latent = latent.into_data().to_vec::<f32>()?;

        for (index, fields) in annotations.drain(..).enumerate() {
            write_tsv_fields(writer, &fields)?;
            let start = index * latent_width;
            for value in &latent[start..start + latent_width] {
                writer.write_all(b"\t")?;
                write!(writer, "{value:.8}")?;
            }
            writer.write_all(b"\n")?;
        }

        spectra.reserve(batch_size * spectrum_width);
        conditions.reserve(batch_size * condition_width);
        Ok(())
    }

    fn write_header<W: Write>(writer: &mut W, latent_width: usize) -> io::Result<()> {
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

    fn annotation_fields(index: usize, record: &MascotGenericFormat<f64>) -> Vec<String> {
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
        let formula = record.formula().map(ToString::to_string);

        vec![
            index.to_string(),
            clean_option(record.feature_id()),
            clean_option(record.scans()),
            metadata_value(record, SPECTRUM_ID_KEYS),
            metadata_value(record, NAME_KEYS),
            clean_owned(smiles),
            metadata_value(record, INCHIKEY_KEYS),
            clean_owned(formula),
            metadata_value(record, NPC_PATHWAY_KEYS),
            metadata_value(record, NPC_SUPERCLASS_KEYS),
            metadata_value(record, NPC_CLASS_KEYS),
            metadata_value(record, CLASSYFIRE_KINGDOM_KEYS),
            metadata_value(record, CLASSYFIRE_SUPERCLASS_KEYS),
            metadata_value(record, CLASSYFIRE_CLASS_KEYS),
            metadata_value(record, CLASSYFIRE_SUBCLASS_KEYS),
            metadata_value(record, CLASSYFIRE_DIRECT_PARENT_KEYS),
            metadata_value(record, SOURCE_KEYS),
            clean_owned(retention_time),
            clean_owned(precursor_mz),
            clean_owned(charge),
            clean_owned(ion_mode),
            clean_owned(instrument),
        ]
    }

    fn metadata_value(record: &MascotGenericFormat<f64>, keys: &[&str]) -> String {
        let metadata = record.metadata();
        for key in keys {
            if let Some(value) = metadata.arbitrary_metadata_value(key).and_then(clean_label) {
                return value.to_owned();
            }
        }
        for (observed_key, value) in metadata.arbitrary_metadata() {
            if keys
                .iter()
                .any(|key| observed_key.eq_ignore_ascii_case(key))
            {
                if let Some(value) = clean_label(value) {
                    return value.to_owned();
                }
            }
        }
        String::new()
    }

    fn clean_option(value: Option<&str>) -> String {
        value.and_then(clean_label).unwrap_or_default().to_owned()
    }

    fn clean_owned(value: Option<String>) -> String {
        value
            .as_deref()
            .and_then(clean_label)
            .unwrap_or_default()
            .to_owned()
    }

    fn clean_label(value: &str) -> Option<&str> {
        let value = value.trim();
        if value.is_empty()
            || value.eq_ignore_ascii_case("n/a")
            || value.eq_ignore_ascii_case("na")
            || value.eq_ignore_ascii_case("none")
            || value.eq_ignore_ascii_case("null")
            || value.eq_ignore_ascii_case("unknown")
        {
            None
        } else {
            Some(value)
        }
    }

    fn write_tsv_fields<W: Write>(writer: &mut W, fields: &[String]) -> io::Result<()> {
        for (index, field) in fields.iter().enumerate() {
            if index > 0 {
                writer.write_all(b"\t")?;
            }
            write_tsv_field(writer, field)?;
        }
        Ok(())
    }

    fn write_tsv_field<W: Write>(writer: &mut W, field: &str) -> io::Result<()> {
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

    fn record_base_path(path: &Path) -> PathBuf {
        if path.extension().and_then(|ext| ext.to_str()) == Some("mpk") {
            return path.with_extension("");
        }
        path.to_path_buf()
    }

    fn format_f64(value: f64) -> String {
        format!("{value:.8}")
    }

    fn path_var(name: &str, default: &str) -> PathBuf {
        env::var_os(name).map_or_else(|| PathBuf::from(default), PathBuf::from)
    }

    fn usize_var(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
        match env::var(name) {
            Ok(value) => Ok(value.parse()?),
            Err(env::VarError::NotPresent) => Ok(default),
            Err(error) => Err(Box::new(error)),
        }
    }

    fn optional_usize_var(name: &str) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        match env::var(name) {
            Ok(value) if value.trim().is_empty() => Ok(None),
            Ok(value) => Ok(Some(value.parse()?)),
            Err(env::VarError::NotPresent) => Ok(None),
            Err(error) => Err(Box::new(error)),
        }
    }

    fn bool_var(name: &str, default: bool) -> Result<bool, Box<dyn std::error::Error>> {
        match env::var(name) {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{name} must be a boolean"),
                )
                .into()),
            },
            Err(env::VarError::NotPresent) => Ok(default),
            Err(error) => Err(Box::new(error)),
        }
    }
}

#[cfg(all(feature = "cuda-fusion", feature = "std"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::main()
}

#[cfg(not(all(feature = "cuda-fusion", feature = "std")))]
fn main() {
    eprintln!(
        "encode_annotated_ms2_flat requires --no-default-features --features std,cuda-fusion"
    );
}
