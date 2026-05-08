#[cfg(feature = "std")]
mod app {
    use std::{
        env, fs, io,
        io::{BufWriter, Write},
        path::PathBuf,
    };

    use indicatif::{ProgressBar, ProgressStyle};
    use mascot_rs::prelude::{ANNOTATED_MS2_TOP_128_SPECTRA_COUNT, MGFIter, MGFVec};
    use mass_spectrometry::prelude::{Spectrum, SpectrumFloat};

    const DEFAULT_OUTPUT: &str =
        "runs/annotated-ms2-top128-flat-latent96-rtgap30-80epoch/spectrum_metrics.tsv";
    const DEFAULT_DATASET_DIR: &str = "datasets/annotated-ms2-top-128-peaks";

    const OUTPUT_COLUMNS: &[&str] = &[
        "index",
        "num_peaks",
        "pepmass",
        "charge",
        "charge_abs",
        "min_mz",
        "max_mz",
        "mz_range",
        "mean_mz",
        "median_mz",
        "intensity_weighted_mean_mz",
        "fragment_to_precursor_mean_ratio",
        "fragment_to_precursor_median_ratio",
        "tic",
        "base_peak_mz",
        "base_peak_intensity",
        "base_peak_fraction",
        "mean_intensity",
        "median_intensity",
        "intensity_entropy",
        "normalized_intensity_entropy",
        "peaks_below_precursor",
        "fraction_peaks_below_precursor",
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
        println!("output: {}", args.output.display());
        if let Some(limit) = args.limit {
            println!("limit: {limit}");
        }

        let mut records = MGFIter::<f64, _>::from_path(download.path())?.skipping_invalid_records();
        let mut writer = BufWriter::new(fs::File::create(&args.output)?);
        write_header(&mut writer)?;

        let total = args.limit.unwrap_or(ANNOTATED_MS2_TOP_128_SPECTRA_COUNT);
        let bar = progress_bar(total as u64);
        let mut written = 0usize;

        for record in records.by_ref() {
            let record = record?;
            let metrics = SpectrumMetrics::from_spectrum(written, &record);
            metrics.write(&mut writer)?;
            written += 1;
            if written.is_multiple_of(512) {
                bar.set_position(written as u64);
            }
            if args.limit.is_some_and(|limit| written >= limit) {
                break;
            }
        }

        writer.flush()?;
        bar.finish_with_message(format!(
            "wrote {written} spectra, skipped {} malformed records",
            records.skipped_records()
        ));
        println!("wrote metrics: {}", args.output.display());
        Ok(())
    }

    struct Args {
        output: PathBuf,
        dataset_dir: PathBuf,
        limit: Option<usize>,
        force_download: bool,
    }

    impl Args {
        fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
            Ok(Self {
                output: path_var("ANNOTATED_MS2_METRICS_OUT", DEFAULT_OUTPUT),
                dataset_dir: path_var("ANNOTATED_MS2_DIR", DEFAULT_DATASET_DIR),
                limit: optional_usize_var("ANNOTATED_MS2_LIMIT")?,
                force_download: bool_var("ANNOTATED_MS2_FORCE_DOWNLOAD", false)?,
            })
        }
    }

    struct SpectrumMetrics {
        index: usize,
        num_peaks: usize,
        pepmass: f64,
        charge: Option<i8>,
        min_mz: f64,
        max_mz: f64,
        mean_mz: f64,
        median_mz: f64,
        intensity_weighted_mean_mz: f64,
        tic: f64,
        base_peak_mz: f64,
        base_peak_intensity: f64,
        mean_intensity: f64,
        median_intensity: f64,
        intensity_entropy: f64,
        peaks_below_precursor: usize,
    }

    impl SpectrumMetrics {
        fn from_spectrum<S>(index: usize, spectrum: &S) -> Self
        where
            S: Spectrum<Precision = f64> + HasCharge,
        {
            let peaks: Vec<(f64, f64)> = spectrum
                .peaks()
                .map(|(mz, intensity)| (mz.to_f64(), intensity.to_f64()))
                .collect();
            let mz_values: Vec<f64> = peaks.iter().map(|(mz, _)| *mz).collect();
            let mut intensities: Vec<f64> = peaks.iter().map(|(_, intensity)| *intensity).collect();

            let num_peaks = peaks.len();
            let pepmass = spectrum.precursor_mz().to_f64();
            let min_mz = mz_values.first().copied().unwrap_or(f64::NAN);
            let max_mz = mz_values.last().copied().unwrap_or(f64::NAN);
            let mean_mz = mean(&mz_values);
            let median_mz = median_sorted(&mz_values);
            let tic = intensities.iter().copied().sum::<f64>();
            let intensity_weighted_mean_mz = if tic > 0.0 {
                peaks
                    .iter()
                    .map(|(mz, intensity)| mz * intensity)
                    .sum::<f64>()
                    / tic
            } else {
                f64::NAN
            };
            let (base_peak_mz, base_peak_intensity) = peaks
                .iter()
                .copied()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .unwrap_or((f64::NAN, f64::NAN));
            let mean_intensity = mean(&intensities);
            intensities.sort_by(f64::total_cmp);
            let median_intensity = median_sorted(&intensities);
            let intensity_entropy = shannon_entropy(&intensities, tic);
            let peaks_below_precursor = mz_values.iter().filter(|mz| **mz <= pepmass).count();

            Self {
                index,
                num_peaks,
                pepmass,
                charge: spectrum.charge(),
                min_mz,
                max_mz,
                mean_mz,
                median_mz,
                intensity_weighted_mean_mz,
                tic,
                base_peak_mz,
                base_peak_intensity,
                mean_intensity,
                median_intensity,
                intensity_entropy,
                peaks_below_precursor,
            }
        }

        fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
            let charge_abs = self.charge.map_or(f64::NAN, |value| f64::from(value.abs()));
            let mz_range = self.max_mz - self.min_mz;
            let mean_ratio = self.mean_mz / self.pepmass;
            let median_ratio = self.median_mz / self.pepmass;
            let base_peak_fraction = self.base_peak_intensity / self.tic;
            let normalized_entropy = if self.num_peaks > 1 {
                self.intensity_entropy / (self.num_peaks as f64).ln()
            } else {
                0.0
            };
            let fraction_peaks_below_precursor =
                self.peaks_below_precursor as f64 / self.num_peaks as f64;

            write!(writer, "{}", self.index)?;
            write!(writer, "\t{}", self.num_peaks)?;
            write_f64(writer, self.pepmass)?;
            match self.charge {
                Some(charge) => write!(writer, "\t{charge}")?,
                None => writer.write_all(b"\t")?,
            }
            write_f64(writer, charge_abs)?;
            write_f64(writer, self.min_mz)?;
            write_f64(writer, self.max_mz)?;
            write_f64(writer, mz_range)?;
            write_f64(writer, self.mean_mz)?;
            write_f64(writer, self.median_mz)?;
            write_f64(writer, self.intensity_weighted_mean_mz)?;
            write_f64(writer, mean_ratio)?;
            write_f64(writer, median_ratio)?;
            write_f64(writer, self.tic)?;
            write_f64(writer, self.base_peak_mz)?;
            write_f64(writer, self.base_peak_intensity)?;
            write_f64(writer, base_peak_fraction)?;
            write_f64(writer, self.mean_intensity)?;
            write_f64(writer, self.median_intensity)?;
            write_f64(writer, self.intensity_entropy)?;
            write_f64(writer, normalized_entropy)?;
            write!(writer, "\t{}", self.peaks_below_precursor)?;
            write_f64(writer, fraction_peaks_below_precursor)?;
            writer.write_all(b"\n")
        }
    }

    trait HasCharge {
        fn charge(&self) -> Option<i8>;
    }

    impl HasCharge for mascot_rs::prelude::MascotGenericFormat<f64> {
        fn charge(&self) -> Option<i8> {
            self.charge()
        }
    }

    fn write_header<W: Write>(writer: &mut W) -> io::Result<()> {
        for (index, column) in OUTPUT_COLUMNS.iter().enumerate() {
            if index > 0 {
                writer.write_all(b"\t")?;
            }
            writer.write_all(column.as_bytes())?;
        }
        writer.write_all(b"\n")
    }

    fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            f64::NAN
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    fn median_sorted(values: &[f64]) -> f64 {
        if values.is_empty() {
            return f64::NAN;
        }
        let middle = values.len() / 2;
        if values.len().is_multiple_of(2) {
            (values[middle - 1] + values[middle]) / 2.0
        } else {
            values[middle]
        }
    }

    fn shannon_entropy(intensities: &[f64], tic: f64) -> f64 {
        if tic <= 0.0 {
            return 0.0;
        }
        intensities
            .iter()
            .copied()
            .filter(|intensity| *intensity > 0.0)
            .map(|intensity| {
                let probability = intensity / tic;
                -probability * probability.ln()
            })
            .sum()
    }

    fn write_f64<W: Write>(writer: &mut W, value: f64) -> io::Result<()> {
        writer.write_all(b"\t")?;
        if value.is_finite() {
            write!(writer, "{value:.8}")?;
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
        bar.set_message("exporting annotated MS2 metrics");
        bar
    }

    fn path_var(name: &str, default: &str) -> PathBuf {
        env::var_os(name).map_or_else(|| PathBuf::from(default), PathBuf::from)
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

#[cfg(feature = "std")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::main()
}

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!("export_annotated_ms2_metrics requires --no-default-features --features std");
}
