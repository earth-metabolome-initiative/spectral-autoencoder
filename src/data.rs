//! Streaming MGF ingestion and vectorized samples.

use std::path::{Path, PathBuf};

use mascot_rs::mascot_generic_format::MGFPathIter;
use mascot_rs::prelude::{MGFIter, MascotGenericFormat};
use mass_spectrometry::prelude::Spectrum;

use crate::batch::{AutoencoderSample, TokenizedAutoencoderSample};
use crate::conditioning::ConditioningEncoder;
use crate::error::{Error, Result};
use crate::tokenize::SpectrumTokenizer;
use crate::vectorize::SpectrumVectorizer;

/// Summary for an MGF file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MgfSummary {
    /// Input file path.
    pub path: PathBuf,
    /// Number of valid records observed.
    pub records: usize,
    /// Number of malformed records skipped by tolerant parsing.
    pub skipped_records: usize,
    /// Total number of peaks in valid records.
    pub peaks: usize,
    /// Minimum peaks per valid record.
    pub min_peaks: usize,
    /// Maximum peaks per valid record.
    pub max_peaks: usize,
}

impl MgfSummary {
    /// Returns the average number of peaks per record.
    #[must_use]
    pub fn mean_peaks(&self) -> Option<f64> {
        if self.records == 0 {
            None
        } else {
            Some(self.peaks as f64 / self.records as f64)
        }
    }
}

/// Summarizes an MGF file with tolerant parsing.
pub fn summarize_mgf_path(path: impl AsRef<Path>, limit: Option<usize>) -> Result<MgfSummary> {
    let path = path.as_ref();
    let mut iter = MGFIter::<f64, _>::from_path(path)?.skipping_invalid_records();
    let mut records = 0usize;
    let mut peaks = 0usize;
    let mut min_peaks = usize::MAX;
    let mut max_peaks = 0usize;

    for record in iter.by_ref() {
        let record = record?;
        let count = record.len();
        records += 1;
        peaks += count;
        min_peaks = min_peaks.min(count);
        max_peaks = max_peaks.max(count);
        if limit.is_some_and(|limit| records >= limit) {
            break;
        }
    }

    if records == 0 {
        return Err(Error::EmptyInput {
            path: path.to_path_buf(),
        });
    }

    Ok(MgfSummary {
        path: path.to_path_buf(),
        records,
        skipped_records: iter.skipped_records(),
        peaks,
        min_peaks,
        max_peaks,
    })
}

enum MgfRecordIter {
    Single(MGFPathIter<f64>),
    Multiple(MultiMgfRecordIter),
}

impl MgfRecordIter {
    fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::Single(
            MGFIter::<f64, _>::from_path(path)?.skipping_invalid_records(),
        ))
    }

    fn from_paths<PathLike, Paths>(paths: Paths) -> Result<Self>
    where
        PathLike: AsRef<Path>,
        Paths: IntoIterator<Item = PathLike>,
    {
        let paths = paths
            .into_iter()
            .map(|path| path.as_ref().to_path_buf())
            .collect::<Vec<_>>();
        if paths.is_empty() {
            return Err(Error::EmptyInput {
                path: PathBuf::from("<empty MGF path list>"),
            });
        }
        if paths.len() == 1 {
            return Self::from_path(&paths[0]);
        }

        Ok(Self::Multiple(MultiMgfRecordIter::new(paths)))
    }

    fn skipped_records(&self) -> usize {
        match self {
            Self::Single(records) => records.skipped_records(),
            Self::Multiple(records) => records.skipped_records(),
        }
    }
}

impl Iterator for MgfRecordIter {
    type Item = std::result::Result<MascotGenericFormat<f64>, mascot_rs::prelude::MascotError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Single(records) => records.next(),
            Self::Multiple(records) => records.next(),
        }
    }
}

struct MultiMgfRecordIter {
    paths: Vec<PathBuf>,
    next_path: usize,
    current: Option<MGFPathIter<f64>>,
    skipped_records: usize,
}

impl MultiMgfRecordIter {
    fn new(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            next_path: 0,
            current: None,
            skipped_records: 0,
        }
    }

    fn skipped_records(&self) -> usize {
        self.skipped_records + self.current.as_ref().map_or(0, MGFIter::skipped_records)
    }

    fn open_next_path(
        &mut self,
    ) -> Option<std::result::Result<(), mascot_rs::prelude::MascotError>> {
        let path = self.paths.get(self.next_path)?;
        self.next_path += 1;
        match MGFIter::<f64, _>::from_path(path) {
            Ok(records) => {
                self.current = Some(records.skipping_invalid_records());
                Some(Ok(()))
            }
            Err(error) => Some(Err(error)),
        }
    }
}

impl Iterator for MultiMgfRecordIter {
    type Item = std::result::Result<MascotGenericFormat<f64>, mascot_rs::prelude::MascotError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(records) = self.current.as_mut() {
                if let Some(record) = records.next() {
                    return Some(record);
                }
                self.skipped_records += records.skipped_records();
                self.current = None;
            }

            match self.open_next_path()? {
                Ok(()) => {}
                Err(error) => return Some(Err(error)),
            }
        }
    }
}

/// Iterator over vectorized spectra loaded from one or more MGF paths.
pub struct VectorizedMgfIter {
    records: MgfRecordIter,
    vectorizer: SpectrumVectorizer,
    conditioning: ConditioningEncoder,
}

/// Iterator over tokenized spectra loaded from one or more MGF paths.
pub struct TokenizedMgfIter {
    records: MgfRecordIter,
    tokenizer: SpectrumTokenizer,
    conditioning: ConditioningEncoder,
}

impl VectorizedMgfIter {
    /// Creates a new vectorized iterator over an MGF file.
    pub fn from_path(
        path: impl AsRef<Path>,
        vectorizer: SpectrumVectorizer,
        conditioning: ConditioningEncoder,
    ) -> Result<Self> {
        Ok(Self {
            records: MgfRecordIter::from_path(path)?,
            vectorizer,
            conditioning,
        })
    }

    /// Creates a new vectorized iterator over multiple MGF files.
    pub fn from_paths<PathLike, Paths>(
        paths: Paths,
        vectorizer: SpectrumVectorizer,
        conditioning: ConditioningEncoder,
    ) -> Result<Self>
    where
        PathLike: AsRef<Path>,
        Paths: IntoIterator<Item = PathLike>,
    {
        Ok(Self {
            records: MgfRecordIter::from_paths(paths)?,
            vectorizer,
            conditioning,
        })
    }

    /// Returns the number of invalid records skipped so far.
    #[must_use]
    pub fn skipped_records(&self) -> usize {
        self.records.skipped_records()
    }

    fn encode_record(&mut self, record: &MascotGenericFormat<f64>) -> Result<AutoencoderSample> {
        Ok(AutoencoderSample {
            spectrum: self.vectorizer.encode(record)?.values,
            conditions: self.conditioning.encode(record),
        })
    }
}

impl Iterator for VectorizedMgfIter {
    type Item = Result<AutoencoderSample>;

    fn next(&mut self) -> Option<Self::Item> {
        let record = match self.records.next()? {
            Ok(record) => record,
            Err(error) => return Some(Err(error.into())),
        };
        Some(self.encode_record(&record))
    }
}

impl TokenizedMgfIter {
    /// Creates a new tokenized iterator over an MGF file.
    pub fn from_path(
        path: impl AsRef<Path>,
        tokenizer: SpectrumTokenizer,
        conditioning: ConditioningEncoder,
    ) -> Result<Self> {
        Ok(Self {
            records: MgfRecordIter::from_path(path)?,
            tokenizer,
            conditioning,
        })
    }

    /// Creates a new tokenized iterator over multiple MGF files.
    pub fn from_paths<PathLike, Paths>(
        paths: Paths,
        tokenizer: SpectrumTokenizer,
        conditioning: ConditioningEncoder,
    ) -> Result<Self>
    where
        PathLike: AsRef<Path>,
        Paths: IntoIterator<Item = PathLike>,
    {
        Ok(Self {
            records: MgfRecordIter::from_paths(paths)?,
            tokenizer,
            conditioning,
        })
    }

    /// Returns the number of invalid records skipped so far.
    #[must_use]
    pub fn skipped_records(&self) -> usize {
        self.records.skipped_records()
    }

    fn encode_record(
        &mut self,
        record: &MascotGenericFormat<f64>,
    ) -> Result<TokenizedAutoencoderSample> {
        let tokens = self.tokenizer.encode(record)?;
        Ok(TokenizedAutoencoderSample {
            token_features: tokens.features,
            target_pairs: tokens.target_pairs,
            peak_mask: tokens.peak_mask,
            padding_mask: tokens.padding_mask,
            conditions: self.conditioning.encode(record),
        })
    }
}

impl Iterator for TokenizedMgfIter {
    type Item = Result<TokenizedAutoencoderSample>;

    fn next(&mut self) -> Option<Self::Item> {
        let record = match self.records.next()? {
            Ok(record) => record,
            Err(error) => return Some(Err(error.into())),
        };
        Some(self.encode_record(&record))
    }
}

/// Creates a vectorized iterator over an MGF path.
pub fn vectorized_mgf_iter(
    path: impl AsRef<Path>,
    vectorizer: SpectrumVectorizer,
    conditioning: ConditioningEncoder,
) -> Result<VectorizedMgfIter> {
    VectorizedMgfIter::from_path(path, vectorizer, conditioning)
}

/// Creates a vectorized iterator over multiple MGF paths.
pub fn vectorized_mgf_paths_iter<PathLike, Paths>(
    paths: Paths,
    vectorizer: SpectrumVectorizer,
    conditioning: ConditioningEncoder,
) -> Result<VectorizedMgfIter>
where
    PathLike: AsRef<Path>,
    Paths: IntoIterator<Item = PathLike>,
{
    VectorizedMgfIter::from_paths(paths, vectorizer, conditioning)
}

/// Creates a tokenized iterator over an MGF path.
pub fn tokenized_mgf_iter(
    path: impl AsRef<Path>,
    tokenizer: SpectrumTokenizer,
    conditioning: ConditioningEncoder,
) -> Result<TokenizedMgfIter> {
    TokenizedMgfIter::from_path(path, tokenizer, conditioning)
}

/// Creates a tokenized iterator over multiple MGF paths.
pub fn tokenized_mgf_paths_iter<PathLike, Paths>(
    paths: Paths,
    tokenizer: SpectrumTokenizer,
    conditioning: ConditioningEncoder,
) -> Result<TokenizedMgfIter>
where
    PathLike: AsRef<Path>,
    Paths: IntoIterator<Item = PathLike>,
{
    TokenizedMgfIter::from_paths(paths, tokenizer, conditioning)
}
