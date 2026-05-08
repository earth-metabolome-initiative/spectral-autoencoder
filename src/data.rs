//! Streaming MGF ingestion and vectorized samples.

use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
};

use mascot_rs::prelude::{Dataset, MGFIter, MascotError, MascotGenericFormat};
use mass_spectrometry::prelude::{Spectrum, SpectrumFloat};

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

/// Summarizes an MGF file.
pub fn summarize_mgf_path(path: impl AsRef<Path>, limit: Option<usize>) -> Result<MgfSummary> {
    let path = path.as_ref();
    let mut iter = MGFIter::<f32, _>::from_path(path)?;
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
        peaks,
        min_peaks,
        max_peaks,
    })
}

type MascotRecordResult<P> = std::result::Result<MascotGenericFormat<P>, MascotError>;
type MascotRecordIter<P> = dyn Iterator<Item = MascotRecordResult<P>>;

/// Iterator over vectorized spectra loaded from a mascot MGF stream.
pub struct VectorizedMgfIter<P = f32>
where
    P: SpectrumFloat,
{
    records: Box<MascotRecordIter<P>>,
    vectorizer: SpectrumVectorizer,
    conditioning: ConditioningEncoder,
    precision: PhantomData<P>,
}

/// Iterator over tokenized spectra loaded from a mascot MGF stream.
pub struct TokenizedMgfIter<P = f32>
where
    P: SpectrumFloat,
{
    records: Box<MascotRecordIter<P>>,
    tokenizer: SpectrumTokenizer,
    conditioning: ConditioningEncoder,
    precision: PhantomData<P>,
}

impl<P> VectorizedMgfIter<P>
where
    P: SpectrumFloat + 'static,
{
    /// Creates a new vectorized iterator over a mascot MGF record stream.
    pub fn from_records<R>(
        records: R,
        vectorizer: SpectrumVectorizer,
        conditioning: ConditioningEncoder,
    ) -> Self
    where
        R: Iterator<Item = MascotRecordResult<P>> + 'static,
    {
        Self {
            records: Box::new(records),
            vectorizer,
            conditioning,
            precision: PhantomData,
        }
    }

    /// Creates a new vectorized iterator over an MGF file.
    pub fn from_path(
        path: impl AsRef<Path>,
        vectorizer: SpectrumVectorizer,
        conditioning: ConditioningEncoder,
    ) -> Result<Self> {
        let records = MGFIter::<P, _>::from_path(path)?;
        Ok(Self::from_records(records, vectorizer, conditioning))
    }

    fn encode_record(&mut self, record: &MascotGenericFormat<P>) -> Result<AutoencoderSample> {
        Ok(AutoencoderSample {
            spectrum: self.vectorizer.encode(record)?.values,
            conditions: self.conditioning.encode(record),
        })
    }
}

impl<P> Iterator for VectorizedMgfIter<P>
where
    P: SpectrumFloat + 'static,
{
    type Item = Result<AutoencoderSample>;

    fn next(&mut self) -> Option<Self::Item> {
        let record = match self.records.next()? {
            Ok(record) => record,
            Err(error) => return Some(Err(error.into())),
        };
        Some(self.encode_record(&record))
    }
}

impl<P> TokenizedMgfIter<P>
where
    P: SpectrumFloat + 'static,
{
    /// Creates a new tokenized iterator over a mascot MGF record stream.
    pub fn from_records<R>(
        records: R,
        tokenizer: SpectrumTokenizer,
        conditioning: ConditioningEncoder,
    ) -> Self
    where
        R: Iterator<Item = MascotRecordResult<P>> + 'static,
    {
        Self {
            records: Box::new(records),
            tokenizer,
            conditioning,
            precision: PhantomData,
        }
    }

    /// Creates a new tokenized iterator over an MGF file.
    pub fn from_path(
        path: impl AsRef<Path>,
        tokenizer: SpectrumTokenizer,
        conditioning: ConditioningEncoder,
    ) -> Result<Self> {
        let records = MGFIter::<P, _>::from_path(path)?;
        Ok(Self::from_records(records, tokenizer, conditioning))
    }

    fn encode_record(
        &mut self,
        record: &MascotGenericFormat<P>,
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

impl<P> Iterator for TokenizedMgfIter<P>
where
    P: SpectrumFloat + 'static,
{
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

/// Creates a vectorized iterator over a mascot dataset.
pub async fn vectorized_dataset_iter<D>(
    dataset: D,
    vectorizer: SpectrumVectorizer,
    conditioning: ConditioningEncoder,
) -> Result<VectorizedMgfIter>
where
    D: Dataset,
    D::Iter: Iterator<Item = MascotRecordResult<f32>> + 'static,
{
    let records = dataset.mgf_iter().await?;
    Ok(VectorizedMgfIter::from_records(
        records,
        vectorizer,
        conditioning,
    ))
}

/// Creates a tokenized iterator over an MGF path.
pub fn tokenized_mgf_iter(
    path: impl AsRef<Path>,
    tokenizer: SpectrumTokenizer,
    conditioning: ConditioningEncoder,
) -> Result<TokenizedMgfIter> {
    TokenizedMgfIter::from_path(path, tokenizer, conditioning)
}

/// Creates a tokenized iterator over a mascot dataset.
pub async fn tokenized_dataset_iter<D>(
    dataset: D,
    tokenizer: SpectrumTokenizer,
    conditioning: ConditioningEncoder,
) -> Result<TokenizedMgfIter>
where
    D: Dataset,
    D::Iter: Iterator<Item = MascotRecordResult<f32>> + 'static,
{
    let records = dataset.mgf_iter().await?;
    Ok(TokenizedMgfIter::from_records(
        records,
        tokenizer,
        conditioning,
    ))
}
