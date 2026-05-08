use mascot_rs::prelude::{Dataset, DatasetFuture, MGFIter};
use spectral_autoencoder::{
    ConditioningEncoder, Result, SpectrumTokenizer, SpectrumVectorizer, VectorizedMgfIter,
    tokenized_dataset_iter, vectorized_dataset_iter,
};

const MGF: &str = r#"BEGIN IONS
FEATURE_ID=1
PEPMASS=250.0
CHARGE=1
MSLEVEL=2
IONMODE=Positive
SOURCE_INSTRUMENT=Orbitrap
RTINSECONDS=37.5
FILENAME=run-a.mzML
100.0 10.0
150.0 30.0
200.0 20.0
END IONS
"#;

#[test]
fn vectorizes_mgf_record_with_conditions() -> Result<()> {
    let mut records = MGFIter::<f32>::from_document(MGF);
    let record = match records.next() {
        Some(record) => record?,
        None => panic!("test MGF should contain one record"),
    };

    let spectrum = SpectrumVectorizer::default().encode(&record)?;
    let tokens = SpectrumTokenizer::default().encode(&record)?;
    let conditions = ConditioningEncoder::default().encode(&record);

    assert_eq!(spectrum.values.len(), 120);
    assert_eq!(spectrum.retained_peaks, 3);
    assert_eq!(tokens.features.len(), 60 * 19);
    assert_eq!(tokens.target_pairs.len(), 120);
    assert_eq!(tokens.retained_peaks, 3);
    assert_eq!(&tokens.padding_mask[..3], &[false, false, false]);
    assert!(tokens.padding_mask[3]);
    assert_eq!(conditions.len(), 2);
    assert_eq!(conditions[0], 0.125);
    assert_eq!(conditions[1], 1.0);

    Ok(())
}

#[test]
fn vectorized_iterator_wraps_mascot_mgf_stream() -> Result<()> {
    let records = MGFIter::<f32>::from_document(MGF);
    let mut iter = VectorizedMgfIter::from_records(
        records,
        SpectrumVectorizer::default(),
        ConditioningEncoder::default(),
    );
    let first = iter.next().expect("stream should yield one record")?;
    assert_eq!(first.spectrum.len(), 120);
    assert_eq!(
        first.conditions.len(),
        ConditioningEncoder::default().vector_width()
    );

    assert!(iter.next().is_none());
    Ok(())
}

struct InlineDataset;

impl Dataset for InlineDataset {
    type Download = ();
    type Iter = MGFIter<f32>;
    type Load = ();

    fn download(self) -> DatasetFuture<Self::Download> {
        Box::pin(async { Ok(()) })
    }

    fn mgf_iter(self) -> DatasetFuture<Self::Iter> {
        Box::pin(async { Ok(MGFIter::<f32>::from_document(MGF)) })
    }

    fn load(self) -> DatasetFuture<Self::Load> {
        Box::pin(async { Ok(()) })
    }
}

#[test]
fn vectorized_dataset_iterator_uses_mascot_dataset_stream() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime should be created");
    let mut iter = runtime.block_on(vectorized_dataset_iter(
        InlineDataset,
        SpectrumVectorizer::default(),
        ConditioningEncoder::default(),
    ))?;

    let sample = iter
        .next()
        .expect("dataset stream should yield one record")?;
    assert_eq!(sample.spectrum.len(), 120);
    assert!(iter.next().is_none());
    Ok(())
}

#[test]
fn tokenized_dataset_iterator_uses_mascot_dataset_stream() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime should be created");
    let mut iter = runtime.block_on(tokenized_dataset_iter(
        InlineDataset,
        SpectrumTokenizer::default(),
        ConditioningEncoder::default(),
    ))?;

    let sample = iter
        .next()
        .expect("dataset stream should yield one record")?;
    assert_eq!(sample.target_pairs.len(), 120);
    assert!(iter.next().is_none());
    Ok(())
}
