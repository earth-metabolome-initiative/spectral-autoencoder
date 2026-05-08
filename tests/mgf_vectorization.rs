use mascot_rs::prelude::MGFIter;
use spectral_autoencoder::{
    ConditioningEncoder, Result, SpectrumTokenizer, SpectrumVectorizer, vectorized_mgf_paths_iter,
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
    let mut records = MGFIter::<f64>::from_document(MGF);
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
fn vectorized_iterator_streams_multiple_mgf_paths() -> Result<()> {
    let directory = std::env::temp_dir().join(format!(
        "spectral-autoencoder-mgf-paths-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).expect("temporary directory should be writable");
    let first_path = directory.join("first.mgf");
    let second_path = directory.join("second.mgf");
    std::fs::write(&first_path, MGF).expect("test MGF should be writable");
    std::fs::write(&second_path, MGF.replace("FEATURE_ID=1", "FEATURE_ID=2"))
        .expect("test MGF should be writable");

    let mut iter = vectorized_mgf_paths_iter(
        [&first_path, &second_path],
        SpectrumVectorizer::default(),
        ConditioningEncoder::default(),
    )?;
    let first = iter.next().expect("first path should yield one record")?;
    assert_eq!(first.spectrum.len(), 120);
    assert_eq!(
        first.conditions.len(),
        ConditioningEncoder::default().vector_width()
    );

    let second = iter.next().expect("second path should yield one record")?;
    assert_eq!(second.spectrum.len(), 120);
    assert_eq!(
        second.conditions.len(),
        ConditioningEncoder::default().vector_width()
    );
    assert!(iter.next().is_none());

    std::fs::remove_file(first_path).ok();
    std::fs::remove_file(second_path).ok();
    std::fs::remove_dir(directory).ok();
    Ok(())
}
