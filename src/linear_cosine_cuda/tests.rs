use super::*;

use burn::{
    backend::{Autodiff, Cuda, cuda::CudaDevice},
    tensor::{Shape, Tensor as BurnTensor, TensorData, TensorPrimitive},
};
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::cubecl::{CubeDim, calculate_cube_count_elemwise};
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use mass_spectrometry::prelude::*;

type TestBackend = Autodiff<Cuda<f32, i32>>;
type RawTestBackend = CubeBackend<cubecl::cuda::CudaRuntime, f32, i32, u8>;
type ReferenceSpectrum = GenericSpectrum<f32>;

const TEST_MZ_POWER: f32 = 0.15;
const TEST_INTENSITY_POWER: f32 = 0.7;
const TEST_MZ_TOLERANCE: f32 = 0.02;
const TEST_EPSILON: f32 = 1.0e-8;
const TEST_MAX_PEAKS: usize = 128;
const PAIR_CHUNK_SIZE: usize = 64;

#[cube(launch)]
fn modified_linear_cosine_paired_test_forward<F: Float>(
    left_mz: &Tensor<F>,
    left_intensity: &Tensor<F>,
    left_precursor: &Tensor<F>,
    right_mz: &Tensor<F>,
    right_intensity: &Tensor<F>,
    right_precursor: &Tensor<F>,
    output: &mut Tensor<F>,
    mz_power: f32,
    intensity_power: f32,
    mz_tolerance: f32,
    epsilon: f32,
    #[comptime] max_peaks: usize,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let row = ABSOLUTE_POS;
    let mz_p = F::cast_from(mz_power);
    let intensity_p = F::cast_from(intensity_power);
    let tolerance = F::cast_from(mz_tolerance);
    let eps = F::cast_from(epsilon);

    output[row] = super::kernels::modified_linear_cosine_score_rows(
        left_mz,
        left_intensity,
        left_precursor,
        row,
        right_mz,
        right_intensity,
        right_precursor,
        row,
        mz_p,
        intensity_p,
        tolerance,
        eps,
        max_peaks,
    );
}

#[test]
fn paired_kernel_matches_cpu_linear_cosine_for_all_reference_pairs() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let pair_indices = all_pair_indices(spectra.len());
    let scorer = LinearCosine::new(
        f64::from(TEST_MZ_POWER),
        f64::from(TEST_INTENSITY_POWER),
        f64::from(TEST_MZ_TOLERANCE),
    )
    .expect("CPU linear cosine config should be valid");

    for chunk in pair_indices.chunks(PAIR_CHUNK_SIZE) {
        let pairs = pair_rows(&spectra, chunk);
        let row_count = pairs.indices.len();
        let scores = linear_cosine_preprocessed_paired_kernel(
            BurnTensor::<TestBackend, 2>::from_data(
                TensorData::new(pairs.left_mz, [row_count, pairs.peak_width]),
                &device,
            ),
            BurnTensor::<TestBackend, 2>::from_data(
                TensorData::new(pairs.left_intensity, [row_count, pairs.peak_width]),
                &device,
            ),
            BurnTensor::<TestBackend, 2>::from_data(
                TensorData::new(pairs.right_mz, [row_count, pairs.peak_width]),
                &device,
            ),
            BurnTensor::<TestBackend, 2>::from_data(
                TensorData::new(pairs.right_intensity, [row_count, pairs.peak_width]),
                &device,
            ),
            BurnTensor::<TestBackend, 1>::from_data(
                TensorData::new(vec![TEST_MZ_POWER; row_count], [row_count]),
                &device,
            ),
            BurnTensor::<TestBackend, 1>::from_data(
                TensorData::new(vec![TEST_INTENSITY_POWER; row_count], [row_count]),
                &device,
            ),
            BurnTensor::<TestBackend, 1>::from_data(
                TensorData::new(vec![TEST_MZ_TOLERANCE; row_count], [row_count]),
                &device,
            ),
            LinearCosineKernelConfig {
                epsilon: f64::from(TEST_EPSILON),
            },
        )
        .into_data()
        .to_vec::<f32>()
        .expect("kernel output should be f32");

        assert_all_pair_scores_match(&spectra, &pairs.indices, &scores, 1.0e-4, |left, right| {
            scorer
                .similarity(left, right)
                .map(|(score, _)| score as f32)
                .unwrap_or(0.0)
        });
    }
}

#[test]
fn paired_kernel_matches_cpu_modified_linear_cosine_for_all_reference_pairs() {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let pair_indices = all_pair_indices(spectra.len());
    let scorer = ModifiedLinearCosine::new(
        f64::from(TEST_MZ_POWER),
        f64::from(TEST_INTENSITY_POWER),
        f64::from(TEST_MZ_TOLERANCE),
    )
    .expect("CPU modified linear cosine config should be valid");

    for chunk in pair_indices.chunks(PAIR_CHUNK_SIZE) {
        let pairs = pair_rows(&spectra, chunk);
        let row_count = pairs.indices.len();
        let scores = modified_linear_cosine_paired_test_kernel(
            BurnTensor::<RawTestBackend, 2>::from_data(
                TensorData::new(pairs.left_mz, [row_count, pairs.peak_width]),
                &device,
            ),
            BurnTensor::<RawTestBackend, 2>::from_data(
                TensorData::new(pairs.left_intensity, [row_count, pairs.peak_width]),
                &device,
            ),
            BurnTensor::<RawTestBackend, 1>::from_data(
                TensorData::new(pairs.left_precursor, [row_count]),
                &device,
            ),
            BurnTensor::<RawTestBackend, 2>::from_data(
                TensorData::new(pairs.right_mz, [row_count, pairs.peak_width]),
                &device,
            ),
            BurnTensor::<RawTestBackend, 2>::from_data(
                TensorData::new(pairs.right_intensity, [row_count, pairs.peak_width]),
                &device,
            ),
            BurnTensor::<RawTestBackend, 1>::from_data(
                TensorData::new(pairs.right_precursor, [row_count]),
                &device,
            ),
            ModifiedLinearCosinePairConfig {
                mz_power: TEST_MZ_POWER,
                intensity_power: TEST_INTENSITY_POWER,
                mz_tolerance: TEST_MZ_TOLERANCE,
                max_peaks: pairs.peak_width,
                epsilon: TEST_EPSILON,
            },
        )
        .into_data()
        .to_vec::<f32>()
        .expect("kernel output should be f32");

        assert_all_pair_scores_match(&spectra, &pairs.indices, &scores, 2.0e-4, |left, right| {
            scorer
                .similarity(left, right)
                .map(|(score, _)| score as f32)
                .unwrap_or(0.0)
        });
    }
}

#[test]
fn ranking_kernel_matches_cpu_linear_cosine_for_reference_spectra() {
    assert_ranking_kernel_matches_cpu_reference(SIMILARITY_METRIC_LINEAR_COSINE, 1.0e-4);
}

#[test]
fn ranking_kernel_matches_cpu_modified_linear_cosine_for_reference_spectra() {
    assert_ranking_kernel_matches_cpu_reference(SIMILARITY_METRIC_MODIFIED_LINEAR_COSINE, 2.0e-4);
}

fn assert_ranking_kernel_matches_cpu_reference(metric: u32, tolerance: f32) {
    let device = CudaDevice::default();
    let spectra = reference_spectra();
    let spectra = &spectra[..12];
    let rows = spectrum_rows(spectra);
    let config = SimilarityRankingKernelConfig {
        batch_start: 1,
        batch_items: 10,
        candidates_per_anchor: 7,
        mz_power: f64::from(TEST_MZ_POWER),
        intensity_power: f64::from(TEST_INTENSITY_POWER),
        mz_tolerance: f64::from(TEST_MZ_TOLERANCE),
        metric,
        max_peaks: rows.peak_width,
        seed: 12_345,
        epsilon: f64::from(TEST_EPSILON),
    };

    let (candidate_index, best_candidate_position, top2_gap) =
        linear_cosine_similarity_ranking_kernel(
            BurnTensor::<TestBackend, 2>::from_data(
                TensorData::new(rows.mz, [spectra.len(), rows.peak_width]),
                &device,
            ),
            BurnTensor::<TestBackend, 2>::from_data(
                TensorData::new(rows.intensity, [spectra.len(), rows.peak_width]),
                &device,
            ),
            BurnTensor::<TestBackend, 1>::from_data(
                TensorData::new(rows.precursor, [spectra.len()]),
                &device,
            ),
            config,
        );

    let candidate_index = candidate_index
        .into_data()
        .to_vec::<i32>()
        .expect("candidate indices should be i32");
    let best_candidate_position = best_candidate_position
        .into_data()
        .to_vec::<i32>()
        .expect("best candidate positions should be i32");
    let top2_gap = top2_gap
        .into_data()
        .to_vec::<f32>()
        .expect("top-2 gaps should be f32");

    let candidate_count = config.effective_candidates_per_anchor();
    for anchor in 0..config.batch_items {
        let expected = ranking_reference(spectra, anchor, config);
        let start = anchor * candidate_count;
        let actual_candidates = &candidate_index[start..start + candidate_count];
        assert_eq!(
            actual_candidates,
            expected.candidate_indices.as_slice(),
            "anchor {anchor}"
        );
        assert_eq!(
            best_candidate_position[anchor] as usize, expected.best_candidate_position,
            "anchor {anchor}"
        );
        assert!(
            (top2_gap[anchor] - expected.top2_gap).abs() < tolerance,
            "anchor {anchor}: cuda={} cpu={}",
            top2_gap[anchor],
            expected.top2_gap,
        );
        assert!(!actual_candidates.contains(&(anchor as i32)));
        for (left, left_value) in actual_candidates.iter().enumerate() {
            for right_value in actual_candidates.iter().skip(left + 1) {
                assert_ne!(
                    left_value, right_value,
                    "anchor {anchor}: duplicate candidate {left_value}"
                );
            }
        }
    }
}

fn assert_all_pair_scores_match(
    spectra: &[(&'static str, ReferenceSpectrum)],
    indices: &[(usize, usize)],
    scores: &[f32],
    tolerance: f32,
    mut reference_score: impl FnMut(&ReferenceSpectrum, &ReferenceSpectrum) -> f32,
) {
    assert!(
        !indices.is_empty(),
        "reference collection should contain spectra"
    );
    let mut max_delta = 0.0_f32;
    let mut max_row = 0usize;
    let mut max_pair = ("", "");
    let mut failures = 0usize;

    for (row, &(left_index, right_index)) in indices.iter().enumerate() {
        let (left_name, left) = &spectra[left_index];
        let (right_name, right) = &spectra[right_index];
        let expected = reference_score(left, right);
        let delta = (scores[row] - expected).abs();
        if delta > max_delta {
            max_delta = delta;
            max_row = row;
            max_pair = (left_name, right_name);
        }
        if delta >= tolerance {
            failures += 1;
        }
    }

    assert!(
        failures == 0,
        "{failures} pair scores exceeded tolerance {tolerance}; max row {max_row} {} vs {} delta={max_delta}",
        max_pair.0,
        max_pair.1
    );
}

struct RankingReference {
    candidate_indices: Vec<i32>,
    best_candidate_position: usize,
    top2_gap: f32,
}

fn ranking_reference(
    spectra: &[(&'static str, ReferenceSpectrum)],
    anchor: usize,
    config: SimilarityRankingKernelConfig,
) -> RankingReference {
    assert!(config.batch_start + config.batch_items <= spectra.len());
    let linear_scorer =
        LinearCosine::new(config.mz_power, config.intensity_power, config.mz_tolerance)
            .expect("CPU linear cosine config should be valid");
    let modified_scorer =
        ModifiedLinearCosine::new(config.mz_power, config.intensity_power, config.mz_tolerance)
            .expect("CPU modified linear cosine config should be valid");
    let mut state =
        config.seed as u32 ^ (((anchor as u32) + 1) * 40503) ^ ((config.batch_start as u32) >> 16);
    if state == 0 {
        state = 0x6d2b_79f5;
    }
    let partner_slots = (config.batch_items - 1) as u32;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let offset = state % partner_slots;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    let mut stride = (state % partner_slots) + 1;
    while gcd(stride, partner_slots) != 1 {
        stride += 1;
        if stride > partner_slots {
            stride = 1;
        }
    }
    let mut best_score = f32::NEG_INFINITY;
    let mut second_best_score = f32::NEG_INFINITY;
    let mut best_candidate_position = 0usize;
    let candidates = config.effective_candidates_per_anchor();
    let anchor_spectrum = &spectra[config.batch_start + anchor].1;
    let mut candidate_indices = Vec::with_capacity(candidates);

    for candidate_position in 0..candidates {
        let mut local_partner =
            ((offset + (candidate_position as u32) * stride) % partner_slots) as usize;
        if local_partner >= anchor {
            local_partner += 1;
        }
        candidate_indices.push(local_partner as i32);
        let partner_spectrum = &spectra[config.batch_start + local_partner].1;
        let score = if config.metric == SIMILARITY_METRIC_MODIFIED_LINEAR_COSINE {
            modified_scorer
                .similarity(anchor_spectrum, partner_spectrum)
                .map(|(score, _)| score as f32)
                .unwrap_or(0.0)
        } else {
            linear_scorer
                .similarity(anchor_spectrum, partner_spectrum)
                .map(|(score, _)| score as f32)
                .unwrap_or(0.0)
        };
        if score > best_score {
            second_best_score = best_score;
            best_score = score;
            best_candidate_position = candidate_position;
        } else if score > second_best_score {
            second_best_score = score;
        }
    }

    RankingReference {
        candidate_indices,
        best_candidate_position,
        top2_gap: (best_score - second_best_score).max(0.0),
    }
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

#[derive(Clone, Copy)]
struct ModifiedLinearCosinePairConfig {
    mz_power: f32,
    intensity_power: f32,
    mz_tolerance: f32,
    max_peaks: usize,
    epsilon: f32,
}

fn modified_linear_cosine_paired_test_kernel<R, F, I, BT>(
    left_mz: BurnTensor<CubeBackend<R, F, I, BT>, 2>,
    left_intensity: BurnTensor<CubeBackend<R, F, I, BT>, 2>,
    left_precursor: BurnTensor<CubeBackend<R, F, I, BT>, 1>,
    right_mz: BurnTensor<CubeBackend<R, F, I, BT>, 2>,
    right_intensity: BurnTensor<CubeBackend<R, F, I, BT>, 2>,
    right_precursor: BurnTensor<CubeBackend<R, F, I, BT>, 1>,
    config: ModifiedLinearCosinePairConfig,
) -> BurnTensor<CubeBackend<R, F, I, BT>, 1>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let left_mz = left_mz.into_primitive().tensor();
    let left_intensity = left_intensity.into_primitive().tensor();
    let left_precursor = left_precursor.into_primitive().tensor();
    let right_mz = right_mz.into_primitive().tensor();
    let right_intensity = right_intensity.into_primitive().tensor();
    let right_precursor = right_precursor.into_primitive().tensor();

    left_mz.assert_is_on_same_device(&left_intensity);
    left_mz.assert_is_on_same_device(&left_precursor);
    left_mz.assert_is_on_same_device(&right_mz);
    left_mz.assert_is_on_same_device(&right_intensity);
    left_mz.assert_is_on_same_device(&right_precursor);

    let [batch_size, left_peaks] = left_mz.meta.shape().dims();
    let [right_rows, right_peaks] = right_mz.meta.shape().dims();
    assert_eq!(
        batch_size, right_rows,
        "paired modified linear cosine requires the same number of left and right rows"
    );
    assert!(left_peaks <= config.max_peaks);
    assert!(right_peaks <= config.max_peaks);

    let output_shape = Shape::new([batch_size]);
    let output = empty_device_dtype(
        left_mz.client.clone(),
        left_mz.device.clone(),
        output_shape,
        left_mz.dtype,
    );
    let cube_dim = CubeDim::new(&left_mz.client, batch_size);
    let cube_count = calculate_cube_count_elemwise(&left_mz.client, batch_size, cube_dim);

    let client = left_mz.client.clone();
    modified_linear_cosine_paired_test_forward::launch::<F, R>(
        &client,
        cube_count,
        cube_dim,
        left_mz.into_tensor_arg(),
        left_intensity.into_tensor_arg(),
        left_precursor.into_tensor_arg(),
        right_mz.into_tensor_arg(),
        right_intensity.into_tensor_arg(),
        right_precursor.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        config.mz_power,
        config.intensity_power,
        config.mz_tolerance,
        config.epsilon,
        config.max_peaks,
    );

    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}

struct SpectrumRows {
    mz: Vec<f32>,
    intensity: Vec<f32>,
    precursor: Vec<f32>,
    peak_width: usize,
}

struct PairRows {
    left_mz: Vec<f32>,
    left_intensity: Vec<f32>,
    left_precursor: Vec<f32>,
    right_mz: Vec<f32>,
    right_intensity: Vec<f32>,
    right_precursor: Vec<f32>,
    indices: Vec<(usize, usize)>,
    peak_width: usize,
}

fn spectrum_rows(spectra: &[(&'static str, ReferenceSpectrum)]) -> SpectrumRows {
    let peak_width = peak_width(spectra);
    let mut mz = Vec::with_capacity(spectra.len() * peak_width);
    let mut intensity = Vec::with_capacity(spectra.len() * peak_width);
    let mut precursor = Vec::with_capacity(spectra.len());

    for (_name, spectrum) in spectra {
        append_spectrum_row(spectrum, peak_width, &mut mz, &mut intensity);
        precursor.push(spectrum.precursor_mz());
    }

    SpectrumRows {
        mz,
        intensity,
        precursor,
        peak_width,
    }
}

fn all_pair_indices(count: usize) -> Vec<(usize, usize)> {
    (0..count)
        .flat_map(|left_index| (0..count).map(move |right_index| (left_index, right_index)))
        .collect()
}

fn pair_rows(
    spectra: &[(&'static str, ReferenceSpectrum)],
    indices: &[(usize, usize)],
) -> PairRows {
    let peak_width = peak_width(spectra);
    let pair_count = indices.len();
    let mut left_mz = Vec::with_capacity(pair_count * peak_width);
    let mut left_intensity = Vec::with_capacity(pair_count * peak_width);
    let mut left_precursor = Vec::with_capacity(pair_count);
    let mut right_mz = Vec::with_capacity(pair_count * peak_width);
    let mut right_intensity = Vec::with_capacity(pair_count * peak_width);
    let mut right_precursor = Vec::with_capacity(pair_count);

    for &(left_index, right_index) in indices {
        let left_spectrum = &spectra[left_index].1;
        let right_spectrum = &spectra[right_index].1;
        append_spectrum_row(left_spectrum, peak_width, &mut left_mz, &mut left_intensity);
        append_spectrum_row(
            right_spectrum,
            peak_width,
            &mut right_mz,
            &mut right_intensity,
        );
        left_precursor.push(left_spectrum.precursor_mz());
        right_precursor.push(right_spectrum.precursor_mz());
    }

    PairRows {
        left_mz,
        left_intensity,
        left_precursor,
        right_mz,
        right_intensity,
        right_precursor,
        indices: indices.to_vec(),
        peak_width,
    }
}

fn append_spectrum_row(
    spectrum: &ReferenceSpectrum,
    width: usize,
    mz_values: &mut Vec<f32>,
    intensity_values: &mut Vec<f32>,
) {
    let start = mz_values.len();
    for (mz, intensity) in spectrum.peaks() {
        mz_values.push(mz);
        intensity_values.push(intensity);
    }
    let length = mz_values.len() - start;
    assert!(
        length <= width,
        "reference spectrum has more peaks than the fixed row width"
    );
    mz_values.resize(start + width, 0.0);
    intensity_values.resize(start + width, 0.0);
}

fn peak_width(spectra: &[(&'static str, ReferenceSpectrum)]) -> usize {
    spectra
        .iter()
        .map(|(_name, spectrum)| spectrum.len())
        .max()
        .unwrap_or(0)
}

fn reference_spectra() -> Vec<(&'static str, ReferenceSpectrum)> {
    let processor = SiriusMergeClosePeaks::<f32>::new_with_precision(f64::from(TEST_MZ_TOLERANCE))
        .expect("reference-spectrum preprocess config should be valid");

    macro_rules! spectrum {
        ($method:ident) => {
            (
                stringify!($method),
                processor.process(
                    &ReferenceSpectrum::$method()
                        .expect("reference spectrum should build")
                        .top_k_peaks(TEST_MAX_PEAKS)
                        .expect("reference spectrum top-k should build"),
                ),
            )
        };
    }

    vec![
        spectrum!(acephate),
        spectrum!(acetyl_coenzyme_a),
        spectrum!(adenine),
        spectrum!(adenosine),
        spectrum!(adenosine_5_diphosphate),
        spectrum!(adenosine_5_monophosphate),
        spectrum!(alanine),
        spectrum!(arachidic_acid),
        spectrum!(arachidonic_acid),
        spectrum!(arginine),
        spectrum!(ascorbic_acid),
        spectrum!(aspartic_acid),
        spectrum!(aspirin),
        spectrum!(avermectin),
        spectrum!(biotin),
        spectrum!(boscalid),
        spectrum!(chlorantraniliprole),
        spectrum!(chlorfluazuron),
        spectrum!(chlorotoluron),
        spectrum!(citric_acid),
        spectrum!(clothianidin),
        spectrum!(cocaine),
        spectrum!(cyazofamid),
        spectrum!(cymoxanil),
        spectrum!(cysteine),
        spectrum!(cytidine),
        spectrum!(cytidine_5_diphosphate),
        spectrum!(cytidine_5_triphosphate),
        spectrum!(desmosterol),
        spectrum!(diflubenzuron),
        spectrum!(dihydrosphingosine),
        spectrum!(diniconazole),
        spectrum!(dinotefuran),
        spectrum!(diuron),
        spectrum!(doramectin),
        spectrum!(elaidic_acid),
        spectrum!(epimeloscine),
        spectrum!(eprinomectin),
        spectrum!(ethiprole),
        spectrum!(ethirimol),
        spectrum!(fipronil),
        spectrum!(flonicamid),
        spectrum!(fluazinam),
        spectrum!(fludioxinil),
        spectrum!(flufenoxuron),
        spectrum!(fluometuron),
        spectrum!(flutolanil),
        spectrum!(folic_acid),
        spectrum!(forchlorfenuron),
        spectrum!(fuberidazole),
        spectrum!(glucose),
        spectrum!(halofenozide),
        spectrum!(hexaflumuron),
        spectrum!(hydramethylnon),
        spectrum!(hydroxy_cholesterol),
        spectrum!(ivermectin),
        spectrum!(lufenuron),
        spectrum!(metaflumizone),
        spectrum!(neburon),
        spectrum!(nitenpyram),
        spectrum!(novaluron),
        spectrum!(phenylalanine),
        spectrum!(prothioconazole),
        spectrum!(pymetrozine),
        spectrum!(pyrimethanil),
        spectrum!(salicin),
        spectrum!(stypoltrione),
        spectrum!(sulfentrazone),
        spectrum!(tebufenozide),
        spectrum!(teflubenzuron),
        spectrum!(thidiazuron),
        spectrum!(thiophanate),
        spectrum!(triadimefon),
        spectrum!(triflumuron),
    ]
}
