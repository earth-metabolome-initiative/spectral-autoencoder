//! Shared top-peak preprocessing.

use mass_spectrometry::prelude::{
    ELECTRON_MASS, GenericSpectrum, MAX_MZ, SpectrumAlloc, SpectrumFloat, SpectrumMut,
};

use crate::error::{Error, Result as CrateResult};

/// Top-k peak selection output.
pub(crate) struct TopPeaks {
    /// Retained peaks in model slot order: descending intensity, then ascending m/z.
    pub(crate) peaks: Vec<(f64, f64)>,
    /// Number of peaks available after finite/range filtering, before top-k truncation.
    pub(crate) candidate_peaks: usize,
}

/// Filters invalid/out-of-range peaks, selects top-k peaks through
/// [`SpectrumAlloc::top_k_peaks`], then returns them in model slot order.
pub(crate) fn filtered_top_peaks<S>(
    spectrum: &S,
    max_peaks: usize,
    min_mz: f64,
    max_mz: f64,
) -> Result<TopPeaks, S::MutationError>
where
    S: SpectrumAlloc,
{
    let lower_mz = finite_or(min_mz, ELECTRON_MASS).max(ELECTRON_MASS);
    let upper_mz = finite_or(max_mz, MAX_MZ).min(MAX_MZ);

    let mut peaks = spectrum
        .peaks()
        .filter_map(|(mz, intensity)| {
            let mz = mz.to_f64();
            let intensity = intensity.to_f64();
            if mz.is_finite()
                && intensity.is_finite()
                && mz >= lower_mz
                && mz <= upper_mz
                && intensity > 0.0
            {
                Some((mz, intensity))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let candidate_peaks = peaks.len();
    peaks.sort_by(|left, right| left.0.total_cmp(&right.0));
    let peaks = merge_duplicate_mz(peaks);

    let mut retained = if peaks.len() <= max_peaks {
        peaks
    } else {
        let mut filtered = S::with_capacity(
            sanitized_precursor_mz(spectrum.precursor_mz().to_f64()),
            peaks.len(),
        )?;
        filtered.add_peaks(peaks.into_iter().map(|(mz, intensity)| {
            (
                <S::Precision as SpectrumFloat>::from_f64_lossy(mz),
                <S::Precision as SpectrumFloat>::from_f64_lossy(intensity),
            )
        }))?;

        filtered
            .top_k_peaks(max_peaks)?
            .peaks()
            .map(|(mz, intensity)| (mz.to_f64(), intensity.to_f64()))
            .collect::<Vec<_>>()
    };
    retained.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.total_cmp(&right.0))
    });

    Ok(TopPeaks {
        peaks: retained,
        candidate_peaks,
    })
}

pub(crate) fn decode_normalized_peak_pairs(
    values: &[f32],
    expected: usize,
    max_mz: f64,
    precursor_mz: f64,
) -> CrateResult<GenericSpectrum<f64>> {
    if values.len() != expected {
        return Err(Error::InvalidVectorLength {
            actual: values.len(),
            expected,
        });
    }

    let mut peaks = values
        .as_chunks::<2>()
        .0
        .iter()
        .filter_map(|pair| {
            let mz = f64::from(pair[0]) * max_mz;
            let intensity = f64::from(pair[1]);
            if mz > 0.0 && intensity > 0.0 && mz.is_finite() && intensity.is_finite() {
                Some((mz, intensity))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    peaks.sort_by(|left, right| left.0.total_cmp(&right.0));
    peaks.dedup_by(|left, right| {
        if left.0.to_bits() == right.0.to_bits() {
            right.1 += left.1;
            true
        } else {
            false
        }
    });

    let mut spectrum = GenericSpectrum::try_with_capacity(precursor_mz.max(1.0), peaks.len())?;
    for (mz, intensity) in peaks {
        spectrum.add_peak(mz, intensity)?;
    }
    Ok(spectrum)
}

fn merge_duplicate_mz(peaks: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(peaks.len());
    for (mz, intensity) in peaks {
        if let Some((last_mz, last_intensity)) = merged.last_mut()
            && last_mz.to_bits() == mz.to_bits()
        {
            *last_intensity += intensity;
            continue;
        }
        merged.push((mz, intensity));
    }
    merged
}

fn sanitized_precursor_mz(precursor_mz: f64) -> f64 {
    if precursor_mz.is_finite() && (ELECTRON_MASS..=MAX_MZ).contains(&precursor_mz) {
        precursor_mz
    } else {
        1.0
    }
}

fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() { value } else { fallback }
}
