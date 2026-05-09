use std::sync::Arc;

use burn::{
    data::dataloader::{DataLoader, DataLoaderIterator, Progress},
    tensor::{Bool, Distribution, Tensor, TensorData, backend::Backend},
};
use spectral_autoencoder::{
    SpectrumAugmentationConfig, TokenizedAutoencoderBatch, TokenizedMgfIter,
};

use crate::gems_common::{
    LoaderProgress, PairSamplingEpoch, RunArgs, SimilarityTeacherBackend, SimilarityTeacherConfig,
    StreamingLoaderOptions, StreamingTrainingLoaderConfig, TeacherGpuCache, TeacherSpectraBuilder,
    finish_loader_once, loader_epoch_items, mask_precursor_conditions, open_records_or_panic,
    probability_mask, signed_random, similarity_pair_seed, skip_split_records,
    teacher_similarity_ranking_batch,
};

pub fn streaming_tokenized_loader<B, Open>(
    args: &RunArgs,
    device: B::Device,
    progress: LoaderProgress,
    start_item: usize,
    augment: Option<SpectrumAugmentationConfig>,
    loader_config: StreamingTrainingLoaderConfig,
    open_records: Open,
) -> Arc<dyn DataLoader<B, TokenizedAutoencoderBatch<B>>>
where
    B: SimilarityTeacherBackend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<TokenizedMgfIter> + Send + Sync + Clone + 'static,
{
    Arc::new(StreamingTokenizedMgfLoader::new(
        StreamingLoaderOptions {
            mgf_source: args.mgf_source.clone(),
            batch_size: args.batch_size,
            max_batches: progress.max_batches,
            start_item,
            window_items: loader_config.window_items(args.batch_size, progress.max_batches),
            device,
            progress,
            augment,
            similarity_teacher: loader_config.similarity_teacher,
            randomize_pair_sampling: augment.is_some(),
        },
        open_records,
    ))
}

type AugmentedTokenBatch<B> = (
    Tensor<B, 3>,
    Tensor<B, 2>,
    Tensor<B, 2, Bool>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
);

fn augment_token_batch<B: Backend>(
    token_features: Tensor<B, 3>,
    peak_mask: Tensor<B, 2>,
    padding_mask: Tensor<B, 2, Bool>,
    conditions: Tensor<B, 2>,
    config: SpectrumAugmentationConfig,
) -> AugmentedTokenBatch<B> {
    let [batch_size, max_peaks, feature_width] = token_features.dims();
    if config.is_disabled() {
        let device = token_features.device();
        return (
            token_features,
            peak_mask,
            padding_mask,
            conditions,
            Tensor::<B, 2>::zeros([batch_size, 1], &device),
            Tensor::<B, 2>::zeros([batch_size, max_peaks], &device),
            Tensor::<B, 2>::zeros([batch_size, max_peaks], &device),
        );
    }

    let device = token_features.device();
    let mut mz = token_features
        .clone()
        .narrow(2, 0, 1)
        .reshape([batch_size, max_peaks]);
    let mut intensity = token_features
        .clone()
        .narrow(2, 1, 1)
        .reshape([batch_size, max_peaks]);
    let mut presence = token_features
        .clone()
        .narrow(2, 2, 1)
        .reshape([batch_size, max_peaks]);
    let mut input_peak_mask = peak_mask;
    let mut input_padding_mask = padding_mask;
    let original_active = input_peak_mask.clone().greater_elem(0.0);
    let empty_original_slots = original_active.clone().bool_not();
    let real_peak_present = original_active
        .clone()
        .float()
        .sum_dim(1)
        .greater_elem(0.0)
        .expand([batch_size, max_peaks]);
    let source_min_mz = mz
        .clone()
        .mask_fill(original_active.clone().bool_not(), 1.0)
        .min_dim(1)
        .expand([batch_size, max_peaks]);
    let source_max_mz = mz
        .clone()
        .mask_fill(original_active.clone().bool_not(), 0.0)
        .max_dim(1)
        .expand([batch_size, max_peaks]);
    let source_min_intensity = intensity
        .clone()
        .mask_fill(original_active.clone().bool_not(), 1.0)
        .min_dim(1)
        .expand([batch_size, max_peaks]);
    let source_max_intensity = intensity
        .clone()
        .mask_fill(original_active.clone().bool_not(), 0.0)
        .max_dim(1)
        .expand([batch_size, max_peaks]);
    let mut active = original_active.clone();
    let mut masked_peak_mask =
        Tensor::<B, 2>::zeros([batch_size, max_peaks], &device).greater_elem(0.0);

    if let Some(dropout_mask) = probability_mask(
        [batch_size, max_peaks],
        &device,
        config.peak_dropout_probability,
    ) {
        let dropout_mask = dropout_mask.bool_and(active.clone());
        active = active.bool_and(dropout_mask.clone().bool_not());
        masked_peak_mask = masked_peak_mask.bool_or(dropout_mask.clone());
        mz = mz.mask_fill(dropout_mask.clone(), 0.0);
        intensity = intensity.mask_fill(dropout_mask.clone(), 0.0);
        presence = presence.mask_fill(dropout_mask.clone(), 0.0);
        input_peak_mask = input_peak_mask.mask_fill(dropout_mask.clone(), 0.0);
        input_padding_mask = input_padding_mask.bool_or(dropout_mask);
    }

    let mut mz_mask = Tensor::<B, 2>::zeros([batch_size, max_peaks], &device).greater_elem(0.0);
    if let Some(mask) =
        probability_mask([batch_size, max_peaks], &device, config.mz_mask_probability)
    {
        mz_mask = mask.bool_and(active.clone());
        masked_peak_mask = masked_peak_mask.bool_or(mz_mask.clone());
        mz = mz.mask_fill(mz_mask.clone(), 0.0);
    }

    let mz_editable = active.clone().bool_and(mz_mask.clone().bool_not());
    let mz_shift_range = config.mz_shift_range.max(0.0);
    let mz_jitter_range = config.mz_jitter_range.max(0.0);
    if mz_shift_range > 0.0 || mz_jitter_range > 0.0 {
        let shift =
            signed_random([batch_size, 1], &device, mz_shift_range).expand([batch_size, max_peaks]);
        let jitter = signed_random([batch_size, max_peaks], &device, mz_jitter_range);
        let shifted = (mz.clone() + shift + jitter).clamp_min(0.0).clamp_max(1.0);
        mz = mz.mask_where(mz_editable, shifted);
    }

    let intensity_jitter = config.intensity_jitter_fraction.max(0.0);
    if intensity_jitter > 0.0 {
        let scale = (signed_random([batch_size, max_peaks], &device, intensity_jitter) + 1.0)
            .clamp_min(0.0);
        let jittered = (intensity.clone() * scale).clamp_max(1.0);
        intensity = intensity.mask_where(active.clone(), jittered);
    }

    let mut intruder_peak_mask =
        Tensor::<B, 2>::zeros([batch_size, max_peaks], &device).greater_elem(0.0);
    if let Some(mask) = probability_mask(
        [batch_size, max_peaks],
        &device,
        config.intruder_peak_probability,
    ) {
        intruder_peak_mask = mask
            .bool_and(empty_original_slots)
            .bool_and(real_peak_present);
        let random_mz = Tensor::<B, 2>::random(
            [batch_size, max_peaks],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let random_intensity = Tensor::<B, 2>::random(
            [batch_size, max_peaks],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );
        let intruder_mz = (source_min_mz.clone()
            + random_mz * (source_max_mz - source_min_mz.clone()))
        .clamp_min(0.0)
        .clamp_max(1.0);
        let intruder_intensity = (source_min_intensity.clone()
            + random_intensity * (source_max_intensity - source_min_intensity.clone()))
        .clamp_min(0.0)
        .clamp_max(1.0);
        mz = mz.mask_where(intruder_peak_mask.clone(), intruder_mz);
        intensity = intensity.mask_where(intruder_peak_mask.clone(), intruder_intensity);
        presence = presence.mask_fill(intruder_peak_mask.clone(), 1.0);
        input_peak_mask = input_peak_mask.mask_fill(intruder_peak_mask.clone(), 1.0);
        input_padding_mask = input_padding_mask.bool_and(intruder_peak_mask.clone().bool_not());
        active = active.bool_or(intruder_peak_mask.clone());
    }

    let mut feature_parts = vec![
        mz.clone().unsqueeze_dim::<3>(2),
        intensity.unsqueeze_dim::<3>(2),
        presence.unsqueeze_dim::<3>(2),
    ];
    let token_features = if feature_width > 3 {
        let fourier = fourier_features(mz, feature_width - 3, active.bool_and(mz_mask.bool_not()));
        feature_parts.push(fourier);
        Tensor::cat(feature_parts, 2)
    } else {
        Tensor::cat(feature_parts, 2)
    };

    let (conditions, masked_precursor_mask) =
        mask_precursor_conditions(conditions, config.precursor_mask_probability);

    (
        token_features,
        input_peak_mask,
        input_padding_mask,
        conditions,
        masked_precursor_mask,
        masked_peak_mask.float(),
        intruder_peak_mask.float(),
    )
}

fn fourier_features<B: Backend>(
    mz: Tensor<B, 2>,
    feature_count: usize,
    active: Tensor<B, 2, Bool>,
) -> Tensor<B, 3> {
    let [batch_size, max_peaks] = mz.dims();
    let mut features = Vec::with_capacity(feature_count);
    let fourier_pairs = feature_count / 2;
    let mut frequency = 1.0_f32;
    for _ in 0..fourier_pairs {
        let angle = mz.clone() * (std::f32::consts::TAU * frequency);
        features.push(angle.clone().sin().unsqueeze_dim::<3>(2));
        features.push(angle.cos().unsqueeze_dim::<3>(2));
        frequency *= 2.0;
    }
    if feature_count % 2 == 1 {
        features.push(
            Tensor::<B, 2>::zeros([batch_size, max_peaks], &mz.device()).unsqueeze_dim::<3>(2),
        );
    }

    let active = active
        .unsqueeze_dim::<3>(2)
        .expand([batch_size, max_peaks, feature_count]);
    Tensor::cat(features, 2).mask_fill(active.bool_not(), 0.0)
}

struct StreamingTokenizedMgfLoader<B, Open>
where
    B: Backend,
{
    mgf_source: String,
    progress: LoaderProgress,
    batch_size: usize,
    max_batches: usize,
    start_item: usize,
    window_items: usize,
    device: B::Device,
    augment: Option<SpectrumAugmentationConfig>,
    similarity_teacher: SimilarityTeacherConfig,
    open_records: Open,
    pair_sampling_epoch: PairSamplingEpoch,
}

impl<B, Open> Clone for StreamingTokenizedMgfLoader<B, Open>
where
    B: Backend,
    B::Device: Clone,
    Open: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mgf_source: self.mgf_source.clone(),
            progress: self.progress.clone(),
            batch_size: self.batch_size,
            max_batches: self.max_batches,
            start_item: self.start_item,
            window_items: self.window_items,
            device: self.device.clone(),
            augment: self.augment,
            similarity_teacher: self.similarity_teacher,
            open_records: self.open_records.clone(),
            pair_sampling_epoch: self.pair_sampling_epoch.clone(),
        }
    }
}

impl<B, Open> StreamingTokenizedMgfLoader<B, Open>
where
    B: Backend,
{
    fn new(options: StreamingLoaderOptions<B>, open_records: Open) -> Self {
        Self {
            mgf_source: options.mgf_source,
            progress: options.progress,
            batch_size: options.batch_size,
            max_batches: options.max_batches,
            start_item: options.start_item,
            window_items: options.window_items,
            device: options.device,
            augment: options.augment,
            similarity_teacher: options.similarity_teacher,
            open_records,
            pair_sampling_epoch: PairSamplingEpoch::new(options.randomize_pair_sampling),
        }
    }
}

impl<B, Open> DataLoader<B, TokenizedAutoencoderBatch<B>> for StreamingTokenizedMgfLoader<B, Open>
where
    B: SimilarityTeacherBackend + 'static,
    Open: Fn() -> spectral_autoencoder::Result<TokenizedMgfIter> + Send + Sync + Clone + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<TokenizedAutoencoderBatch<B>> + 'a> {
        self.progress
            .start_epoch(loader_epoch_items(self.batch_size, self.max_batches));
        let epoch_index = self.pair_sampling_epoch.next();

        let mut records = open_records_or_panic(&self.open_records, "tokenized", &self.mgf_source);
        skip_split_records(&mut records, self.start_item, &self.progress, None);

        Box::new(StreamingTokenizedMgfIter {
            loader: self,
            records: Some(records),
            cache: None,
            cache_offset: 0,
            batches_processed: 0,
            items_processed: 0,
            epoch_index,
            finished: false,
        })
    }

    fn num_items(&self) -> usize {
        loader_epoch_items(self.batch_size, self.max_batches)
    }

    fn to_device(
        &self,
        device: &B::Device,
    ) -> Arc<dyn DataLoader<B, TokenizedAutoencoderBatch<B>>> {
        Arc::new(Self {
            device: device.clone(),
            ..self.clone()
        })
    }

    fn slice(
        &self,
        start: usize,
        end: usize,
    ) -> Arc<dyn DataLoader<B, TokenizedAutoencoderBatch<B>>> {
        let start_batch = start / self.batch_size;
        let end_batch = end.div_ceil(self.batch_size);
        Arc::new(Self {
            max_batches: end_batch.saturating_sub(start_batch),
            start_item: self.start_item + start_batch * self.batch_size,
            progress: self.progress.clone_for_slice(),
            ..self.clone()
        })
    }
}

#[derive(Clone)]
struct TokenGpuCache<B: Backend> {
    token_features: Tensor<B, 3>,
    target_pairs: Tensor<B, 2>,
    peak_mask: Tensor<B, 2>,
    target_peak_mask: Tensor<B, 2>,
    padding_mask: Tensor<B, 2, Bool>,
    conditions: Tensor<B, 2>,
    teacher_gpu: Option<TeacherGpuCache<B>>,
    items: usize,
}

struct StreamingTokenizedMgfIter<'a, B, Open>
where
    B: Backend,
{
    loader: &'a StreamingTokenizedMgfLoader<B, Open>,
    records: Option<TokenizedMgfIter>,
    cache: Option<TokenGpuCache<B>>,
    cache_offset: usize,
    batches_processed: usize,
    items_processed: usize,
    epoch_index: u64,
    finished: bool,
}

impl<B, Open> Iterator for StreamingTokenizedMgfIter<'_, B, Open>
where
    B: SimilarityTeacherBackend,
{
    type Item = TokenizedAutoencoderBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.batches_processed >= self.loader.max_batches {
            self.finish();
            return None;
        }

        if self
            .cache
            .as_ref()
            .is_none_or(|cache| self.cache_offset >= cache.items)
        {
            self.cache = self.fill_cache();
            self.cache_offset = 0;
        }

        let cache = match &self.cache {
            Some(cache) => cache,
            None => {
                self.finish();
                return None;
            }
        };
        let batch_items = self
            .loader
            .batch_size
            .min(cache.items.saturating_sub(self.cache_offset));
        if batch_items == 0 {
            self.finish();
            return None;
        }

        let token_features = cache
            .token_features
            .clone()
            .narrow(0, self.cache_offset, batch_items);
        let target_pairs = cache
            .target_pairs
            .clone()
            .narrow(0, self.cache_offset, batch_items);
        let peak_mask = cache
            .peak_mask
            .clone()
            .narrow(0, self.cache_offset, batch_items);
        let target_peak_mask =
            cache
                .target_peak_mask
                .clone()
                .narrow(0, self.cache_offset, batch_items);
        let padding_mask = cache
            .padding_mask
            .clone()
            .narrow(0, self.cache_offset, batch_items);
        let conditions = cache
            .conditions
            .clone()
            .narrow(0, self.cache_offset, batch_items);
        let target_conditions = conditions.clone();

        let (
            token_features,
            peak_mask,
            padding_mask,
            conditions,
            masked_precursor_mask,
            masked_peak_mask,
            intruder_peak_mask,
        ) = match self.loader.augment {
            Some(config) => {
                let clean_token_features = token_features.clone();
                let clean_peak_mask = peak_mask.clone();
                let clean_padding_mask = padding_mask.clone();
                let (
                    token_features,
                    peak_mask,
                    padding_mask,
                    conditions,
                    masked_precursor_mask,
                    masked_peak_mask,
                    intruder_peak_mask,
                ) = augment_token_batch(
                    clean_token_features.clone(),
                    clean_peak_mask.clone(),
                    clean_padding_mask.clone(),
                    conditions.clone(),
                    config,
                );
                (
                    token_features,
                    peak_mask,
                    padding_mask,
                    conditions,
                    masked_precursor_mask,
                    masked_peak_mask,
                    intruder_peak_mask,
                )
            }
            None => {
                let device = token_features.device();
                let [batch_size, max_peaks, _feature_width] = token_features.dims();
                (
                    token_features.clone(),
                    peak_mask.clone(),
                    padding_mask.clone(),
                    conditions.clone(),
                    Tensor::<B, 2>::zeros([batch_size, 1], &device),
                    Tensor::<B, 2>::zeros([batch_size, max_peaks], &device),
                    Tensor::<B, 2>::zeros([batch_size, max_peaks], &device),
                )
            }
        };
        let similarity_ranking = teacher_similarity_ranking_batch(
            cache.teacher_gpu.as_ref(),
            self.cache_offset,
            batch_items,
            similarity_pair_seed(
                self.epoch_index,
                self.batches_processed,
                self.items_processed,
            ),
            self.loader.similarity_teacher,
            &self.loader.device,
        );

        self.cache_offset += batch_items;
        self.batches_processed += 1;
        self.items_processed += batch_items;
        self.loader.progress.batch_loaded(
            batch_items,
            self.batches_processed,
            self.loader.max_batches,
        );

        Some(TokenizedAutoencoderBatch {
            token_features,
            target_pairs,
            peak_mask,
            target_peak_mask,
            padding_mask,
            conditions,
            target_conditions,
            masked_precursor_mask,
            masked_peak_mask,
            intruder_peak_mask,
            similarity_ranking,
        })
    }
}

impl<B, Open> StreamingTokenizedMgfIter<'_, B, Open>
where
    B: SimilarityTeacherBackend,
{
    fn fill_cache(&mut self) -> Option<TokenGpuCache<B>> {
        let records = self.records.as_mut()?;
        let remaining_items = loader_epoch_items(self.loader.batch_size, self.loader.max_batches)
            .saturating_sub(self.items_processed);
        let target_items = self.loader.window_items.min(remaining_items);
        if target_items == 0 {
            return None;
        }

        let mut token_features = Vec::new();
        let mut target_pairs = Vec::new();
        let mut peak_mask = Vec::new();
        let mut target_peak_mask = Vec::new();
        let mut padding_mask = Vec::new();
        let mut conditions = Vec::new();
        let mut teacher_builder = self
            .loader
            .similarity_teacher
            .enabled()
            .then(|| TeacherSpectraBuilder::new(self.loader.similarity_teacher, target_items));
        let mut max_peaks = 0usize;
        let mut token_feature_width = 0usize;
        let mut target_width = 0usize;
        let mut condition_width = 0usize;
        let mut items = 0usize;

        while items < target_items {
            let sample = match records.next() {
                Some(Ok(sample)) => sample,
                Some(Err(error)) => {
                    panic!("failed to tokenize MGF record: {error}");
                }
                None => break,
            };

            if items == 0 {
                max_peaks = sample.peak_mask.len();
                token_feature_width = sample.token_features.len() / max_peaks;
                target_width = sample.target_pairs.len();
                condition_width = sample.conditions.len();
                token_features.reserve(target_items * max_peaks * token_feature_width);
                target_pairs.reserve(target_items * target_width);
                peak_mask.reserve(target_items * max_peaks);
                target_peak_mask.reserve(target_items * max_peaks);
                padding_mask.reserve(target_items * max_peaks);
                conditions.reserve(target_items * condition_width);
            }
            if let Some(builder) = &mut teacher_builder {
                builder.push_pairs(&sample.target_pairs, &sample.conditions);
            }
            token_features.extend(sample.token_features);
            target_pairs.extend(sample.target_pairs);
            peak_mask.extend_from_slice(&sample.peak_mask);
            target_peak_mask.extend(sample.peak_mask);
            padding_mask.extend(sample.padding_mask);
            conditions.extend(sample.conditions);
            items += 1;
            if items.is_multiple_of(self.loader.batch_size) || items == target_items {
                self.loader.progress.cache_filling(
                    items,
                    target_items,
                    self.batches_processed,
                    self.loader.max_batches,
                );
            }
        }

        if items == 0 {
            return None;
        }

        let teacher = teacher_builder.map(|builder| Arc::new(builder.finish()));
        let teacher_gpu = self
            .loader
            .similarity_teacher
            .enabled()
            .then(|| {
                teacher.as_ref().and_then(|teacher| {
                    TeacherGpuCache::from_cpu_window(teacher, 0, items, &self.loader.device)
                })
            })
            .flatten();

        let cache = TokenGpuCache {
            token_features: Tensor::<B, 3>::from_data(
                TensorData::new(token_features, [items, max_peaks, token_feature_width]),
                &self.loader.device,
            ),
            target_pairs: Tensor::<B, 2>::from_data(
                TensorData::new(target_pairs, [items, target_width]),
                &self.loader.device,
            ),
            peak_mask: Tensor::<B, 2>::from_data(
                TensorData::new(peak_mask, [items, max_peaks]),
                &self.loader.device,
            ),
            target_peak_mask: Tensor::<B, 2>::from_data(
                TensorData::new(target_peak_mask, [items, max_peaks]),
                &self.loader.device,
            ),
            padding_mask: Tensor::<B, 2, Bool>::from_bool(
                TensorData::new(padding_mask, [items, max_peaks]),
                &self.loader.device,
            ),
            conditions: Tensor::<B, 2>::from_data(
                TensorData::new(conditions, [items, condition_width]),
                &self.loader.device,
            ),
            teacher_gpu,
            items,
        };
        Some(cache)
    }

    fn finish(&mut self) {
        finish_loader_once(
            &self.loader.progress,
            self.items_processed,
            self.batches_processed,
            &mut self.finished,
        );
    }
}

impl<B, Open> DataLoaderIterator<TokenizedAutoencoderBatch<B>>
    for StreamingTokenizedMgfIter<'_, B, Open>
where
    B: SimilarityTeacherBackend,
{
    fn progress(&self) -> Progress {
        Progress {
            items_processed: self.items_processed,
            items_total: loader_epoch_items(self.loader.batch_size, self.loader.max_batches),
        }
    }
}
