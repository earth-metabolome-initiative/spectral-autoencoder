//! Burn training helpers.

use burn::{
    backend::NdArray,
    prelude::*,
    tensor::Transaction,
    train::{
        InferenceStep, LearningComponentsTypes, SupervisedTraining, TrainStep,
        metric::{
            Adaptor, ItemLazy, Metric, MetricAttributes, MetricMetadata, MetricName, Numeric,
            NumericAttributes, NumericEntry, SerializedEntry,
            state::{FormatOptions, NumericMetricState},
        },
    },
};

use std::sync::Arc;

/// Weighted loss components reported to the TUI.
pub struct AutoencoderLossBreakdown<B: Backend> {
    /// Reconstruction contribution to the total loss.
    pub reconstruction: Tensor<B, 1>,
    /// Masked-peak contribution to the total loss.
    pub masked: Tensor<B, 1>,
    /// Synthetic intruder-peak detection contribution to the total loss.
    pub intruder: Tensor<B, 1>,
    /// Precursor m/z reconstruction contribution to the total loss.
    pub precursor: Tensor<B, 1>,
    /// Masked precursor m/z reconstruction contribution to the total loss.
    pub masked_precursor: Tensor<B, 1>,
    /// In-batch clean-spectrum similarity-ranking contribution to the total loss.
    pub similarity_ranking: Tensor<B, 1>,
    /// Optional explicit model-parameter regularization contribution.
    pub regularization: Tensor<B, 1>,
    /// Chamfer-style m/z magnet contribution (flat-vector only).
    pub chamfer_mz: Tensor<B, 1>,
}

impl<B: Backend> AutoencoderLossBreakdown<B> {
    /// Returns a zero-valued breakdown.
    pub fn zeros(device: &B::Device) -> Self {
        Self {
            reconstruction: Tensor::zeros([1], device),
            masked: Tensor::zeros([1], device),
            intruder: Tensor::zeros([1], device),
            precursor: Tensor::zeros([1], device),
            masked_precursor: Tensor::zeros([1], device),
            similarity_ranking: Tensor::zeros([1], device),
            regularization: Tensor::zeros([1], device),
            chamfer_mz: Tensor::zeros([1], device),
        }
    }

    /// Returns the weighted total loss.
    pub fn total(&self) -> Tensor<B, 1> {
        self.reconstruction.clone()
            + self.masked.clone()
            + self.intruder.clone()
            + self.precursor.clone()
            + self.masked_precursor.clone()
            + self.similarity_ranking.clone()
            + self.regularization.clone()
            + self.chamfer_mz.clone()
    }
}

/// Non-loss diagnostics reported to the TUI.
pub struct AutoencoderDiagnostics<B: Backend> {
    /// Number of valid similarity-ranking pairs contributing to the batch.
    pub similarity_ranking_pairs: Tensor<B, 1>,
    /// Latent-vs-target ordering accuracy over valid similarity-ranking pairs.
    pub similarity_ranking_accuracy: Tensor<B, 1>,
    /// Mean absolute precursor m/z reconstruction error in Da.
    pub precursor_mae_da: Tensor<B, 1>,
    /// Linear-cosine similarity between target and reconstructed spectra.
    pub self_linear_cosine: Tensor<B, 1>,
    /// Modified-linear-cosine similarity between target and reconstructed spectra.
    pub self_modified_linear_cosine: Tensor<B, 1>,
    /// Number of spectra sampled for self-similarity diagnostics.
    pub self_similarity_items: Tensor<B, 1>,
}

impl<B: Backend> AutoencoderDiagnostics<B> {
    /// Returns zero-valued diagnostics.
    pub fn zeros(device: &B::Device) -> Self {
        Self {
            similarity_ranking_pairs: Tensor::zeros([1], device),
            similarity_ranking_accuracy: Tensor::zeros([1], device),
            precursor_mae_da: Tensor::zeros([1], device),
            self_linear_cosine: Tensor::zeros([1], device),
            self_modified_linear_cosine: Tensor::zeros([1], device),
            self_similarity_items: Tensor::zeros([1], device),
        }
    }
}

/// Autoencoder output adapted for Burn metrics.
pub struct AutoencoderTrainingOutput<B: Backend> {
    /// The total loss.
    pub loss: Tensor<B, 1>,
    /// The flattened model output.
    pub output: Tensor<B, 2>,
    /// The flattened targets.
    pub targets: Tensor<B, 2>,
    /// Weighted component losses.
    pub losses: AutoencoderLossBreakdown<B>,
    /// Non-loss diagnostics.
    pub diagnostics: AutoencoderDiagnostics<B>,
}

impl<B: Backend> AutoencoderTrainingOutput<B> {
    /// Creates a new training output from component losses.
    pub fn new(
        output: Tensor<B, 2>,
        targets: Tensor<B, 2>,
        losses: AutoencoderLossBreakdown<B>,
    ) -> Self {
        let device = losses.reconstruction.device();
        Self::new_with_diagnostics(
            output,
            targets,
            losses,
            AutoencoderDiagnostics::zeros(&device),
        )
    }

    /// Creates a new training output from losses and diagnostics.
    pub fn new_with_diagnostics(
        output: Tensor<B, 2>,
        targets: Tensor<B, 2>,
        losses: AutoencoderLossBreakdown<B>,
        diagnostics: AutoencoderDiagnostics<B>,
    ) -> Self {
        let loss = losses.total();
        Self {
            loss,
            output,
            targets,
            losses,
            diagnostics,
        }
    }
}

impl<B: Backend> Adaptor<AutoencoderLossComponentsInput<B>> for AutoencoderTrainingOutput<B> {
    fn adapt(&self) -> AutoencoderLossComponentsInput<B> {
        AutoencoderLossComponentsInput {
            loss: self.loss.clone(),
            reconstruction: self.losses.reconstruction.clone(),
            masked: self.losses.masked.clone(),
            intruder: self.losses.intruder.clone(),
            precursor: self.losses.precursor.clone(),
            masked_precursor: self.losses.masked_precursor.clone(),
            similarity_ranking: self.losses.similarity_ranking.clone(),
            chamfer_mz: self.losses.chamfer_mz.clone(),
            similarity_ranking_pairs: self.diagnostics.similarity_ranking_pairs.clone(),
            similarity_ranking_accuracy: self.diagnostics.similarity_ranking_accuracy.clone(),
            precursor_mae_da: self.diagnostics.precursor_mae_da.clone(),
            self_linear_cosine: self.diagnostics.self_linear_cosine.clone(),
            self_modified_linear_cosine: self.diagnostics.self_modified_linear_cosine.clone(),
            self_similarity_items: self.diagnostics.self_similarity_items.clone(),
        }
    }
}

impl<B: Backend> ItemLazy for AutoencoderTrainingOutput<B> {
    type ItemSync = AutoencoderTrainingOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [
            loss,
            reconstruction,
            masked,
            intruder,
            precursor,
            masked_precursor,
            similarity_ranking,
            regularization,
            chamfer_mz,
            similarity_ranking_pairs,
            similarity_ranking_accuracy,
            precursor_mae_da,
            self_linear_cosine,
            self_modified_linear_cosine,
            self_similarity_items,
        ] = Transaction::default()
            .register(self.loss)
            .register(self.losses.reconstruction)
            .register(self.losses.masked)
            .register(self.losses.intruder)
            .register(self.losses.precursor)
            .register(self.losses.masked_precursor)
            .register(self.losses.similarity_ranking)
            .register(self.losses.regularization)
            .register(self.losses.chamfer_mz)
            .register(self.diagnostics.similarity_ranking_pairs)
            .register(self.diagnostics.similarity_ranking_accuracy)
            .register(self.diagnostics.precursor_mae_da)
            .register(self.diagnostics.self_linear_cosine)
            .register(self.diagnostics.self_modified_linear_cosine)
            .register(self.diagnostics.self_similarity_items)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();
        AutoencoderTrainingOutput {
            loss: Tensor::from_data(loss, device),
            output: Tensor::zeros([1, 1], device),
            targets: Tensor::zeros([1, 1], device),
            losses: AutoencoderLossBreakdown {
                reconstruction: Tensor::from_data(reconstruction, device),
                masked: Tensor::from_data(masked, device),
                intruder: Tensor::from_data(intruder, device),
                precursor: Tensor::from_data(precursor, device),
                masked_precursor: Tensor::from_data(masked_precursor, device),
                similarity_ranking: Tensor::from_data(similarity_ranking, device),
                regularization: Tensor::from_data(regularization, device),
                chamfer_mz: Tensor::from_data(chamfer_mz, device),
            },
            diagnostics: AutoencoderDiagnostics {
                similarity_ranking_pairs: Tensor::from_data(similarity_ranking_pairs, device),
                similarity_ranking_accuracy: Tensor::from_data(similarity_ranking_accuracy, device),
                precursor_mae_da: Tensor::from_data(precursor_mae_da, device),
                self_linear_cosine: Tensor::from_data(self_linear_cosine, device),
                self_modified_linear_cosine: Tensor::from_data(self_modified_linear_cosine, device),
                self_similarity_items: Tensor::from_data(self_similarity_items, device),
            },
        }
    }
}

/// Metric input containing reported autoencoder loss components and diagnostics.
pub struct AutoencoderLossComponentsInput<B: Backend> {
    loss: Tensor<B, 1>,
    reconstruction: Tensor<B, 1>,
    masked: Tensor<B, 1>,
    intruder: Tensor<B, 1>,
    precursor: Tensor<B, 1>,
    masked_precursor: Tensor<B, 1>,
    similarity_ranking: Tensor<B, 1>,
    chamfer_mz: Tensor<B, 1>,
    similarity_ranking_pairs: Tensor<B, 1>,
    similarity_ranking_accuracy: Tensor<B, 1>,
    precursor_mae_da: Tensor<B, 1>,
    self_linear_cosine: Tensor<B, 1>,
    self_modified_linear_cosine: Tensor<B, 1>,
    self_similarity_items: Tensor<B, 1>,
}

/// A numeric TUI metric for one autoencoder loss component or diagnostic.
#[derive(Clone)]
pub struct AutoencoderLossComponentMetric<B: Backend> {
    component: AutoencoderLossComponent,
    name: Arc<String>,
    state: NumericMetricState,
    _backend: B,
}

impl<B: Backend> AutoencoderLossComponentMetric<B> {
    /// Total weighted loss.
    pub fn loss() -> Self {
        Self::new(AutoencoderLossComponent::Loss)
    }

    /// Reconstruction loss contribution.
    pub fn reconstruction() -> Self {
        Self::new(AutoencoderLossComponent::Reconstruction)
    }

    /// Masked-peak loss contribution.
    pub fn masked() -> Self {
        Self::new(AutoencoderLossComponent::Masked)
    }

    /// Synthetic intruder-peak detection loss contribution.
    pub fn intruder() -> Self {
        Self::new(AutoencoderLossComponent::Intruder)
    }

    /// Precursor m/z reconstruction loss contribution.
    pub fn precursor() -> Self {
        Self::new(AutoencoderLossComponent::Precursor)
    }

    /// Masked precursor m/z reconstruction loss contribution.
    pub fn masked_precursor() -> Self {
        Self::new(AutoencoderLossComponent::MaskedPrecursor)
    }

    /// Mean absolute precursor reconstruction error in Da.
    pub fn precursor_mae_da() -> Self {
        Self::new(AutoencoderLossComponent::PrecursorMaeDa)
    }

    /// In-batch clean-spectrum similarity-ranking loss contribution.
    pub fn similarity_ranking() -> Self {
        Self::new(AutoencoderLossComponent::SimilarityRanking)
    }

    /// Similarity-ranking latent-vs-target ordering accuracy.
    pub fn similarity_ranking_accuracy() -> Self {
        Self::new(AutoencoderLossComponent::SimilarityRankingAccuracy)
    }

    /// Target-vs-reconstruction linear-cosine similarity.
    pub fn self_linear_cosine() -> Self {
        Self::new(AutoencoderLossComponent::SelfLinearCosine)
    }

    /// Target-vs-reconstruction modified-linear-cosine similarity.
    pub fn self_modified_linear_cosine() -> Self {
        Self::new(AutoencoderLossComponent::SelfModifiedLinearCosine)
    }

    /// Chamfer-style m/z magnet loss contribution (flat-vector only).
    pub fn chamfer_mz() -> Self {
        Self::new(AutoencoderLossComponent::ChamferMz)
    }

    fn new(component: AutoencoderLossComponent) -> Self {
        Self {
            component,
            name: Arc::new(component.name().to_string()),
            state: NumericMetricState::default(),
            _backend: Default::default(),
        }
    }
}

impl<B: Backend> Metric for AutoencoderLossComponentMetric<B> {
    type Input = AutoencoderLossComponentsInput<B>;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn description(&self) -> Option<String> {
        Some(format!(
            "Autoencoder {} metric.",
            self.component.description()
        ))
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: self.component.higher_is_better(),
        }
        .into()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let tensor = match self.component {
            AutoencoderLossComponent::Loss => item.loss.clone(),
            AutoencoderLossComponent::Reconstruction => item.reconstruction.clone(),
            AutoencoderLossComponent::Masked => item.masked.clone(),
            AutoencoderLossComponent::Intruder => item.intruder.clone(),
            AutoencoderLossComponent::Precursor => item.precursor.clone(),
            AutoencoderLossComponent::MaskedPrecursor => item.masked_precursor.clone(),
            AutoencoderLossComponent::PrecursorMaeDa => item.precursor_mae_da.clone(),
            AutoencoderLossComponent::SimilarityRanking => item.similarity_ranking.clone(),
            AutoencoderLossComponent::SimilarityRankingAccuracy => {
                item.similarity_ranking_accuracy.clone()
            }
            AutoencoderLossComponent::SelfLinearCosine => item.self_linear_cosine.clone(),
            AutoencoderLossComponent::SelfModifiedLinearCosine => {
                item.self_modified_linear_cosine.clone()
            }
            AutoencoderLossComponent::ChamferMz => item.chamfer_mz.clone(),
        };
        let value = tensor
            .mean()
            .into_data()
            .iter::<f64>()
            .next()
            .expect("component loss should contain one value");
        let batch_size = self.component.batch_size(item);

        self.state.update(
            value,
            batch_size,
            FormatOptions::new(self.name()).precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for AutoencoderLossComponentMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

#[derive(Debug, Clone, Copy)]
enum AutoencoderLossComponent {
    Loss,
    Reconstruction,
    Masked,
    Intruder,
    Precursor,
    MaskedPrecursor,
    PrecursorMaeDa,
    SimilarityRanking,
    SimilarityRankingAccuracy,
    SelfLinearCosine,
    SelfModifiedLinearCosine,
    ChamferMz,
}

impl AutoencoderLossComponent {
    const fn name(self) -> &'static str {
        match self {
            Self::Loss => "Loss",
            Self::Reconstruction => "Reconstruction Loss",
            Self::Masked => "Masked Loss",
            Self::Intruder => "Intruder Loss",
            Self::Precursor => "Precursor Loss",
            Self::MaskedPrecursor => "Masked Precursor Loss",
            Self::PrecursorMaeDa => "Precursor MAE Da",
            Self::SimilarityRanking => "Similarity Ranking Loss",
            Self::SimilarityRankingAccuracy => "Similarity Ranking Accuracy",
            Self::SelfLinearCosine => "Self Linear Cosine",
            Self::SelfModifiedLinearCosine => "Self Modified Linear Cosine",
            Self::ChamferMz => "Chamfer mz",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Loss => "total weighted loss",
            Self::Reconstruction => "reconstruction",
            Self::Masked => "masked-peak",
            Self::Intruder => "intruder-peak detection",
            Self::Precursor => "precursor reconstruction",
            Self::MaskedPrecursor => "masked precursor reconstruction",
            Self::PrecursorMaeDa => "precursor reconstruction mean absolute error in Da",
            Self::SimilarityRanking => "teacher similarity ranking",
            Self::SimilarityRankingAccuracy => "teacher similarity-ranking ordering accuracy",
            Self::SelfLinearCosine => "target-vs-reconstruction linear cosine",
            Self::SelfModifiedLinearCosine => "target-vs-reconstruction modified linear cosine",
            Self::ChamferMz => "Chamfer-style m/z magnet pulling pred toward nearest real target",
        }
    }

    const fn higher_is_better(self) -> bool {
        matches!(
            self,
            Self::SimilarityRankingAccuracy
                | Self::SelfLinearCosine
                | Self::SelfModifiedLinearCosine
        )
    }

    fn batch_size<B: Backend>(self, item: &AutoencoderLossComponentsInput<B>) -> usize {
        match self {
            Self::SimilarityRankingAccuracy => metric_scalar(&item.similarity_ranking_pairs)
                .max(1.0)
                .round() as usize,
            Self::SelfLinearCosine | Self::SelfModifiedLinearCosine => {
                metric_scalar(&item.self_similarity_items).max(1.0).round() as usize
            }
            _ => 1,
        }
    }
}

fn metric_scalar<B: Backend>(tensor: &Tensor<B, 1>) -> f64 {
    tensor
        .clone()
        .mean()
        .into_data()
        .iter::<f64>()
        .next()
        .expect("metric tensor should contain one value")
}

/// Metric registration preset for autoencoder training.
///
/// Burn's TUI only displays metrics that are registered on the training
/// runner. This extension keeps the crate's default metric setup in one place
/// while leaving the actual training loop to applications.
pub trait AutoencoderTrainingMetricsExt<LC>
where
    LC: LearningComponentsTypes,
{
    /// Registers the default autoencoder metrics.
    ///
    /// This includes train/validation loss and task diagnostics.
    fn with_autoencoder_metrics(self) -> Self;

    /// Registers reconstruction metrics that apply to both CPU and GPU builds.
    fn with_reconstruction_metrics(self) -> Self;
}

impl<LC> AutoencoderTrainingMetricsExt<LC> for SupervisedTraining<LC>
where
    LC: LearningComponentsTypes,
    <<<LC as LearningComponentsTypes>::TrainingModel as TrainStep>::Output as ItemLazy>::ItemSync:
        Adaptor<AutoencoderLossComponentsInput<NdArray>> + Adaptor<()>,
    <<<LC as LearningComponentsTypes>::InferenceModel as InferenceStep>::Output as ItemLazy>::ItemSync:
        Adaptor<AutoencoderLossComponentsInput<NdArray>>,
{
    fn with_autoencoder_metrics(self) -> Self {
        self.with_reconstruction_metrics()
    }

    fn with_reconstruction_metrics(self) -> Self {
        self.metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::loss())
            .metric_valid_numeric(AutoencoderLossComponentMetric::<NdArray>::loss())
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::reconstruction())
            .metric_valid_numeric(AutoencoderLossComponentMetric::<NdArray>::reconstruction())
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::chamfer_mz())
            .metric_valid_numeric(AutoencoderLossComponentMetric::<NdArray>::chamfer_mz())
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::masked())
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::intruder())
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::precursor())
            .metric_valid_numeric(AutoencoderLossComponentMetric::<NdArray>::precursor())
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::masked_precursor())
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::precursor_mae_da())
            .metric_valid_numeric(AutoencoderLossComponentMetric::<NdArray>::precursor_mae_da())
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::similarity_ranking())
            .metric_valid_numeric(AutoencoderLossComponentMetric::<NdArray>::similarity_ranking())
            .metric_train_numeric(
                AutoencoderLossComponentMetric::<NdArray>::similarity_ranking_accuracy(),
            )
            .metric_valid_numeric(
                AutoencoderLossComponentMetric::<NdArray>::similarity_ranking_accuracy(),
            )
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::self_linear_cosine())
            .metric_valid_numeric(AutoencoderLossComponentMetric::<NdArray>::self_linear_cosine())
            .metric_train_numeric(
                AutoencoderLossComponentMetric::<NdArray>::self_modified_linear_cosine(),
            )
            .metric_valid_numeric(
                AutoencoderLossComponentMetric::<NdArray>::self_modified_linear_cosine(),
            )
    }
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;

    use burn::data::dataloader::Progress;

    type TestBackend = burn::backend::NdArray<f32, i64>;
    type TestDevice = burn::backend::ndarray::NdArrayDevice;

    fn device() -> TestDevice {
        TestDevice::default()
    }

    fn scalar(value: f32, device: &TestDevice) -> Tensor<TestBackend, 1> {
        Tensor::<TestBackend, 1>::from_floats([value], device)
    }

    fn matrix(device: &TestDevice) -> Tensor<TestBackend, 2> {
        Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], device)
    }

    fn losses(device: &TestDevice) -> AutoencoderLossBreakdown<TestBackend> {
        AutoencoderLossBreakdown {
            reconstruction: scalar(1.0, device),
            masked: scalar(2.0, device),
            intruder: scalar(4.0, device),
            precursor: scalar(5.0, device),
            masked_precursor: scalar(6.0, device),
            similarity_ranking: scalar(7.0, device),
            regularization: scalar(8.0, device),
            chamfer_mz: scalar(3.0, device),
        }
    }

    fn diagnostics(device: &TestDevice) -> AutoencoderDiagnostics<TestBackend> {
        AutoencoderDiagnostics {
            similarity_ranking_pairs: scalar(9.0, device),
            similarity_ranking_accuracy: scalar(0.75, device),
            precursor_mae_da: scalar(12.5, device),
            self_linear_cosine: scalar(0.82, device),
            self_modified_linear_cosine: scalar(0.91, device),
            self_similarity_items: scalar(11.0, device),
        }
    }

    fn metadata() -> MetricMetadata {
        MetricMetadata {
            progress: Progress {
                items_processed: 1,
                items_total: 1,
            },
            global_progress: Progress {
                items_processed: 0,
                items_total: 1,
            },
            iteration: Some(0),
            lr: None,
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_tensor_close(tensor: Tensor<TestBackend, 1>, expected: f32) {
        assert_close(tensor.into_scalar(), expected);
    }

    fn assert_single_matrix_zero(tensor: Tensor<TestBackend, 2>) {
        let values = tensor
            .into_data()
            .to_vec::<f32>()
            .expect("synced placeholder tensor values");
        assert_eq!(values, vec![0.0]);
    }

    fn assert_metric_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-8,
            "expected {expected}, got {actual}"
        );
    }

    fn output(device: &TestDevice) -> AutoencoderTrainingOutput<TestBackend> {
        AutoencoderTrainingOutput::new_with_diagnostics(
            matrix(device),
            matrix(device),
            losses(device),
            diagnostics(device),
        )
    }

    #[test]
    fn loss_breakdown_total_sums_all_components() {
        let device = device();
        let total = losses(&device).total();

        assert_tensor_close(total, 36.0);
    }

    #[test]
    fn zeros_are_zero() {
        let device = device();
        let losses = AutoencoderLossBreakdown::<TestBackend>::zeros(&device);
        let diagnostics = AutoencoderDiagnostics::<TestBackend>::zeros(&device);

        assert_tensor_close(losses.total(), 0.0);
        assert_tensor_close(diagnostics.similarity_ranking_pairs, 0.0);
        assert_tensor_close(diagnostics.similarity_ranking_accuracy, 0.0);
        assert_tensor_close(diagnostics.precursor_mae_da, 0.0);
        assert_tensor_close(diagnostics.self_linear_cosine, 0.0);
        assert_tensor_close(diagnostics.self_modified_linear_cosine, 0.0);
        assert_tensor_close(diagnostics.self_similarity_items, 0.0);
    }

    #[test]
    fn training_output_new_computes_total_loss() {
        let device = device();
        let output =
            AutoencoderTrainingOutput::new(matrix(&device), matrix(&device), losses(&device));

        assert_tensor_close(output.loss, 36.0);
        assert_eq!(output.output.dims(), [2, 2]);
        assert_eq!(output.targets.dims(), [2, 2]);
        assert_tensor_close(output.diagnostics.similarity_ranking_pairs, 0.0);
        assert_tensor_close(output.diagnostics.similarity_ranking_accuracy, 0.0);
        assert_tensor_close(output.diagnostics.precursor_mae_da, 0.0);
        assert_tensor_close(output.diagnostics.self_linear_cosine, 0.0);
        assert_tensor_close(output.diagnostics.self_modified_linear_cosine, 0.0);
        assert_tensor_close(output.diagnostics.self_similarity_items, 0.0);
    }

    #[test]
    fn training_output_adapt_exposes_all_components() {
        let device = device();
        let adapted: AutoencoderLossComponentsInput<TestBackend> = output(&device).adapt();

        assert_tensor_close(adapted.loss, 36.0);
        assert_tensor_close(adapted.reconstruction, 1.0);
        assert_tensor_close(adapted.masked, 2.0);
        assert_tensor_close(adapted.intruder, 4.0);
        assert_tensor_close(adapted.precursor, 5.0);
        assert_tensor_close(adapted.masked_precursor, 6.0);
        assert_tensor_close(adapted.similarity_ranking, 7.0);
        assert_tensor_close(adapted.chamfer_mz, 3.0);
        assert_tensor_close(adapted.similarity_ranking_pairs, 9.0);
        assert_tensor_close(adapted.similarity_ranking_accuracy, 0.75);
        assert_tensor_close(adapted.precursor_mae_da, 12.5);
        assert_tensor_close(adapted.self_linear_cosine, 0.82);
        assert_tensor_close(adapted.self_modified_linear_cosine, 0.91);
        assert_tensor_close(adapted.self_similarity_items, 11.0);
    }

    #[test]
    fn training_output_sync_preserves_metric_tensors() {
        let device = device();
        let synced = output(&device).sync();

        assert_tensor_close(synced.loss, 36.0);
        assert_eq!(synced.output.dims(), [1, 1]);
        assert_eq!(synced.targets.dims(), [1, 1]);
        assert_single_matrix_zero(synced.output);
        assert_single_matrix_zero(synced.targets);
        assert_tensor_close(synced.losses.reconstruction, 1.0);
        assert_tensor_close(synced.losses.masked, 2.0);
        assert_tensor_close(synced.losses.intruder, 4.0);
        assert_tensor_close(synced.losses.precursor, 5.0);
        assert_tensor_close(synced.losses.masked_precursor, 6.0);
        assert_tensor_close(synced.losses.similarity_ranking, 7.0);
        assert_tensor_close(synced.losses.regularization, 8.0);
        assert_tensor_close(synced.losses.chamfer_mz, 3.0);
        assert_tensor_close(synced.diagnostics.similarity_ranking_pairs, 9.0);
        assert_tensor_close(synced.diagnostics.similarity_ranking_accuracy, 0.75);
        assert_tensor_close(synced.diagnostics.precursor_mae_da, 12.5);
        assert_tensor_close(synced.diagnostics.self_linear_cosine, 0.82);
        assert_tensor_close(synced.diagnostics.self_modified_linear_cosine, 0.91);
        assert_tensor_close(synced.diagnostics.self_similarity_items, 11.0);
    }

    #[test]
    fn component_metrics_report_expected_names_descriptions_and_attributes() {
        let cases = [
            (
                AutoencoderLossComponentMetric::<TestBackend>::loss(),
                "Loss",
                false,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::reconstruction(),
                "Reconstruction Loss",
                false,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::masked(),
                "Masked Loss",
                false,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::intruder(),
                "Intruder Loss",
                false,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::precursor(),
                "Precursor Loss",
                false,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::masked_precursor(),
                "Masked Precursor Loss",
                false,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::precursor_mae_da(),
                "Precursor MAE Da",
                false,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::similarity_ranking(),
                "Similarity Ranking Loss",
                false,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::similarity_ranking_accuracy(),
                "Similarity Ranking Accuracy",
                true,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::self_linear_cosine(),
                "Self Linear Cosine",
                true,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::self_modified_linear_cosine(),
                "Self Modified Linear Cosine",
                true,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::chamfer_mz(),
                "Chamfer mz",
                false,
            ),
        ];

        for (metric, expected_name, higher_is_better) in cases {
            assert_eq!(metric.name().as_str(), expected_name);
            assert!(
                metric
                    .description()
                    .expect("metric description")
                    .contains("Autoencoder")
            );

            let MetricAttributes::Numeric(attributes) = metric.attributes() else {
                panic!("autoencoder component metrics should be numeric");
            };
            assert_eq!(attributes.higher_is_better, higher_is_better);
            assert_eq!(attributes.unit, None);
        }
    }

    #[test]
    fn component_metrics_update_from_expected_tensor() {
        let device = device();
        let item: AutoencoderLossComponentsInput<TestBackend> = output(&device).adapt();
        let metadata = metadata();
        let cases = [
            (AutoencoderLossComponentMetric::<TestBackend>::loss(), 36.0),
            (
                AutoencoderLossComponentMetric::<TestBackend>::masked_precursor(),
                6.0,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::precursor_mae_da(),
                12.5,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::similarity_ranking(),
                7.0,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::similarity_ranking_accuracy(),
                0.75,
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::self_linear_cosine(),
                f64::from(0.82_f32),
            ),
            (
                AutoencoderLossComponentMetric::<TestBackend>::self_modified_linear_cosine(),
                f64::from(0.91_f32),
            ),
        ];

        for (mut metric, expected) in cases {
            let entry = metric.update(&item, &metadata);

            assert_metric_close(metric.value().current(), expected);
            assert_metric_close(metric.running_value().current(), expected);
            assert_eq!(entry.serialized, metric.value().serialize());
        }
    }

    #[test]
    fn similarity_ranking_accuracy_uses_pair_count_for_aggregation() {
        let device = device();
        let item: AutoencoderLossComponentsInput<TestBackend> = output(&device).adapt();
        let metadata = metadata();
        let mut metric =
            AutoencoderLossComponentMetric::<TestBackend>::similarity_ranking_accuracy();

        metric.update(&item, &metadata);

        match metric.value() {
            NumericEntry::Aggregated {
                aggregated_value,
                count,
            } => {
                assert_metric_close(aggregated_value, 0.75);
                assert_eq!(count, 9);
            }
            NumericEntry::Value(_) => panic!("similarity ranking accuracy should be aggregated"),
        }
    }

    #[test]
    fn self_similarity_metrics_use_sample_count_for_aggregation() {
        let device = device();
        let item: AutoencoderLossComponentsInput<TestBackend> = output(&device).adapt();
        let metadata = metadata();
        let mut metric = AutoencoderLossComponentMetric::<TestBackend>::self_linear_cosine();

        metric.update(&item, &metadata);

        match metric.value() {
            NumericEntry::Aggregated {
                aggregated_value,
                count,
            } => {
                assert_metric_close(aggregated_value, 0.82);
                assert_eq!(count, 11);
            }
            NumericEntry::Value(_) => panic!("self-similarity metric should be aggregated"),
        }
    }

    #[test]
    fn component_metric_clear_resets_state() {
        let device = device();
        let item: AutoencoderLossComponentsInput<TestBackend> = output(&device).adapt();
        let metadata = metadata();
        let mut metric = AutoencoderLossComponentMetric::<TestBackend>::loss();

        metric.update(&item, &metadata);
        metric.clear();

        match metric.value() {
            NumericEntry::Aggregated {
                aggregated_value,
                count,
            } => {
                assert!(aggregated_value.is_nan());
                assert_eq!(count, 0);
            }
            NumericEntry::Value(_) => panic!("cleared metric should be aggregated"),
        }
    }
}
