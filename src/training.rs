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
    /// Latent-consistency contribution to the total loss.
    pub consistency: Tensor<B, 1>,
    /// Synthetic intruder-peak detection contribution to the total loss.
    pub intruder: Tensor<B, 1>,
    /// In-batch clean-spectrum similarity-ranking contribution to the total loss.
    pub similarity_ranking: Tensor<B, 1>,
    /// Optional explicit model-parameter regularization contribution.
    pub regularization: Tensor<B, 1>,
}

impl<B: Backend> AutoencoderLossBreakdown<B> {
    /// Returns a zero-valued breakdown.
    pub fn zeros(device: &B::Device) -> Self {
        Self {
            reconstruction: Tensor::zeros([1], device),
            masked: Tensor::zeros([1], device),
            consistency: Tensor::zeros([1], device),
            intruder: Tensor::zeros([1], device),
            similarity_ranking: Tensor::zeros([1], device),
            regularization: Tensor::zeros([1], device),
        }
    }

    /// Returns the weighted total loss.
    pub fn total(&self) -> Tensor<B, 1> {
        self.reconstruction.clone()
            + self.masked.clone()
            + self.consistency.clone()
            + self.intruder.clone()
            + self.similarity_ranking.clone()
            + self.regularization.clone()
    }
}

/// Non-loss diagnostics reported to the TUI.
pub struct AutoencoderDiagnostics<B: Backend> {
    /// Number of valid similarity-ranking pairs contributing to the batch.
    pub similarity_ranking_pairs: Tensor<B, 1>,
    /// Latent-vs-target ordering accuracy over valid similarity-ranking pairs.
    pub similarity_ranking_accuracy: Tensor<B, 1>,
}

impl<B: Backend> AutoencoderDiagnostics<B> {
    /// Returns zero-valued diagnostics.
    pub fn zeros(device: &B::Device) -> Self {
        Self {
            similarity_ranking_pairs: Tensor::zeros([1], device),
            similarity_ranking_accuracy: Tensor::zeros([1], device),
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
            consistency: self.losses.consistency.clone(),
            intruder: self.losses.intruder.clone(),
            similarity_ranking: self.losses.similarity_ranking.clone(),
            similarity_ranking_pairs: self.diagnostics.similarity_ranking_pairs.clone(),
            similarity_ranking_accuracy: self.diagnostics.similarity_ranking_accuracy.clone(),
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
            consistency,
            intruder,
            similarity_ranking,
            regularization,
            similarity_ranking_pairs,
            similarity_ranking_accuracy,
        ] = Transaction::default()
            .register(self.loss)
            .register(self.losses.reconstruction)
            .register(self.losses.masked)
            .register(self.losses.consistency)
            .register(self.losses.intruder)
            .register(self.losses.similarity_ranking)
            .register(self.losses.regularization)
            .register(self.diagnostics.similarity_ranking_pairs)
            .register(self.diagnostics.similarity_ranking_accuracy)
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
                consistency: Tensor::from_data(consistency, device),
                intruder: Tensor::from_data(intruder, device),
                similarity_ranking: Tensor::from_data(similarity_ranking, device),
                regularization: Tensor::from_data(regularization, device),
            },
            diagnostics: AutoencoderDiagnostics {
                similarity_ranking_pairs: Tensor::from_data(similarity_ranking_pairs, device),
                similarity_ranking_accuracy: Tensor::from_data(similarity_ranking_accuracy, device),
            },
        }
    }
}

/// Metric input containing reported autoencoder loss components and diagnostics.
pub struct AutoencoderLossComponentsInput<B: Backend> {
    loss: Tensor<B, 1>,
    reconstruction: Tensor<B, 1>,
    masked: Tensor<B, 1>,
    consistency: Tensor<B, 1>,
    intruder: Tensor<B, 1>,
    similarity_ranking: Tensor<B, 1>,
    similarity_ranking_pairs: Tensor<B, 1>,
    similarity_ranking_accuracy: Tensor<B, 1>,
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

    /// Latent-consistency loss contribution.
    pub fn consistency() -> Self {
        Self::new(AutoencoderLossComponent::Consistency)
    }

    /// Synthetic intruder-peak detection loss contribution.
    pub fn intruder() -> Self {
        Self::new(AutoencoderLossComponent::Intruder)
    }

    /// In-batch clean-spectrum similarity-ranking loss contribution.
    pub fn similarity_ranking() -> Self {
        Self::new(AutoencoderLossComponent::SimilarityRanking)
    }

    /// Number of valid similarity-ranking pairs.
    pub fn similarity_ranking_pairs() -> Self {
        Self::new(AutoencoderLossComponent::SimilarityRankingPairs)
    }

    /// Similarity-ranking latent-vs-target ordering accuracy.
    pub fn similarity_ranking_accuracy() -> Self {
        Self::new(AutoencoderLossComponent::SimilarityRankingAccuracy)
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
            AutoencoderLossComponent::Consistency => item.consistency.clone(),
            AutoencoderLossComponent::Intruder => item.intruder.clone(),
            AutoencoderLossComponent::SimilarityRanking => item.similarity_ranking.clone(),
            AutoencoderLossComponent::SimilarityRankingPairs => {
                item.similarity_ranking_pairs.clone()
            }
            AutoencoderLossComponent::SimilarityRankingAccuracy => {
                item.similarity_ranking_accuracy.clone()
            }
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
    Consistency,
    Intruder,
    SimilarityRanking,
    SimilarityRankingPairs,
    SimilarityRankingAccuracy,
}

impl AutoencoderLossComponent {
    const fn name(self) -> &'static str {
        match self {
            Self::Loss => "Loss",
            Self::Reconstruction => "Reconstruction Loss",
            Self::Masked => "Masked Loss",
            Self::Consistency => "Consistency Loss",
            Self::Intruder => "Intruder Loss",
            Self::SimilarityRanking => "Similarity Ranking Loss",
            Self::SimilarityRankingPairs => "Similarity Ranking Pairs",
            Self::SimilarityRankingAccuracy => "Similarity Ranking Accuracy",
        }
    }

    const fn description(self) -> &'static str {
        match self {
            Self::Loss => "total weighted loss",
            Self::Reconstruction => "reconstruction",
            Self::Masked => "masked-peak",
            Self::Consistency => "latent-consistency",
            Self::Intruder => "intruder-peak detection",
            Self::SimilarityRanking => "teacher similarity ranking",
            Self::SimilarityRankingPairs => "teacher similarity-ranking valid pairs",
            Self::SimilarityRankingAccuracy => "teacher similarity-ranking ordering accuracy",
        }
    }

    const fn higher_is_better(self) -> bool {
        matches!(self, Self::SimilarityRankingAccuracy)
    }

    fn batch_size<B: Backend>(self, item: &AutoencoderLossComponentsInput<B>) -> usize {
        match self {
            Self::SimilarityRankingAccuracy => metric_scalar(&item.similarity_ranking_pairs)
                .max(1.0)
                .round() as usize,
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
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::masked())
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::consistency())
            .metric_valid_numeric(AutoencoderLossComponentMetric::<NdArray>::consistency())
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::intruder())
            .metric_train_numeric(AutoencoderLossComponentMetric::<NdArray>::similarity_ranking())
            .metric_valid_numeric(AutoencoderLossComponentMetric::<NdArray>::similarity_ranking())
            .metric_train_numeric(
                AutoencoderLossComponentMetric::<NdArray>::similarity_ranking_pairs(),
            )
            .metric_valid_numeric(
                AutoencoderLossComponentMetric::<NdArray>::similarity_ranking_pairs(),
            )
            .metric_train_numeric(
                AutoencoderLossComponentMetric::<NdArray>::similarity_ranking_accuracy(),
            )
            .metric_valid_numeric(
                AutoencoderLossComponentMetric::<NdArray>::similarity_ranking_accuracy(),
            )
    }
}
