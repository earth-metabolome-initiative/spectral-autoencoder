//! Model-parameter regularization helpers.

use burn::{
    module::{Module, ModuleVisitor, Param},
    prelude::*,
    tensor::Tensor,
};
use serde::{Deserialize, Serialize};

/// L1/L2 model-parameter regularization configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// L1 penalty weight.
    #[serde(default)]
    pub l1: f64,
    /// L2 penalty weight.
    #[serde(default)]
    pub l2: f64,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self { l1: 0.0, l2: 0.0 }
    }
}

impl RegularizationConfig {
    /// Returns `true` when no regularization penalty is active.
    #[must_use]
    pub fn is_disabled(&self) -> bool {
        self.l1 == 0.0 && self.l2 == 0.0
    }

    /// Computes the regularization penalty over all float parameters in a module.
    pub fn penalty<B, M>(&self, module: &M, device: &B::Device) -> Tensor<B, 1>
    where
        B: Backend,
        M: Module<B>,
    {
        if self.is_disabled() {
            return Tensor::zeros([1], device);
        }

        let mut visitor = RegularizationVisitor::<B>::new(*self, device);
        module.visit(&mut visitor);
        visitor.finish()
    }
}

struct RegularizationVisitor<B: Backend> {
    config: RegularizationConfig,
    penalty: Tensor<B, 1>,
}

impl<B: Backend> RegularizationVisitor<B> {
    fn new(config: RegularizationConfig, device: &B::Device) -> Self {
        Self {
            config,
            penalty: Tensor::zeros([1], device),
        }
    }

    fn finish(self) -> Tensor<B, 1> {
        self.penalty
    }
}

impl<B: Backend> ModuleVisitor<B> for RegularizationVisitor<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let tensor = param.val();
        if self.config.l1 != 0.0 {
            self.penalty = self.penalty.clone() + tensor.clone().abs().sum() * self.config.l1;
        }
        if self.config.l2 != 0.0 {
            self.penalty =
                self.penalty.clone() + tensor.clone().powf_scalar(2.0).sum() * self.config.l2;
        }
    }
}
