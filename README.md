## Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry

Codebase for the synthetic experiments and figures in the paper ["Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry"](https://arxiv.org/abs/2503.01822) .

## Overview

1. `models.py` defines SAEs- ReLU, JumpReLU, TopK and SpaDE (see `functions/utils.py` for implementation of JumpReLU)
2. `syntheticdatasets/` consists of jupyter notebooks which create datasets of gaussian clusters for the separability and heterogeneity experiments
3. `functions/` includes files to preprocess/load data (`get_data.py`), training pipeline for models (`train_test.py`) and miscellaneous functions (`utils.py`)
4. `experiments/expt_heterogeneity` and `experiments/expt_separability` consist of relevant experiment settings (`settings.txt`); code to train SAEs (`train.py`), and jupyter notebooks to generate figures (both for main paper and appendix)
5. `experiments/makefigs_vision`, `experiments/makefigs_formallanguage` generate main paper figures for vision and formal language experiments (from results obtained already).

## Usage

1. Go to `experiments/expt_separability` or `experiments/expt_heterogeneity`, choose configuration of SAE in hyperparameters (`hyperparameters2.csv`) and settings (`settings.txt`) files and run `trainmodel.py`. Note: _gamma_reg_ is the scaling constant for sparsity regularizer in the loss, _kval_topk_ is the _K_ in TopK
2. Run jupyter notebooks to create main figures/ appendix figures
