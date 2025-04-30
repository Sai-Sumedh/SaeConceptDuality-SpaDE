## Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry

Codebase for the synthetic experiments and figures in the paper ["Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry"](https://arxiv.org/abs/2503.01822) .

## Overview

1. *SAE Definitions*: `models.py` defines SAEs- ReLU, JumpReLU, TopK and SpaDE
2. *Generating Synthetic Data*: `syntheticdatasets/` consists of jupyter notebooks which create datasets of gaussian clusters for the separability and heterogeneity experiments. Choose location to save data by modifying `dataset_dir` in these notebooks. 
3. *Synthetic data experiments*: `experiments/expt_heterogeneity` and `experiments/expt_separability` consist of relevant experiment settings (`settings.txt`), hyperparameter files (`hyperparameters2.csv`); code to train SAEs (`train.py`), and jupyter notebooks to generate figures (both for main paper and appendix)
4. *Making figures for formal language and vision experiments*: `experiments/makefigs_vision`, `experiments/makefigs_formallanguage` generate main paper figures for vision and formal language experiments (from results obtained already).
5. *Functions for data, training and utilities*: `functions/` includes files to preprocess/load data (`get_data.py`), training pipeline for models (`train_test.py`) and miscellaneous functions (`utils.py`)

## Usage

1. *Generate synthetic data*: Go to `syntheticdatasets/` and in either notebook, modify `dataset_dir` to choose location to save data. Run this notebook.
2. *Run Synthetic data experiments*: 
    a. Go to `experiments/expt_separability` or `experiments/expt_heterogeneity`, 
    b. choose configuration of SAE in hyperparameters (`hyperparameters2.csv`) and 
    c. modify `dataset` and `datapath` in settings (`settings.txt`) files and 
    d. run `trainmodel.py`. 
Note: _gamma_reg_ is the scaling constant for sparsity regularizer in the loss, _kval_topk_ is the _K_ in TopK
3. *Generating figures*: Open `mainfigures.ipynb` or `appendixfigures.ipynb`, modify `dataname` and `DATA_PATH`, and run the jupyter notebook to replicate main figures/ appendix figures
