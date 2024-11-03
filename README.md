# BrainNormativeCVAE

A Python package for normative modeling of brain imaging data using conditional Variational Autoencoders (cVAE).

## Overview

BrainNormativeCVAE is a comprehensive toolkit for building and applying normative models to neuroimaging data. It implements a conditional Variational Autoencoder approach that can:
- Learn normative patterns in brain imaging data
- Account for multiple demographic and clinical covariates
- Generate statistical estimates of deviation from normative patterns
- Perform bootstrap analysis for robust statistical inference

## Installation

### From PyPI
```bash
pip install BrainNormativeCVAE
```

### From Source
```bash
git clone https://github.com/maiho24/BrainNormativeCVAE.git
cd BrainNormativeCVAE
pip install -e .


## Quick Start

### Training a Model

```bash
# Direct training with specified parameters
brain-cvae-train \
    --config configs/default_config.yaml \
    --mode direct \
    --data_dir path/to/data \
    --output_dir path/to/output \
    --gpu
```

```bash
# Hyperparameter optimization with Optuna
brain-cvae-train \
    --config configs/default_config.yaml \
    --mode optuna \
    --data_dir path/to/data \
    --output_dir path/to/output \
    --gpu
```

### Running Inference

```bash
brain-cvae-inference \
    --model_path path/to/model.pkl \
    --data_dir path/to/test_data \
    --output_dir path/to/results \
    --num_samples 1000 \
    --num_bootstraps 1000
```

## Data Format

### Required Files
- `train_data_subset.csv`: Training data features
- `train_covariates_subset.csv`: Training data covariates
- `test_data_subset.csv`: Test data features
- `test_covariates_subset.csv`: Test data covariates

### Covariate Structure
The following covariates are expected for the current implementation:
- Age (continuous)
- wb_EstimatedTotalIntraCranial_Vol (continuous)
- Sex (categorical)
- Diabetes Status (categorical)
- Smoking Status (categorical)
- Hypercholesterolemia Status (categorical)
- Obesity Status (categorical)

To apply the package to other covariates, modifications are required in the `process_covariates()` function located in `utils/data.py`.

## Configuration

Create a YAML configuration file with the following structure:
```yaml
model:
  input_dim: 56
  hidden_dim: [128]
  latent_dim: 32
  non_linear: true
  beta: 1.0
  learning_rate: 0.001

training:
  mode: "direct"  # or "optuna"
  epochs: 200
  batch_size: 64
  early_stopping_patience: 20
  validation_split: 0.15

optuna:
  n_trials: 100
  study_name: "cvae_optimization"
  search_space:
    hidden_dim:
      choices: [[32], [64], [128]]
    latent_dim:
      choices: [8, 16, 32]
    learning_rate:
      min: 1e-5
      max: 1e-3

paths:
  data_dir: "data/"
  model_dir: "models/"
  output_dir: "output/"
```

## Output Files

### Training
- `final_model.pkl`: Trained model
- `config.yaml`: Used configuration
- `best_params.yaml`: Best hyperparameters (Optuna mode)
- `Losses_training_validation.png`: Training curves

### Inference
- `test_reconstruction_vars.csv`: Reconstruction variances
- `bootstrapped_means.csv`: Bootstrap analysis means
- `bootstrapped_variances.csv`: Bootstrap analysis variances
- Bootstrap confidence intervals and visualization plots

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Attribution

This implementation extends the normative modelling framework by [Lawry Aguila et al. (2022)](https://github.com/alawryaguila/normativecVAE), with refinements in both the model architecture and inference approach.

If you use this package, please cite both our work and the original implementation:

```bibtex
@software{brainnormativecvae2024,
  author = {Ho. M},
  title = {An Enhanced Conditional Variational Autoencoder-Based Normative Model with Latent Space Sampling and Bootstrapping for Neuroimaging Analysis},
  year = {2024},
  url = {https://github.com/maiho24/BrainNormativeCVAE}
}

@software{lawryaguila2022normativecvae,
  author = {Lawry Aguila, A., Chapman, J., Janahi, M., Altmann, A.},
  title = {Conditional VAEs for confound removal and normative modelling of neurodegenerative diseases},
  year = {2022},
  url = {https://github.com/alawryaguila/normativecVAE}
}
```