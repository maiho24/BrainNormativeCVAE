# BrainNormativeCVAE

A Python package for normative modeling of brain imaging data using conditional Variational Autoencoders (cVAE).

## Overview

BrainNormativeCVAE is a comprehensive toolkit for building and applying normative models to neuroimaging data. It implements a conditional Variational Autoencoder approach that can:
- Learn normative patterns from brain imaging data.
- Incorporate multiple demographic and clinical covariates.
- Generate probabilistic estimates (means and standard deviations) for input covariates.
- Perform robust statistical inference using bootstrap analysis.

## Installation

Using Conda helps manage dependencies and ensures compatibility across different systems. We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to keep your environment lightweight.

1. Create and activate a new conda environment:
```bash
conda create -n brain_cvae
conda activate brain_cvae
```
2. Clone and install the package:
```bash
git clone https://github.com/maiho24/BrainNormativeCVAE.git
cd BrainNormativeCVAE
pip install -e .
```
**Note**: Make sure to always activate the environment before using the package:
```bash
conda activate brain_cvae
```

## Quick Start

### Training a Model

```bash
# Show help message and available options
brain-cvae-train --help

# Direct training with specified parameters
brain-cvae-train \
    --config configs/default_config.yaml \
    --mode direct \
    --gpu

# Hyperparameter optimization with Optuna
brain-cvae-train \
    --config configs/default_config.yaml \
    --mode optuna \
    --gpu
```

### Running Inference

```bash
# Show help message and available options
brain-cvae-inference --help

# Run inference and bootstrap analysis
brain-cvae-inference \
    --model_path path/to/output/models/final_model.pkl \
    --config configs/default_config.yaml \
    --num_samples 1000 \
    --num_bootstraps 1000 \
```

## Data Format

### Required Files
- `train_data.csv`: Training data features
- `train_covariates.csv`: Training data covariates
- `test_data.csv`: Test data features
- `test_covariates.csv`: Test data covariates

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
  hidden_dim: [128]  # Architecture of hidden layers, e.g., [128, 64] means two hidden layers with 128 and 64 neurons
  latent_dim: 32  # Required if "direct" mode is used
  non_linear: true
  beta: 1.0  # Required if "direct" mode is used
  learning_rate: 0.001  # Required if "direct" mode is used

training:
  mode: "direct"  # Options: "direct" or "optuna"
  epochs: 200
  batch_size: 64
  early_stopping_patience: 20
  validation_split: 0.15

optuna: # Optional
  n_trials: 100
  study_name: "cvae_optimization"
  search_space:
    hidden_dim:
      type: "categorical"
      choices: [[32], [64], [128]]
    latent_dim:
      type: "categorical"
      choices: [16, 32]
    learning_rate:
      type: "loguniform"
      min: 1e-5
      max: 1e-3
    batch_size:
      type: "categorical"
      choices: [32, 64]
    beta:
      type: "categorical"
      choices: [0.75, 1.0]

paths:
  data_dir: "path/to/data/"
  output_dir: "path/to/output/"

device:
  gpu: true
```

## Output Directory Structure

The package organizes all outputs in a consistent directory structure:

```
output_dir/
├── logs/                               # Logging directory
│   ├── training_YYYYMMDD_HHMMSS.log    # Training process logs
│   └── inference_YYYYMMDD_HHMMSS.log   # Inference and bootstrap analysis logs
│
├── models/                             # Model directory
│   ├── config.yaml                     # Training configuration parameters
│   ├── final_model.pkl                 # Saved trained model
│   └── best_params.yaml                # Best hyperparameters (Optuna mode only)
│
└── results/                            # Analysis results directory
    ├── bootstrapped_means.csv          # Mean predictions for each covariate combination
    ├── bootstrapped_variances.csv      # Variance of predictions for each combination
    │
    ├── ci_means_lower.csv              # Lower confidence interval bounds for means
    ├── ci_means_upper.csv              # Upper confidence interval bounds for means
    ├── ci_variances_lower.csv          # Lower confidence interval bounds for variances
    ├── ci_variances_upper.csv          # Upper confidence interval bounds for variances
    │
    ├── feature_variability.csv         # Variability of each feature across covariates
    ├── covariate_sensitivity.csv       # Impact of each covariate on predictions
    └── summary_statistics.xlsx         # Overall statistical summary of results
```

### Results File Descriptions

#### Bootstrap Analysis Results
- `bootstrapped_means.csv`: Average predictions for each covariate combination across bootstrap samples
- `bootstrapped_variances.csv`: Variation in predictions for each covariate combination across bootstrap samples
- `ci_means_lower.csv`: Lower bound of confidence interval for mean predictions
- `ci_means_upper.csv`: Upper bound of confidence interval for mean predictions
- `ci_variances_lower.csv`: Lower bound of confidence interval for prediction variances
- `ci_variances_upper.csv`: Upper bound of confidence interval for prediction variances

#### Feature Analysis
- `feature_variability.csv`: Measures how much each feature varies across different covariate combinations
- `covariate_sensitivity.csv`: Quantifies how much each covariate influences the predictions
- `summary_statistics.xlsx`: Comprehensive statistical summary including:
  - Global means and standard deviations
  - Quantile distributions (25th, 50th, 75th percentiles)
  - Summary statistics for both means and variances

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Attribution

This implementation extends the normative modelling framework by [Lawry Aguila et al. (2022)](https://github.com/alawryaguila/normativecVAE), with substantial refinements in both the model architecture and inference approach.

If you use this package, please cite both our work and the original implementation:

```bibtex
@software{Ho_BrainNormativecVAE,
  author = {Ho, M., Song, Y., Sachdev, P., Jiang, J., Wen, W.},
  title = {An Enhanced Conditional Variational Autoencoder-Based Normative Model for Neuroimaging Analysis},
  year = {2025},
  url = {https://github.com/maiho24/BrainNormativeCVAE}
}

@software{LawryAguila_normativecVAE,
  author = {Lawry Aguila, A., Chapman, J., Janahi, M., Altmann, A.},
  title = {Conditional VAEs for confound removal and normative modelling of neurodegenerative diseases},
  year = {2022},
  url = {https://github.com/alawryaguila/normativecVAE},
}
```