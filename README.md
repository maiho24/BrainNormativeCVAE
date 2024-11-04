# BrainNormativeCVAE

A Python package for normative modeling of brain imaging data using conditional Variational Autoencoders (cVAE).

## Overview

BrainNormativeCVAE is a comprehensive toolkit for building and applying normative models to neuroimaging data. It implements a conditional Variational Autoencoder approach that can:
- Learn normative patterns in brain imaging data
- Account for multiple demographic and clinical covariates
- Generate statistical estimates of deviation from normative patterns
- Perform bootstrap analysis for robust statistical inference

## Installation

```bash
git clone https://github.com/maiho24/BrainNormativeCVAE.git
cd BrainNormativeCVAE
pip install -e .
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
    --output_dir path/to/output \
    --gpu

# Hyperparameter optimization with Optuna
brain-cvae-train \
    --config configs/default_config.yaml \
    --mode optuna \
    --output_dir path/to/output \
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
    --output_dir path/to/output \
    --num_samples 1000 \
    --num_bootstraps 1000 \
    --gpu
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

To apply the package to other covariates, modifications are required in the process_covariates() function located in utils/data.py.

## Configuration

Create a YAML configuration file with the following structure:

```yaml
model:
  input_dim: 56 # Replace with your actual input dimension
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

optuna: # Optional, for hyperparameter optimization
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
  gpu: false  # Set to true to use GPU if available
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
    ├── reconstruction_variances.csv    # Model's reconstruction uncertainty for test data
    │
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

#### Reconstruction Analysis
- `reconstruction_variances.csv`: Model's uncertainty in reconstructing test data points, indicating reliability of predictions

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

This implementation extends the normative modelling framework by [Lawry Aguila et al. (2022)](https://github.com/alawryaguila/normativecVAE), with refinements in both the model architecture and inference approach.

If you use this package, please cite both our work and the original implementation:

```bibtex
@software{brainnormativecvae2024,
  author = {Ho. M},
  title = {An Enhanced Conditional Variational Autoencoder-Based Normative Model for Neuroimaging Analysis},
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