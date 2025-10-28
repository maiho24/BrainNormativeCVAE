# BrainNormativeCVAE

A Python package for normative modeling of brain imaging data using conditional Variational Autoencoders (cVAE).

## Overview

BrainNormativeCVAE is a comprehensive toolkit for building and applying normative models to neuroimaging data. It implements a conditional Variational Autoencoder approach that can:
- Learn normative patterns from brain imaging data
- Perform robust statistical inference using bootstrap analysis
- Generate probabilistic estimates (means and standard deviations) for input covariates

## Installation

Using Conda helps manage dependencies and ensures compatibility across different systems. We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to keep your environment lightweight.

1. Create and activate a new conda environment:
```bash
conda create -n brain_cvae python=3.9
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

# Run inference with bootstrap analysis (default behaviour)
brain-cvae-inference \
    --model_path path/to/output/models/final_model.pkl \
    --config configs/default_config.yaml \
    --num_samples 1000 \
    --bootstrap \
    --num_bootstraps 1000

# Run inference with simple sampling (no bootstrap)
brain-cvae-inference \
    --model_path path/to/output/models/final_model.pkl \
    --config configs/default_config.yaml \
    --num_samples 1000 \
    --no-bootstrap

# Run inference without config file (specifying directories directly)
brain-cvae-inference \
    --model_path path/to/output/models/final_model.pkl \
    --data_dir path/to/data \
    --output_dir path/to/output \
    --num_samples 1000 \
    --no-bootstrap
```

## Data Format

### Required Files
For training, these files are required:
- `train_data.csv`: Training data features
- `train_covariates.csv`: Training data covariates

For inference, the following files should be present in the data directory:
- `test_data.csv`: Test data features
- `test_covariates.csv`: Test data covariates

### Covariate Structure
The following covariates are expected for the current implementation:
- Age (continuous)
- eTIV (continuous)
- Sex (categorical)
- Diabetes Status (categorical)
- Smoking Status (categorical)
- Hypercholesterolemia Status (categorical)
- Obesity Status (categorical)

To apply the package to other covariates, modifications are required in the `process_covariates()` function located in `utils/data.py`.

## Configuration

### Training Configuration
Create a YAML configuration file with the following structure for training:

```yaml
model:
  hidden_dim: "128_64"  # Required if "direct" mode is used. Hidden layer architecture in string format, e.g., "128_64" for two layers
  latent_dim: 32        # Required if "direct" mode is used
  non_linear: true
  beta: 1.0             # Required if "direct" mode is used
  learning_rate: 0.001  # Required if "direct" mode is used

training:
  epochs: 200           # Required if "direct" mode is used
  batch_size: 64
  early_stopping_patience: 20
  validation_split: 0.15

optuna: # Optional
  n_trials: 100
  study_name: "cvae_optimisation"
  search_space:
    hidden_dim:
      type: "categorical"
      choices: ["32", "64_32", "128_64"]
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
```
#### Command Line Arguments

```
Required argument:
  --config PATH              Path to configuration file

Optional arguments:
  --mode TYPE                Training mode: direct or optuna (default: direct)
  --data_dir PATH            Override data directory specified in config file
  --output_dir PATH          Override output directory specified in config file
  --gpu                      Use GPU for training if available
```
The package provides two training options:
- **direct** (default): Train with parameters specified in config file
- **optuna**: Perform hyperparameter optimization using the [Optuna](https://optuna.org/) framework

### Inference Configuration
For inference, you can either:
1. Use a config file with the `--config` option
2. Specify required directories directly using `--data_dir` and `--output_dir`

If using a config file, only these sections are required:

```yaml
paths:
  data_dir: "path/to/data/"     # Directory containing input data files
  output_dir: "path/to/output/" # Directory for storing results
```

Note: Command line arguments (`--data_dir`, `--output_dir`) will override the corresponding values in the config file if both are provided.

#### Bootstrap vs Simple Sampling
The package offers two analysis modes for inference:

##### Bootstrap Analysis (Default)
* Provides robust statistical estimates with confidence intervals
* More computationally intensive but offers uncertainty quantification
* Outputs: means, variances, and confidence intervals

##### Simple Sampling
* Fast computation with basic statistics
* Provides conditional means and variances without confidence intervals
* Outputs: means and variances only

#### Command Line Arguments

```
Required arguments (either --config OR both --data_dir and --output_dir must be provided):
  --model_path PATH          Path to trained model checkpoint (required)
  --config PATH              Path to configuration file
  --data_dir PATH            Directory containing input data (required if --config not provided)
  --output_dir PATH          Directory for output files (required if --config not provided)

Optional arguments:
  --num_samples INT          Number of samples for analysis (default: 1000)
  --bootstrap                Enable bootstrap analysis (default: enabled)
  --no-bootstrap             Disable bootstrap analysis and use simple sampling
  --num_bootstraps INT       Number of bootstrap iterations (default: 1000, only used with --bootstrap)
  --confidence_level FLOAT   Confidence level for bootstrap CIs (default: 0.95, only used with --bootstrap)
  --prediction_type TYPE     Method for prediction: covariate or dual_input (default: covariate)
  --gpu                      Use GPU for inference if available
```

The package provides two prediction modes:
- **Covariate-based (Default, Primary approach)**: Generates normative predictions directly from demographic and clinical variables
- **Dual-input (For comparison only)**: Uses both observed data and covariates for prediction, implemented solely for methodological comparison purposes (see our [preprint](https://www.biorxiv.org/content/10.1101/2025.01.05.631276v1) for detailed discussion)

## Output Directory Structure

The package organises all outputs in a consistent directory structure. The structure varies depending on whether bootstrap analysis is enabled:

### With Bootstrap Analysis (Default)
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

### With Simple Sampling (--no-bootstrap)

```
output_dir/
├── logs/                               # Logging directory
│   ├── training_YYYYMMDD_HHMMSS.log    # Training process logs
│   └── inference_YYYYMMDD_HHMMSS.log   # Inference logs
│
├── models/                             # Model directory
│   ├── config.yaml                     # Training configuration parameters
│   ├── final_model.pkl                 # Saved trained model
│   └── best_params.yaml                # Best hyperparameters (Optuna mode only)
│
└── results/                            # Analysis results directory
    ├── means.csv                       # Mean predictions for each covariate combination
    ├── variances.csv                   # Variance of predictions for each combination
    │
    ├── feature_variability.csv         # Variability of each feature across covariates
    ├── covariate_sensitivity.csv       # Impact of each covariate on predictions
    └── summary_statistics.xlsx         # Overall statistical summary of results
```

### Results File Descriptions

#### Bootstrap Analysis Results (--bootstrap)
- `bootstrapped_means.csv`: Average predictions for each covariate combination across bootstrap samples
- `bootstrapped_variances.csv`: Variation in predictions for each covariate combination across bootstrap samples
- `ci_means_lower.csv`: Lower bound of confidence interval for mean predictions
- `ci_means_upper.csv`: Upper bound of confidence interval for mean predictions
- `ci_variances_lower.csv`: Lower bound of confidence interval for prediction variances
- `ci_variances_upper.csv`: Upper bound of confidence interval for prediction variances

#### Simple Sampling Results (--no-bootstrap)
- `means.csv`: Mean predictions for each covariate combination
- `variances.csv`: Variance of predictions for each covariate combination

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
  author = {Ho M., Song Y., Sachdev P., Fan L., Jiang J., Wen W.},
  title = {An Enhanced Conditional Variational Autoencoder-Based Normative Model for Neuroimaging Analysis},
  year = {2025},
  url = {https://github.com/maiho24/BrainNormativeCVAE}
}

@software{LawryAguila_normativecVAE,
  author = {Lawry Aguila A., Chapman J., Janahi M., Altmann A.},
  title = {Conditional VAEs for confound removal and normative modelling of neurodegenerative diseases},
  year = {2022},
  url = {https://github.com/alawryaguila/normativecVAE}
}
```