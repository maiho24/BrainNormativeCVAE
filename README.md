# BrainNormativeCVAE

A Python package for normative modelling of brain imaging data using conditional Variational Autoencoders (cVAE).

## Overview

BrainNormativeCVAE is a comprehensive toolkit for building and applying normative models to neuroimaging data. It implements a conditional Variational Autoencoder approach that can:
- Learn normative patterns from brain imaging data
- Generate probabilistic estimates (means and variances) for input covariates
- Optionally perform robust statistical inference using bootstrap analysis for confidence intervals

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
    --config configs/direct_config.yaml \
    --mode direct \
    --gpu

# Hyperparameter optimisation with Optuna
brain-cvae-train \
    --config configs/optuna_config.yaml \
    --mode optuna \
    --gpu
```

### Running Inference

```bash
# Show help message and available options
brain-cvae-inference --help

# Run inference without config file (specifying directories directly)
brain-cvae-inference \
    --model_path path/to/output/models/final_model.pkl \
    --data_dir path/to/data \
    --output_dir path/to/output \
    --num_samples 1000 \
    --no-bootstrap
    --gpu

# Run inference with bootstrap analysis for confidence intervals
brain-cvae-inference \
    --model_path path/to/output/models/final_model.pkl \
    --data_dir path/to/data \
    --output_dir path/to/output \
    --num_samples 1000 \
    --bootstrap \
    --num_bootstraps 1000
    --gpu
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
- Sex (binary)
- Diabetes Status (binary)
- Smoking Status (binary)
- Hypercholesterolemia Status (binary)
- Obesity Status (binary)

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
  variance_type: "two_heads"  # Variance modelling: "two_heads" (default), "global_learnable", or "covariate_specific"
  varnet_hidden_dim: "32"     # Required if variance_type is "covariate_specific"

training:
  epochs: 200           # Required if "direct" mode is used
  batch_size: 64
  early_stopping_patience: 20
  validation_split: 0.15

cross_validation: # Optional - only used in Optuna mode
  enabled: false        # Set to true to enable k-fold cross-validation during hyperparameter search
  n_folds: 5            # Number of folds for cross-validation
  stratified: false     # Use stratified k-fold (attempts to preserve covariate distribution)
  random_state: 42      # Random seed for reproducibility

optuna: # Optional
  n_trials: 100
  n_jobs: -1            # Number of parallel trials (-1 for auto: 1 per GPU or 1 for CPU)
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

device:
  gpu: true

paths:
  data_dir: "path/to/data/"
  output_dir: "path/to/output/"
```

#### Variance Modelling Options

The package supports three variance modelling approaches:

1. **Two Heads (Default, Recommended)**: The decoder has separate heads for mean and variance, allowing fully learnable variance
2. **Global Learnable**: A single learnable parameter shared across all features
3. **Covariate-Specific**: A small network that maps covariates to variance (requires `varnet_hidden_dim`)

#### Cross-Validation (Optuna Mode Only)

When using Optuna for hyperparameter optimisation, you can enable k-fold cross-validation for more robust hyperparameter selection:

- Standard k-fold: Randomly splits data into k folds
- Stratified k-fold: Attempts to preserve covariate distributions across folds

**Note:** Cross-validation increases computation time but provides more reliable hyperparameter estimates, especially with limited data.

#### Command Line Arguments

```
Required argument:
  --config PATH              Path to configuration file

Optional arguments:
  --mode TYPE                Training mode: direct or optuna (default: direct)
  --data_dir PATH            Override data directory specified in config file
  --output_dir PATH          Override output directory specified in config file
  --gpu                      Use GPU for training if available
  --suffix TEXT              Suffix for the output folder's name
```
The package provides two training options:
- **direct** (default): Train with parameters specified in config file
- **optuna**: Perform hyperparameter optimisation using the [Optuna](https://optuna.org/) framework

### Inference Configuration
For inference, you can either:
1. Specify required directories directly using `--data_dir` and `--output_dir`
2. Use a config file with the `--config` option

If using a config file, only these sections are required:

```yaml
paths:
  data_dir: "path/to/data/"     # Directory containing input data files
  output_dir: "path/to/output/" # Directory for storing results
```

Note: Command line arguments (`--data_dir`, `--output_dir`) will override the corresponding values in the config file if both are provided.

#### Bootstrap vs Simple Sampling
The package offers two analysis modes for inference:

##### Simple Sampling (Default)
* Fast computation with basic statistics
* Provides conditional means and variances without confidence intervals
* Outputs: means and variances only

##### Bootstrap Analysis (Optional)
* Provides robust statistical estimates with confidence intervals
* More computationally intensive but offers uncertainty quantification
* Outputs: means, variances, and confidence intervals
* Enable with `--bootstrap` flag

#### Command Line Arguments

```
Required arguments (either --config OR both --data_dir and --output_dir must be provided):
  --model_path PATH          Path to trained model checkpoint (required)
  --config PATH              Path to configuration file
  --data_dir PATH            Directory containing input data (required if --config not provided)
  --output_dir PATH          Directory for output files (required if --config not provided)

Optional arguments:
  --num_samples INT          Number of samples for analysis (default: 1000)
  --bootstrap                Enable bootstrap analysis with confidence intervals
  --no-bootstrap             Use simple sampling without bootstrap (default)
  --num_bootstraps INT       Number of bootstrap iterations (default: 1000, only used with --bootstrap)
  --confidence_level FLOAT   Confidence level for bootstrap CIs (default: 0.95, only used with --bootstrap)
  --prediction_type TYPE     Method for prediction: covariate or dual_input (default: covariate)
  --gpu                      Use GPU for inference if available
  --summary_report           Generate summary reports (feature importance and sensitivity analysis)
  --suffix TEXT              Suffix for the output folder's name
```

The package provides two prediction modes:
- **Covariate-based (Default, Primary approach)**: Generates normative predictions directly from demographic and clinical variables
- **Dual-input (For comparison only)**: Uses both observed data and covariates for prediction, implemented solely for methodological comparison purposes (see our [preprint](https://www.biorxiv.org/content/10.1101/2025.01.05.631276v1) for detailed discussion)

## Output Directory Structure

The package organises all outputs in a consistent directory structure. The structure varies depending on whether bootstrap analysis is enabled:

### With Simple Sampling (Default)

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
    ├── total_variances.csv             # Total variance (within + between) for each combination
    ├── within_variances.csv            # Within-group variance component
    ├── between_variances.csv           # Between-group variance component
    │
    ├── feature_variability.csv         # Variability of each feature across covariates (optional)
    ├── covariate_sensitivity.csv       # Impact of each covariate on predictions (optional)
    └── summary_statistics.xlsx         # Overall statistical summary of results (optional)
```

### With Bootstrap Analysis (--bootstrap)
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
    ├── means.csv                       # Mean predictions for each covariate combination
    ├── total_variances.csv             # Total variance (within + between) for each combination
    ├── within_variances.csv            # Within-group variance component
    ├── between_variances.csv           # Between-group variance component
    │
    ├── ci_means_lower.csv              # Lower confidence interval bounds for means
    ├── ci_means_upper.csv              # Upper confidence interval bounds for means
    ├── ci_variances_lower.csv          # Lower CI bounds for within-group variances
    ├── ci_variances_upper.csv          # Upper CI bounds for within-group variances
    ├── ci_total_variances_lower.csv    # Lower CI bounds for total variances
    ├── ci_total_variances_upper.csv    # Upper CI bounds for total variances
    │
    ├── feature_variability.csv         # Variability of each feature across covariates (optional)
    ├── covariate_sensitivity.csv       # Impact of each covariate on predictions (optional)
    └── summary_statistics.xlsx         # Overall statistical summary of results (optional)
```

### Results File Descriptions

#### Simple Sampling Results (Default)
- `means.csv`: Mean predictions for each covariate combination
- `total_variances.csv`: Total variance for each covariate combination
- `within_variances.csv`: Within-group variance component
- `between_variances.csv`: Between-group variance component

#### Bootstrap Analysis Results (--bootstrap)
- `means.csv`: Average predictions for each covariate combination across bootstrap samples
- `total_variances.csv`: Total variance for each covariate combination
- `within_variances.csv`: Within-group variance component
- `between_variances.csv`: Between-group variance component
- `ci_means_lower.csv`: Lower bound of confidence interval for mean predictions
- `ci_means_upper.csv`: Upper bound of confidence interval for mean predictions
- `ci_variances_lower.csv`: Lower bound of CI for within-group variances
- `ci_variances_upper.csv`: Upper bound of CI for within-group variances
- `ci_total_variances_lower.csv`: Lower bound of CI for total variances
- `ci_total_variances_upper.csv`: Upper bound of CI for total variances

#### Variance Components

The package decomposes the total variance into two components using the law of total variance:

- **Within-group variance**: Average variance from the decoder's predicted distributions across latent samples
- **Between-group variance**: Variance of the predicted means across latent samples  
- **Total variance**: Sum of within-group and between-group variances

#### Feature Analysis (Optional, with --summary_report)
- `feature_variability.csv`: Measures how much each feature varies across different covariate combinations
- `covariate_sensitivity.csv`: Quantifies how much each covariate influences the predictions
- `summary_statistics.xlsx`: Comprehensive statistical summary including:
  - Global means and standard deviations
  - Quantile distributions (25th, 50th, 75th percentiles)
  - Summary statistics for both means and variances

## Advanced Features

### Hyperparameter Optimisation with Cross-Validation

For small to medium-sized datasets, enabling cross-validation during Optuna optimisation can provide more reliable hyperparameter estimates:

```yaml
cross_validation:
  enabled: true
  n_folds: 5
  stratified: true  # Attempts to preserve covariate distributions
  random_state: 42
```

This approach is particularly useful when:
- Your dataset is relatively small
- You want to ensure hyperparameters generalise well
- You have sufficient computational resources (CV increases runtime)

### Parallel Training with Optuna

The package supports parallel hyperparameter search on multi-GPU systems:

```yaml
optuna:
  n_jobs: -1  # Auto-detect: uses 1 trial per GPU, or 1 for CPU
  # or specify manually:
  # n_jobs: 4  # Run 4 trials in parallel
```

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