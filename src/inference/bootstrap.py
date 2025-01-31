import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def generate_bootstrap_stats_by_covariates(model, covariates, covariates_df, feature_cols, config, 
                                         num_samples=1000, num_bootstraps=1000, confidence_level=0.95):
    """
    Generate bootstrap statistics using vectorized operations.
    """
    device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    try:
        covariates_tensor = torch.FloatTensor(covariates).to(device)
        results_dir = Path(config['paths']['output_dir']) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        bootstrap_means_list = []
        bootstrap_variances_list = []
        ci_means_list = []
        ci_variances_list = []
        
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        logger.info(f"Starting bootstrap analysis for {len(covariates_tensor)} covariate combinations")
        
        for idx, covariate in enumerate(covariates_tensor):
            # Generate all latent vectors at once
            z = torch.randn(num_samples, model.latent_dim, device=device)
            covariate_expanded = covariate.unsqueeze(0).expand(num_samples, -1)
            
            # Generate all samples in one batch
            with torch.no_grad():
                samples = model.decode(z, covariate_expanded).sample().cpu().numpy()
            
            # Generate all bootstrap indices at once
            bootstrap_indices = np.random.choice(
                num_samples, 
                size=(num_bootstraps, num_samples), 
                replace=True
            )
            
            # Vectorized computation of bootstrap statistics
            bootstrap_means = np.zeros((num_bootstraps, samples.shape[1]))
            bootstrap_variances = np.zeros((num_bootstraps, samples.shape[1]))
            
            for i in range(num_bootstraps):
                bootstrap_sample = samples[bootstrap_indices[i]]
                bootstrap_means[i] = np.mean(bootstrap_sample, axis=0)
                bootstrap_variances[i] = np.var(bootstrap_sample, axis=0)
            
            # Calculate summary statistics
            mean_of_bootstrap_means = np.mean(bootstrap_means, axis=0)
            mean_of_bootstrap_variances = np.mean(bootstrap_variances, axis=0)
            
            # Calculate confidence intervals using vectorized operations
            ci_means = np.percentile(bootstrap_means, [lower_percentile, upper_percentile], axis=0)
            ci_variances = np.percentile(bootstrap_variances, [lower_percentile, upper_percentile], axis=0)
            
            bootstrap_means_list.append(mean_of_bootstrap_means)
            bootstrap_variances_list.append(mean_of_bootstrap_variances)
            ci_means_list.append(ci_means)
            ci_variances_list.append(ci_variances)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(covariates_tensor)} covariate combinations")
        
        # Convert lists to arrays
        bootstrap_means_array = np.array(bootstrap_means_list)
        bootstrap_variances_array = np.array(bootstrap_variances_list)
        ci_means_array = np.array(ci_means_list)
        ci_variances_array = np.array(ci_variances_list)
        
        # Create DataFrames
        results = {
            'bootstrapped_means.csv': pd.concat([
                covariates_df,
                pd.DataFrame(bootstrap_means_array, columns=feature_cols)
            ], axis=1),
            'bootstrapped_variances.csv': pd.concat([
                covariates_df,
                pd.DataFrame(bootstrap_variances_array, columns=feature_cols)
            ], axis=1),
            'ci_means_lower.csv': pd.concat([
                covariates_df,
                pd.DataFrame([ci[0] for ci in ci_means_array], columns=feature_cols)
            ], axis=1),
            'ci_means_upper.csv': pd.concat([
                covariates_df,
                pd.DataFrame([ci[1] for ci in ci_means_array], columns=feature_cols)
            ], axis=1),
            'ci_variances_lower.csv': pd.concat([
                covariates_df,
                pd.DataFrame([ci[0] for ci in ci_variances_array], columns=feature_cols)
            ], axis=1),
            'ci_variances_upper.csv': pd.concat([
                covariates_df,
                pd.DataFrame([ci[1] for ci in ci_variances_array], columns=feature_cols)
            ], axis=1)
        }
        
        # Save results
        for filename, df in results.items():
            df.to_csv(results_dir / filename, index=False)
            
        logger.info("Bootstrap analysis completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Bootstrap analysis failed: {str(e)}")
        raise

def generate_bootstrap_stats_from_encoded(model, features, covariates, covariates_raw, feature_cols, config, 
                                        num_samples=1000, num_bootstraps=1000, confidence_level=0.95):
    """
    Generate bootstrap statistics from encoded features using vectorized operations.
    """
    device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    try:
        features_tensor = torch.FloatTensor(features.values).to(device)
        covariates_tensor = torch.FloatTensor(covariates).to(device)
        results_dir = Path(config['paths']['output_dir']) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        bootstrap_means_list = []
        bootstrap_variances_list = []
        ci_means_list = []
        ci_variances_list = []
        encoded_z_list = []
        
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        for idx, (feature, covariate) in enumerate(zip(features_tensor, covariates_tensor)):
            with torch.no_grad():
                # Encode feature to get latent distribution parameters
                mu_z, logvar_z = model.encode(feature.unsqueeze(0), covariate.unsqueeze(0))
                encoded_z_list.append(mu_z.cpu().numpy())
                
                # Generate all z samples at once using reparameterization
                std = torch.exp(0.5 * logvar_z)
                z = mu_z + std * torch.randn(num_samples, model.latent_dim, device=device)
                covariate_expanded = covariate.unsqueeze(0).expand(num_samples, -1)
                
                # Generate all reconstructions in one batch
                samples = model.decode(z, covariate_expanded).sample().cpu().numpy()
            
            # Generate all bootstrap indices at once
            bootstrap_indices = np.random.choice(
                num_samples, 
                size=(num_bootstraps, num_samples), 
                replace=True
            )
            
            # Vectorized computation of bootstrap statistics
            bootstrap_means = np.zeros((num_bootstraps, samples.shape[1]))
            bootstrap_variances = np.zeros((num_bootstraps, samples.shape[1]))
            
            for i in range(num_bootstraps):
                bootstrap_sample = samples[bootstrap_indices[i]]
                bootstrap_means[i] = np.mean(bootstrap_sample, axis=0)
                bootstrap_variances[i] = np.var(bootstrap_sample, axis=0)
            
            mean_of_bootstrap_means = np.mean(bootstrap_means, axis=0)
            mean_of_bootstrap_variances = np.mean(bootstrap_variances, axis=0)
            
            ci_means = np.percentile(bootstrap_means, [lower_percentile, upper_percentile], axis=0)
            ci_variances = np.percentile(bootstrap_variances, [lower_percentile, upper_percentile], axis=0)
            
            bootstrap_means_list.append(mean_of_bootstrap_means)
            bootstrap_variances_list.append(mean_of_bootstrap_variances)
            ci_means_list.append(ci_means)
            ci_variances_list.append(ci_variances)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(features_tensor)} samples")
        
        # Convert lists to arrays
        bootstrap_means_array = np.array(bootstrap_means_list)
        bootstrap_variances_array = np.array(bootstrap_variances_list)
        ci_means_array = np.array(ci_means_list)
        ci_variances_array = np.array(ci_variances_list)
        encoded_z_array = np.vstack(encoded_z_list)
        
        # Create and save results
        encoded_z_df = pd.DataFrame(
            encoded_z_array, 
            columns=[f'z_dim_{i}' for i in range(encoded_z_array.shape[1])]
        )
        
        results = {
            'bootstrapped_means.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(bootstrap_means_array, columns=feature_cols)
            ], axis=1),
            'bootstrapped_variances.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(bootstrap_variances_array, columns=feature_cols)
            ], axis=1),
            'ci_means_lower.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame([ci[0] for ci in ci_means_array], columns=feature_cols)
            ], axis=1),
            'ci_means_upper.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame([ci[1] for ci in ci_means_array], columns=feature_cols)
            ], axis=1),
            'ci_variances_lower.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame([ci[0] for ci in ci_variances_array], columns=feature_cols)
            ], axis=1),
            'ci_variances_upper.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame([ci[1] for ci in ci_variances_array], columns=feature_cols)
            ], axis=1),
            'latent_vectors.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                encoded_z_df
            ], axis=1)
        }
        
        # Save results
        for filename, df in results.items():
            df.to_csv(results_dir / filename, index=False)
            
        logger.info("Bootstrap analysis from encoded features completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Bootstrap analysis from encoded features failed: {str(e)}")
        raise
        
def compute_feature_importance(bootstrap_results, covariates_df, feature_cols, config):
    """
    Compute feature importance based on bootstrap results.
    
    Args:
        bootstrap_results: Dictionary containing DataFrames with bootstrap results
        covariates_df: DataFrame containing original covariates
        config: Configuration dictionary
    """
    try:
        means_df = bootstrap_results['bootstrapped_means.csv']
        means_features = means_df[feature_cols]
        
        # Calculate feature variability across different covariate combinations
        feature_variability = means_features.std()
        
        # Calculate feature sensitivity to different covariates
        covariate_sensitivity = {}
        
        # Create a DataFrame that combines covariates with features
        analysis_df = pd.concat([covariates_df.reset_index(drop=True), means_features], axis=1)
        
        for covariate in covariates_df.columns:
            groups = analysis_df.groupby(covariate)
            # Calculate differences only for feature columns
            max_diff = groups[feature_cols].mean().max() - groups[feature_cols].mean().min()
            covariate_sensitivity[covariate] = max_diff
            
        results_dir = Path(config['paths']['output_dir']) / 'results'
        
        pd.DataFrame({
            'feature_variability': feature_variability
        }).to_csv(results_dir / 'feature_variability.csv')
        
        pd.DataFrame(covariate_sensitivity).to_csv(results_dir / 'covariate_sensitivity.csv')
        
        logger.info("Feature importance analysis completed")
        return feature_variability, covariate_sensitivity
        
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {str(e)}")
        raise

def generate_summary_statistics(bootstrap_results, feature_cols, config):
    """Generate summary statistics from bootstrap results."""
    try:
        means_df = bootstrap_results['bootstrapped_means.csv']
        means_features = means_df[feature_cols]
        variances_df = bootstrap_results['bootstrapped_variances.csv']
        variances_features = variances_df[feature_cols]
        
        summary_stats = {
            'mean_statistics': {
                'global_mean': means_features.mean(),
                'global_std': means_features.std(),
                'quantiles': means_features.quantile([0.25, 0.5, 0.75])
            },
            'variance_statistics': {
                'global_mean': variances_features.mean(),
                'global_std': variances_features.std(),
                'quantiles': variances_features.quantile([0.25, 0.5, 0.75])
            }
        }
        
        results_dir = Path(config['paths']['output_dir']) / 'results'
        
        with pd.ExcelWriter(results_dir / 'summary_statistics.xlsx') as writer:
            for stat_type, stats in summary_stats.items():
                for stat_name, stat_value in stats.items():
                    stat_value.to_excel(writer, sheet_name=f'{stat_type}_{stat_name}')
        
        logger.info("Summary statistics generated successfully")
        return summary_stats
        
    except Exception as e:
        logger.error(f"Summary statistics generation failed: {str(e)}")
        raise