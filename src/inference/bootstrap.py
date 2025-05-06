import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def generate_bootstrap_stats_by_covariates(model, covariates, covariates_df, feature_cols, config, 
                                         num_samples=1000, num_bootstraps=1000, batch_size=10, confidence_level=0.95):
    """
    Generate bootstrap statistics using the repeat_interleave batch processing method.
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
        bootstrap_medians_list = []
        bootstrap_iqrs_list = []
        ci_means_list = []
        ci_variances_list = []
        ci_medians_list = []
        ci_iqrs_list = []
        
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        num_covariates = len(covariates_tensor)
        logger.info(f"Starting bootstrap analysis for {num_covariates} covariate combinations using repeat_interleave method")
        
        # Process in batches if batch_size > 0
        if batch_size > 0:
            all_samples = []
            for i in range(0, num_covariates, batch_size):
                batch_covariates = covariates_tensor[i:i+batch_size]
                batch_size_actual = len(batch_covariates)
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(num_covariates+batch_size-1)//batch_size}, covariates {i} to {min(i+batch_size-1, num_covariates-1)}")
                
                batch_expanded = batch_covariates.repeat_interleave(num_samples, dim=0)
                
                z = torch.randn(len(batch_expanded), model.latent_dim, device=device)
                
                with torch.no_grad():
                    samples_flat = model.decode(z, batch_expanded).sample().cpu().numpy()
                
                samples_batch = samples_flat.reshape(batch_size_actual, num_samples, -1)
                all_samples.append(samples_batch)
            
            samples_all = np.concatenate(all_samples, axis=0)
            
        else:
            # Process all covariates in a single batch if possible
            logger.info("Processing all covariates in a single batch")
            covariates_expanded = covariates_tensor.repeat_interleave(num_samples, dim=0)
            z = torch.randn(len(covariates_expanded), model.latent_dim, device=device)
            
            with torch.no_grad():
                samples_flat = model.decode(z, covariates_expanded).sample().cpu().numpy()
            
            samples_all = samples_flat.reshape(num_covariates, num_samples, -1)
            
        # Vectorized calculation of bootstrap statistics
        num_features = samples_all.shape[2]
        
        bootstrap_means = np.zeros((num_covariates, num_bootstraps, num_features))
        bootstrap_variances = np.zeros((num_covariates, num_bootstraps, num_features))
        bootstrap_medians = np.zeros((num_covariates, num_bootstraps, num_features))
        bootstrap_iqrs = np.zeros((num_covariates, num_bootstraps, num_features))
        
        # Generate bootstrap indices for all covariates at once
        all_bootstrap_indices = np.random.choice(
            num_samples, 
            size=(num_bootstraps, num_samples), 
            replace=True
        )
        
        logger.info("Generating bootstrap statistics...")
        
        # Process each covariate's samples
        for idx in range(num_covariates):
            if (idx + 1) % 10 == 0:
                logger.info(f"Processing bootstrap statistics for covariate {idx + 1}/{num_covariates}")
                
            samples = samples_all[idx]
            
            for i in range(num_bootstraps):
                bootstrap_sample = samples[all_bootstrap_indices[i]]
                bootstrap_means[idx, i] = np.mean(bootstrap_sample, axis=0)
                bootstrap_variances[idx, i] = np.var(bootstrap_sample, axis=0)
                bootstrap_medians[idx, i] = np.median(bootstrap_sample, axis=0)
                q75 = np.percentile(bootstrap_sample, 75, axis=0)
                q25 = np.percentile(bootstrap_sample, 25, axis=0)
                bootstrap_iqrs[idx, i] = q75 - q25
        
        mean_of_bootstrap_means = np.mean(bootstrap_means, axis=1)
        mean_of_bootstrap_variances = np.mean(bootstrap_variances, axis=1)
        mean_of_bootstrap_medians = np.mean(bootstrap_medians, axis=1)
        mean_of_bootstrap_iqrs = np.mean(bootstrap_iqrs, axis=1)
        
        ci_means = np.percentile(bootstrap_means, [lower_percentile, upper_percentile], axis=1)
        ci_variances = np.percentile(bootstrap_variances, [lower_percentile, upper_percentile], axis=1)
        ci_medians = np.percentile(bootstrap_medians, [lower_percentile, upper_percentile], axis=1)
        ci_iqrs = np.percentile(bootstrap_iqrs, [lower_percentile, upper_percentile], axis=1)
        
        results = {
            'bootstrapped_means.csv': pd.concat([
                covariates_df,
                pd.DataFrame(mean_of_bootstrap_means, columns=feature_cols)
            ], axis=1),
            'bootstrapped_variances.csv': pd.concat([
                covariates_df,
                pd.DataFrame(mean_of_bootstrap_variances, columns=feature_cols)
            ], axis=1),
            'bootstrapped_medians.csv': pd.concat([
                covariates_df,
                pd.DataFrame(mean_of_bootstrap_medians, columns=feature_cols)
            ], axis=1),
            'bootstrapped_iqrs.csv': pd.concat([
                covariates_df,
                pd.DataFrame(mean_of_bootstrap_iqrs, columns=feature_cols)
            ], axis=1),
            'ci_means_lower.csv': pd.concat([
                covariates_df,
                pd.DataFrame(ci_means[0], columns=feature_cols)
            ], axis=1),
            'ci_means_upper.csv': pd.concat([
                covariates_df,
                pd.DataFrame(ci_means[1], columns=feature_cols)
            ], axis=1),
            'ci_variances_lower.csv': pd.concat([
                covariates_df,
                pd.DataFrame(ci_variances[0], columns=feature_cols)
            ], axis=1),
            'ci_variances_upper.csv': pd.concat([
                covariates_df,
                pd.DataFrame(ci_variances[1], columns=feature_cols)
            ], axis=1),
            'ci_medians_lower.csv': pd.concat([
                covariates_df,
                pd.DataFrame(ci_medians[0], columns=feature_cols)
            ], axis=1),
            'ci_medians_upper.csv': pd.concat([
                covariates_df,
                pd.DataFrame(ci_medians[1], columns=feature_cols)
            ], axis=1),
            'ci_iqrs_lower.csv': pd.concat([
                covariates_df,
                pd.DataFrame(ci_iqrs[0], columns=feature_cols)
            ], axis=1),
            'ci_iqrs_upper.csv': pd.concat([
                covariates_df,
                pd.DataFrame(ci_iqrs[1], columns=feature_cols)
            ], axis=1)
        }
        
        for filename, df in results.items():
            df.to_csv(results_dir / filename, index=False)
            
        logger.info("Vectorized bootstrap analysis completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Batch bootstrap analysis failed: {str(e)}")
        raise

def generate_bootstrap_stats_from_encoded(model, features, covariates, covariates_raw, feature_cols, config, 
                                        num_samples=1000, num_bootstraps=1000, batch_size=10, confidence_level=0.95):
    """
    Generate bootstrap statistics from encoded features using vectorized processing.
    """
    device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    try:
        features_tensor = torch.FloatTensor(features.values).to(device)
        covariates_tensor = torch.FloatTensor(covariates).to(device)
        results_dir = Path(config['paths']['output_dir']) / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        num_samples_total = len(features_tensor)
        
        logger.info(f"Starting bootstrap analysis from encoded features using vectorized method")
        
        # Process in batches for encoding
        if batch_size > 0:
            all_mu_z = []
            all_logvar_z = []
            
            num_batches = (num_samples_total + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples_total)
                batch_features = features_tensor[start_idx:end_idx]
                batch_covariates = covariates_tensor[start_idx:end_idx]
                
                logger.info(f"Encoding batch {batch_idx+1}/{num_batches}, samples {start_idx} to {end_idx-1}")
                
                with torch.no_grad():
                    mu_z, logvar_z = model.encode(batch_features, batch_covariates)
                    all_mu_z.append(mu_z)
                    all_logvar_z.append(logvar_z)
            
            mu_z_all = torch.cat(all_mu_z, dim=0)
            logvar_z_all = torch.cat(all_logvar_z, dim=0)
        else:
            with torch.no_grad():
                mu_z_all, logvar_z_all = model.encode(features_tensor, covariates_tensor)
        
        encoded_z_np = mu_z_all.cpu().numpy()
        
        logger.info("Generating samples from encoded latent vectors in batches")
        
        # Pre-allocate arrays for results
        num_samples_per_z = num_samples
        num_features = features.shape[1]
        
        all_samples = []
        
        # Process latent vectors in batches for sampling
        batch_size_sampling = min(batch_size if batch_size > 0 else num_samples_total, 100)
        
        for i in range(0, num_samples_total, batch_size_sampling):
            end_idx = min(i + batch_size_sampling, num_samples_total)
            batch_mu_z = mu_z_all[i:end_idx]
            batch_logvar_z = logvar_z_all[i:end_idx]
            batch_covariates = covariates_tensor[i:end_idx]
            
            batch_size_actual = len(batch_mu_z)
            logger.info(f"Sampling batch {i//batch_size_sampling + 1}/{(num_samples_total+batch_size_sampling-1)//batch_size_sampling}, samples {i} to {end_idx-1}")
            
            batch_samples = []
            
            for j in range(batch_size_actual):
                mu_z = batch_mu_z[j]
                logvar_z = batch_logvar_z[j]
                covariate = batch_covariates[j]
                
                with torch.no_grad():
                    std = torch.exp(0.5 * logvar_z)
                    
                    eps = torch.randn(num_samples_per_z, model.latent_dim, device=device)
                    z = mu_z.unsqueeze(0) + std.unsqueeze(0) * eps
                    
                    covariate_expanded = covariate.repeat(num_samples_per_z, 1)
                    
                    sample_batch = model.decode(z, covariate_expanded).sample().cpu().numpy()
                    batch_samples.append(sample_batch)
            
            all_samples.extend(batch_samples)
        
        samples_all = np.array(all_samples)
        
        # Generate bootstrap indices for all samples at once
        all_bootstrap_indices = np.random.choice(
            num_samples_per_z, 
            size=(num_bootstraps, num_samples_per_z), 
            replace=True
        )
        
        # Vectorized calculation of bootstrap statistics
        bootstrap_means = np.zeros((num_samples_total, num_bootstraps, num_features))
        bootstrap_variances = np.zeros((num_samples_total, num_bootstraps, num_features))
        
        logger.info("Computing bootstrap statistics")
        
        # Process bootstrap for each sample
        for i in range(num_samples_total):
            if (i + 1) % 100 == 0:
                logger.info(f"Computing bootstrap statistics for sample {i + 1}/{num_samples_total}")
                
            samples = samples_all[i]
            
            for j in range(num_bootstraps):
                bootstrap_sample = samples[all_bootstrap_indices[j]]
                bootstrap_means[i, j] = np.mean(bootstrap_sample, axis=0)
                bootstrap_variances[i, j] = np.var(bootstrap_sample, axis=0)
        
        mean_of_bootstrap_means = np.mean(bootstrap_means, axis=1)
        mean_of_bootstrap_variances = np.mean(bootstrap_variances, axis=1)
        
        ci_means = np.percentile(bootstrap_means, [lower_percentile, upper_percentile], axis=1)
        ci_variances = np.percentile(bootstrap_variances, [lower_percentile, upper_percentile], axis=1)
        
        encoded_z_df = pd.DataFrame(
            encoded_z_np, 
            columns=[f'z_dim_{i}' for i in range(encoded_z_np.shape[1])]
        )
        
        results = {
            'bootstrapped_means.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(mean_of_bootstrap_means, columns=feature_cols)
            ], axis=1),
            'bootstrapped_variances.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(mean_of_bootstrap_variances, columns=feature_cols)
            ], axis=1),
            'ci_means_lower.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(ci_means[0], columns=feature_cols)
            ], axis=1),
            'ci_means_upper.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(ci_means[1], columns=feature_cols)
            ], axis=1),
            'ci_variances_lower.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(ci_variances[0], columns=feature_cols)
            ], axis=1),
            'ci_variances_upper.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(ci_variances[1], columns=feature_cols)
            ], axis=1),
            'latent_vectors.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                encoded_z_df
            ], axis=1)
        }
        
        for filename, df in results.items():
            df.to_csv(results_dir / filename, index=False)
            
        logger.info("Vectorized bootstrap analysis from encoded features completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Vectorized bootstrap analysis from encoded features failed: {str(e)}")
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
        
        feature_variability = means_features.std()
        covariate_sensitivity = {}
        
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