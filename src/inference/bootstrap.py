import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_calibrator(calibrator_path):
    """Load improved variance calibrator if available."""
    if calibrator_path is not None:
        try:
            from ..training.calibration import VarianceCalibrator
            calibrator = VarianceCalibrator()
            calibrator.load(calibrator_path)
            logger.info(f"Loaded improved variance calibrator ({calibrator.method})")
            return calibrator
        except Exception as e:
            logger.warning(f"Failed to load variance calibrator: {str(e)}")
            return None
    else:
        logger.info("No variance calibrator found - using uncalibrated variances")
        return None

def apply_calibration(variances, calibrator):
    """Apply variance calibration if calibrator is available."""
    if calibrator is None:
        return variances
    
    try:
        return calibrator.calibrate(variances)
    except Exception as e:
        logger.warning(f"Calibration failed: {str(e)}, using uncalibrated variances")
        return variances

def generate_simple_stats_by_covariates(model, covariates, covariates_df, feature_cols, config, 
                                        num_iterations=1000, suffix=''):
    """
    Generate simple statistics using the total variance formula with calibration support.
    """
    device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load calibrator
    calibrator_path = config.get('paths', {}).get('calibrator_path', None)
    calibrator = load_calibrator(calibrator_path)
    
    try:
        covariates_tensor = torch.FloatTensor(covariates).to(device)
        results_dir = Path(config['paths']['output_dir']) / f'results{suffix}'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        means_list = []
        total_variances_list = []
        within_variances_list = []
        between_variances_list = []
        calibrated_total_variances_list = []
        calibrated_within_variances_list = []
        
        logger.info(f"Starting simple sampling analysis for {len(covariates_tensor)} covariate combinations")
        logger.info(f"Using {num_iterations} iterations per covariate combination")
        
        for idx, covariate in enumerate(covariates_tensor):
            mu_k_values = []
            sigma_k_values = []
            
            # Generate K iterations to get mu_k and sigma_k from model output distributions
            for k in range(num_iterations):
                z = torch.randn(1, model.latent_dim, device=device)
                covariate_expanded = covariate.unsqueeze(0)
                
                with torch.no_grad():
                    # Get the output distribution parameters
                    output_dist = model.decode(z, covariate_expanded)
                    mu_k = output_dist.loc.squeeze(0).cpu().numpy()
                    sigma_k = output_dist.scale.pow(2).squeeze(0).cpu().numpy()
                
                mu_k_values.append(mu_k)
                sigma_k_values.append(sigma_k)
            
            # Convert to arrays for computation
            mu_k_array = np.array(mu_k_values)       # Shape: (num_iterations, num_features)
            sigma_k_array = np.array(sigma_k_values) # Shape: (num_iterations, num_features)
            
            # Calculate overall mean: (1/K)sum(mu_k)
            overall_mean = np.mean(mu_k_array, axis=0)
            
            # Within-group variance: (1/K)sum(sigma^2_k)
            within_variance = np.mean(sigma_k_array, axis=0)
            
            # Between-group variance: (1/K)sum((mu_k - overall_mean)^2)
            between_variance = np.var(mu_k_array, axis=0)
            
            # Total variance using the formula
            total_variance = within_variance + between_variance
            
            # Apply calibration
            if calibrator is not None:
                calibrated_within_variance = apply_calibration(within_variance, calibrator)
                calibrated_total_variance = calibrated_within_variance + between_variance
            
            # Store results
            means_list.append(overall_mean)
            total_variances_list.append(total_variance)
            within_variances_list.append(within_variance)
            between_variances_list.append(between_variance)
            
            if calibrator is not None:
                calibrated_total_variances_list.append(calibrated_total_variance)
                calibrated_within_variances_list.append(calibrated_within_variance)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(covariates_tensor)} covariate combinations")
        
        # Convert to arrays
        means_array = np.array(means_list)
        total_variances_array = np.array(total_variances_list)
        within_variances_array = np.array(within_variances_list)
        between_variances_array = np.array(between_variances_list)
        
        results = {
            'means.csv': pd.concat([
                covariates_df.reset_index(drop=True),
                pd.DataFrame(means_array, columns=feature_cols)
            ], axis=1),
            'total_variances.csv': pd.concat([
                covariates_df.reset_index(drop=True),
                pd.DataFrame(total_variances_array, columns=feature_cols)
            ], axis=1),
            'within_variances.csv': pd.concat([
                covariates_df.reset_index(drop=True),
                pd.DataFrame(within_variances_array, columns=feature_cols)
            ], axis=1),
            'between_variances.csv': pd.concat([
                covariates_df.reset_index(drop=True),
                pd.DataFrame(between_variances_array, columns=feature_cols)
            ], axis=1)
        }
        
        if calibrator is not None:
            calibrated_total_variances_array = np.array(calibrated_total_variances_list)
            calibrated_within_variances_array = np.array(calibrated_within_variances_list)
        
            results["calibrated_total_variances.csv"] = pd.concat(
                [covariates_df, pd.DataFrame(calibrated_total_variances_list, columns=feature_cols)], axis=1
            )
            results["calibrated_within_variances.csv"] = pd.concat(
                [covariates_df, pd.DataFrame(calibrated_within_variances_list, columns=feature_cols)], axis=1
            )
        
        for filename, df in results.items():
            df.to_csv(results_dir / filename, index=False)
            
        logger.info("Simple sampling analysis completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"Simple sampling analysis failed: {str(e)}")
        raise

def generate_simple_stats_from_encoded(model, features, covariates, covariates_raw, feature_cols, config, 
                                       num_iterations=1000, suffix=''):
    """
    Generate simple statistics from encoded features with calibration support.
    """
    device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load calibrator
    calibrator_path = config.get('paths', {}).get('calibrator_path', None)
    calibrator = load_calibrator(calibrator_path)
    
    try:
        features_tensor = torch.FloatTensor(features.values).to(device)
        covariates_tensor = torch.FloatTensor(covariates).to(device)
        results_dir = Path(config['paths']['output_dir']) / f'results{suffix}'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        means_list = []
        total_variances_list = []
        within_variances_list = []
        between_variances_list = []
        calibrated_total_variances_list = []
        calibrated_within_variances_list = []
        encoded_z_list = []
        
        logger.info(f"Starting simple sampling from encoded features for {len(features_tensor)} samples")
        logger.info(f"Using {num_iterations} iterations per sample")
        
        for idx, (feature, covariate) in enumerate(zip(features_tensor, covariates_tensor)):
            with torch.no_grad():
                # Encode feature to get latent distribution parameters
                mu_z, logvar_z = model.encode(feature.unsqueeze(0), covariate.unsqueeze(0))
                encoded_z_list.append(mu_z.cpu().numpy())
                
                mu_k_values = []
                sigma_k_values = []
                
                # Generate K iterations using the encoded latent distribution
                for k in range(num_iterations):
                    # Sample one z from the encoded distribution
                    std = torch.exp(0.5 * logvar_z)
                    z = mu_z + std * torch.randn(1, model.latent_dim, device=device)
                    covariate_expanded = covariate.unsqueeze(0)
                    
                    # Get the output distribution parameters
                    output_dist = model.decode(z, covariate_expanded)
                    mu_k = output_dist.loc.squeeze(0).cpu().numpy()
                    sigma_k = output_dist.scale.pow(2).squeeze(0).cpu().numpy()
                    
                    mu_k_values.append(mu_k)
                    sigma_k_values.append(sigma_k)
            
            # Convert to arrays
            mu_k_array = np.array(mu_k_values)
            sigma_k_array = np.array(sigma_k_values)
            
            # Apply the total variance formula
            overall_mean = np.mean(mu_k_array, axis=0)
            within_variance = np.mean(sigma_k_array, axis=0)
            between_variance = np.var(mu_k_array, axis=0)
            total_variance = within_variance + between_variance
            
            # Apply calibration
            if calibrator is not None:
                calibrated_within_variance = apply_calibration(within_variance, calibrator)
                calibrated_total_variance = calibrated_within_variance + between_variance
            
            # Store results
            means_list.append(overall_mean)
            total_variances_list.append(total_variance)
            within_variances_list.append(within_variance)
            between_variances_list.append(between_variance)
            
            if calibrator is not None:
                calibrated_total_variances_list.append(calibrated_total_variance)
                calibrated_within_variances_list.append(calibrated_within_variance)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(features_tensor)} samples")
        
        # Convert to arrays
        means_array = np.array(means_list)
        total_variances_array = np.array(total_variances_list)
        within_variances_array = np.array(within_variances_list)
        between_variances_array = np.array(between_variances_list)
        encoded_z_array = np.vstack(encoded_z_list)
        
        encoded_z_df = pd.DataFrame(
            encoded_z_array, 
            columns=[f'z_dim_{i}' for i in range(encoded_z_array.shape[1])]
        )
        
        results = {
            'means.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(means_array, columns=feature_cols)
            ], axis=1),
            'total_variances.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(total_variances_array, columns=feature_cols)
            ], axis=1),
            'within_variances.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(within_variances_array, columns=feature_cols)
            ], axis=1),
            'between_variances.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(between_variances_array, columns=feature_cols)
            ], axis=1),
            'latent_vectors.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                encoded_z_df
            ], axis=1)
        }
        
        if calibrator is not None:
            calibrated_total_variances_array = np.array(calibrated_total_variances_list)
            calibrated_within_variances_array = np.array(calibrated_within_variances_list)
            
            results["calibrated_total_variances.csv"] = pd.concat(
                [covariates_raw, pd.DataFrame(calibrated_total_variances_list, columns=feature_cols)], axis=1
            )
            results["calibrated_within_variances.csv"] = pd.concat(
                [covariates_raw, pd.DataFrame(calibrated_within_variances_list, columns=feature_cols)], axis=1
            )
        
        for filename, df in results.items():
            df.to_csv(results_dir / filename, index=False)
            
        logger.info("Simple sampling analysis from encoded features completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Simple sampling analysis from encoded features failed: {str(e)}")
        raise

def generate_bootstrap_stats_by_covariates(model, covariates, covariates_df, feature_cols, config, 
                                           num_samples=1000, num_bootstraps=1000, confidence_level=0.95,
                                           suffix=''):
    """
    Generate bootstrap statistics with calibration support.
    """
    device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load improved calibrator
    calibrator_path = config.get('paths', {}).get('calibrator_path', None)
    calibrator = load_calibrator(calibrator_path)
    
    try:
        covariates_tensor = torch.FloatTensor(covariates).to(device)
        results_dir = Path(config['paths']['output_dir']) / f'results{suffix}'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        bootstrap_means_list = []
        bootstrap_variances_list = []
        total_variances_list = []
        within_variances_list = []
        between_variances_list = []
        calibrated_bootstrap_variances_list = []
        calibrated_total_variances_list = []
        ci_means_list = []
        ci_variances_list = []
        ci_total_variances_list = []
        ci_calibrated_variances_list = []
        ci_calibrated_total_variances_list = []
        
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        logger.info(f"Starting bootstrap analysis for {len(covariates_tensor)} covariate combinations")
        logger.info(f"Using {num_bootstraps} bootstrap samples with {num_samples} samples each")
        
        for idx, covariate in enumerate(covariates_tensor):
            # For each bootstrap iteration, collect the statistics
            bootstrap_means = []
            bootstrap_variances = []
            calibrated_bootstrap_variances = []
            
            for b in range(num_bootstraps):
                # Generate fresh samples for this bootstrap iteration
                mu_k_values = []
                sigma_k_values = []
                
                for s in range(num_samples):
                    # Generate a latent sample for each sample
                    z = torch.randn(1, model.latent_dim, device=device)
                    covariate_expanded = covariate.unsqueeze(0)
                    
                    with torch.no_grad():
                        output_dist = model.decode(z, covariate_expanded)
                        mu_k = output_dist.loc.squeeze(0).cpu().numpy()
                        sigma_k = output_dist.scale.pow(2).squeeze(0).cpu().numpy()
                    
                    mu_k_values.append(mu_k)
                    sigma_k_values.append(sigma_k)
                
                # Calculate statistics for this bootstrap iteration
                mu_k_array = np.array(mu_k_values)
                sigma_k_array = np.array(sigma_k_values)
                
                bootstrap_mean = np.mean(mu_k_array, axis=0)
                bootstrap_var = np.mean(sigma_k_array, axis=0)
                
                # Apply improved calibration to bootstrap variance
                calibrated_bootstrap_var = apply_calibration(bootstrap_var, calibrator)
                
                bootstrap_means.append(bootstrap_mean)
                bootstrap_variances.append(bootstrap_var)
                calibrated_bootstrap_variances.append(calibrated_bootstrap_var)
            
            # Convert to arrays for easier manipulation
            bootstrap_means = np.array(bootstrap_means)                    # Shape: (num_bootstraps, num_features)
            bootstrap_variances = np.array(bootstrap_variances)            # Shape: (num_bootstraps, num_features)
            calibrated_bootstrap_variances = np.array(calibrated_bootstrap_variances)
            
            # Calculate overall statistics across all bootstraps
            overall_mean = np.mean(bootstrap_means, axis=0)
            within_variance = np.mean(bootstrap_variances, axis=0)  # average sigma_k
            calibrated_within_variance = np.mean(calibrated_bootstrap_variances, axis=0)
            between_variance = np.var(bootstrap_means, axis=0)  # variance of means
            total_variance = within_variance + between_variance
            calibrated_total_variance = calibrated_within_variance + between_variance
            
            # Calculate confidence intervals
            ci_means = np.percentile(bootstrap_means, [lower_percentile, upper_percentile], axis=0)
            ci_variances = np.percentile(bootstrap_variances, [lower_percentile, upper_percentile], axis=0)
            ci_calibrated_variances = np.percentile(calibrated_bootstrap_variances, [lower_percentile, upper_percentile], axis=0)
            
            # Calculate total variance for each bootstrap and get CIs
            bootstrap_total_variances = bootstrap_variances + (bootstrap_means - overall_mean)**2
            calibrated_bootstrap_total_variances = calibrated_bootstrap_variances + (bootstrap_means - overall_mean)**2
            ci_total_variances = np.percentile(bootstrap_total_variances, [lower_percentile, upper_percentile], axis=0)
            ci_calibrated_total_variances = np.percentile(calibrated_bootstrap_total_variances, [lower_percentile, upper_percentile], axis=0)
            
            # Store results
            bootstrap_means_list.append(overall_mean)
            bootstrap_variances_list.append(within_variance)
            calibrated_bootstrap_variances_list.append(calibrated_within_variance)
            total_variances_list.append(total_variance)
            calibrated_total_variances_list.append(calibrated_total_variance)
            within_variances_list.append(within_variance)
            between_variances_list.append(between_variance)
            ci_means_list.append(ci_means)
            ci_variances_list.append(ci_variances)
            ci_calibrated_variances_list.append(ci_calibrated_variances)
            ci_total_variances_list.append(ci_total_variances)
            ci_calibrated_total_variances_list.append(ci_calibrated_total_variances)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(covariates_tensor)} covariate combinations")
        
        # Convert to arrays
        bootstrap_means_array = np.array(bootstrap_means_list)
        bootstrap_variances_array = np.array(bootstrap_variances_list)
        calibrated_bootstrap_variances_array = np.array(calibrated_bootstrap_variances_list)
        total_variances_array = np.array(total_variances_list)
        calibrated_total_variances_array = np.array(calibrated_total_variances_list)
        within_variances_array = np.array(within_variances_list)
        between_variances_array = np.array(between_variances_list)
        ci_means_array = np.array(ci_means_list)
        ci_variances_array = np.array(ci_variances_list)
        ci_calibrated_variances_array = np.array(ci_calibrated_variances_list)
        ci_total_variances_array = np.array(ci_total_variances_list)
        ci_calibrated_total_variances_array = np.array(ci_calibrated_total_variances_list)
        
        results = {
            'means.csv': pd.concat([
                covariates_df,
                pd.DataFrame(bootstrap_means_array, columns=feature_cols)
            ], axis=1),
            'total_variances.csv': pd.concat([
                covariates_df,
                pd.DataFrame(total_variances_array, columns=feature_cols)
            ], axis=1),
            'within_variances.csv': pd.concat([
                covariates_df,
                pd.DataFrame(within_variances_array, columns=feature_cols)
            ], axis=1),
            'between_variances.csv': pd.concat([
                covariates_df,
                pd.DataFrame(between_variances_array, columns=feature_cols)
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
                pd.DataFrame([ci[0] for ci in ci_calibrated_variances_array], columns=feature_cols)
            ], axis=1),
            'ci_variances_upper.csv': pd.concat([
                covariates_df,
                pd.DataFrame([ci[1] for ci in ci_calibrated_variances_array], columns=feature_cols)
            ], axis=1),
            'ci_total_variances_lower.csv': pd.concat([
                covariates_df,
                pd.DataFrame([ci[0] for ci in ci_calibrated_total_variances_array], columns=feature_cols)
            ], axis=1),
            'ci_total_variances_upper.csv': pd.concat([
                covariates_df,
                pd.DataFrame([ci[1] for ci in ci_calibrated_total_variances_array], columns=feature_cols)
            ], axis=1)
        }
        
        if calibrator is not None:
            results['calibrated_total_variances.csv'] = pd.concat([
                covariates_df,
                pd.DataFrame(calibrated_total_variances_array, columns=feature_cols)
            ], axis=1)
        
        for filename, df in results.items():
            df.to_csv(results_dir / filename, index=False)
            
        logger.info("Bootstrap analysis completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"Bootstrap analysis failed: {str(e)}")
        raise

def generate_bootstrap_stats_from_encoded(model, features, covariates, covariates_raw, feature_cols, config, 
                                          num_samples=1000, num_bootstraps=1000, confidence_level=0.95,
                                          suffix=''):
    """
    Generate bootstrap statistics from encoded features with calibration support.
    """
    device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load improved calibrator
    calibrator_path = config.get('paths', {}).get('calibrator_path', None)
    calibrator = load_calibrator(calibrator_path)
    
    try:
        features_tensor = torch.FloatTensor(features.values).to(device)
        covariates_tensor = torch.FloatTensor(covariates).to(device)
        results_dir = Path(config['paths']['output_dir']) / f'results{suffix}'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        bootstrap_means_list = []
        bootstrap_variances_list = []
        calibrated_bootstrap_variances_list = []
        total_variances_list = []
        calibrated_total_variances_list = []
        within_variances_list = []
        between_variances_list = []
        ci_means_list = []
        ci_variances_list = []
        ci_calibrated_variances_list = []
        ci_total_variances_list = []
        ci_calibrated_total_variances_list = []
        encoded_z_list = []
        
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        logger.info(f"Starting bootstrap analysis from encoded features for {len(features_tensor)} samples")
        logger.info(f"Using {num_bootstraps} bootstrap samples with {num_samples} samples each")
        
        for idx, (feature, covariate) in enumerate(zip(features_tensor, covariates_tensor)):
            with torch.no_grad():
                # Encode feature to get latent distribution parameters
                mu_z, logvar_z = model.encode(feature.unsqueeze(0), covariate.unsqueeze(0))
                encoded_z_list.append(mu_z.cpu().numpy())
                
                # Generate base samples for bootstrapping from the encoded distribution
                std = torch.exp(0.5 * logvar_z)
                z = mu_z + std * torch.randn(num_samples, model.latent_dim, device=device)
                covariate_expanded = covariate.unsqueeze(0).expand(num_samples, -1)
                
                # Sample from output distributions
                samples = model.decode(z, covariate_expanded).sample().cpu().numpy()
            
            # Generate bootstrap indices
            bootstrap_indices = np.random.choice(
                num_samples, 
                size=(num_bootstraps, num_samples), 
                replace=True
            )
            
            # Calculate bootstrap statistics
            bootstrap_means = np.zeros((num_bootstraps, samples.shape[1]))
            bootstrap_variances = np.zeros((num_bootstraps, samples.shape[1]))
            calibrated_bootstrap_variances = np.zeros((num_bootstraps, samples.shape[1]))
            
            for i in range(num_bootstraps):
                bootstrap_sample = samples[bootstrap_indices[i]]
                bootstrap_mean = np.mean(bootstrap_sample, axis=0)
                bootstrap_var = np.var(bootstrap_sample, axis=0)
                calibrated_bootstrap_var = apply_calibration(bootstrap_var, calibrator)
                
                bootstrap_means[i] = bootstrap_mean
                bootstrap_variances[i] = bootstrap_var
                calibrated_bootstrap_variances[i] = calibrated_bootstrap_var
            
            # Apply the total variance formula
            overall_mean = np.mean(bootstrap_means, axis=0)
            within_variance = np.mean(bootstrap_variances, axis=0)
            calibrated_within_variance = np.mean(calibrated_bootstrap_variances, axis=0)
            between_variance = np.var(bootstrap_means, axis=0)
            total_variance = within_variance + between_variance
            calibrated_total_variance = calibrated_within_variance + between_variance
            
            # Calculate confidence intervals
            ci_means = np.percentile(bootstrap_means, [lower_percentile, upper_percentile], axis=0)
            ci_variances = np.percentile(bootstrap_variances, [lower_percentile, upper_percentile], axis=0)
            ci_calibrated_variances = np.percentile(calibrated_bootstrap_variances, [lower_percentile, upper_percentile], axis=0)
            
            # Calculate confidence intervals for total variance
            bootstrap_total_variances = np.zeros((num_bootstraps, samples.shape[1]))
            calibrated_bootstrap_total_variances = np.zeros((num_bootstraps, samples.shape[1]))
            for i in range(num_bootstraps):
                bootstrap_total_variances[i] = bootstrap_variances[i] + \
                    (bootstrap_means[i] - overall_mean)**2
                calibrated_bootstrap_total_variances[i] = calibrated_bootstrap_variances[i] + \
                    (bootstrap_means[i] - overall_mean)**2
            
            ci_total_variances = np.percentile(bootstrap_total_variances, [lower_percentile, upper_percentile], axis=0)
            ci_calibrated_total_variances = np.percentile(calibrated_bootstrap_total_variances, [lower_percentile, upper_percentile], axis=0)
            
            # Store results
            bootstrap_means_list.append(overall_mean)
            bootstrap_variances_list.append(within_variance)
            calibrated_bootstrap_variances_list.append(calibrated_within_variance)
            total_variances_list.append(total_variance)
            calibrated_total_variances_list.append(calibrated_total_variance)
            within_variances_list.append(within_variance)
            between_variances_list.append(between_variance)
            ci_means_list.append(ci_means)
            ci_variances_list.append(ci_variances)
            ci_calibrated_variances_list.append(ci_calibrated_variances)
            ci_total_variances_list.append(ci_total_variances)
            ci_calibrated_total_variances_list.append(ci_calibrated_total_variances)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(features_tensor)} samples")
        
        # Convert to arrays
        bootstrap_means_array = np.array(bootstrap_means_list)
        bootstrap_variances_array = np.array(bootstrap_variances_list)
        calibrated_bootstrap_variances_array = np.array(calibrated_bootstrap_variances_list)
        total_variances_array = np.array(total_variances_list)
        calibrated_total_variances_array = np.array(calibrated_total_variances_list)
        within_variances_array = np.array(within_variances_list)
        between_variances_array = np.array(between_variances_list)
        ci_means_array = np.array(ci_means_list)
        ci_variances_array = np.array(ci_variances_list)
        ci_calibrated_variances_array = np.array(ci_calibrated_variances_list)
        ci_total_variances_array = np.array(ci_total_variances_list)
        ci_calibrated_total_variances_array = np.array(ci_calibrated_total_variances_list)
        encoded_z_array = np.vstack(encoded_z_list)
        
        encoded_z_df = pd.DataFrame(
            encoded_z_array, 
            columns=[f'z_dim_{i}' for i in range(encoded_z_array.shape[1])]
        )
        
        results = {
            'means.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(bootstrap_means_array, columns=feature_cols)
            ], axis=1),
            'total_variances.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(total_variances_array, columns=feature_cols)
            ], axis=1),
            'within_variances.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(within_variances_array, columns=feature_cols)
            ], axis=1),
            'between_variances.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(between_variances_array, columns=feature_cols)
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
                pd.DataFrame([ci[0] for ci in ci_calibrated_variances_array], columns=feature_cols)
            ], axis=1),
            'ci_variances_upper.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame([ci[1] for ci in ci_calibrated_variances_array], columns=feature_cols)
            ], axis=1),
            'ci_total_variances_lower.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame([ci[0] for ci in ci_calibrated_total_variances_array], columns=feature_cols)
            ], axis=1),
            'ci_total_variances_upper.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame([ci[1] for ci in ci_calibrated_total_variances_array], columns=feature_cols)
            ], axis=1),
            'latent_vectors.csv': pd.concat([
                covariates_raw.reset_index(drop=True),
                encoded_z_df
            ], axis=1)
        }
        
        if calibrator is not None:
            results['calibrated_total_variances.csv'] = pd.concat([
                covariates_raw.reset_index(drop=True),
                pd.DataFrame(calibrated_total_variances_array, columns=feature_cols)
            ], axis=1)
        
        for filename, df in results.items():
            df.to_csv(results_dir / filename, index=False)
            
        logger.info("Bootstrap analysis from encoded features completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"Bootstrap analysis from encoded features failed: {str(e)}")
        raise
        
def compute_feature_importance(results, covariates_df, feature_cols, config, suffix=''):
    """
    Compute feature importance based on results.
    """
    try:
        means_key = 'means.csv'
        means_df = results[means_key]
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
            
        results_dir = Path(config['paths']['output_dir']) / f'results{suffix}'
        
        pd.DataFrame({
            'feature_variability': feature_variability
        }).to_csv(results_dir / 'feature_variability.csv')
        
        pd.DataFrame(covariate_sensitivity).to_csv(results_dir / 'covariate_sensitivity.csv')
        
        logger.info("Feature importance analysis completed")
        return feature_variability, covariate_sensitivity
        
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {str(e)}")
        raise

def generate_summary_statistics(results, feature_cols, config, suffix=''):
    """Generate summary statistics from results."""
    try:
        means_key = 'means.csv'
        variances_key = 'total_variances.csv'
            
        means_df = results[means_key]
        means_features = means_df[feature_cols]
        variances_df = results[variances_key]
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
        
        # Add calibration information if available
        if 'calibrated_total_variances.csv' in results:
            calibrated_variances_df = results['calibrated_total_variances.csv']
            calibrated_variances_features = calibrated_variances_df[feature_cols]
            summary_stats['calibrated_variance_statistics'] = {
                'global_mean': calibrated_variances_features.mean(),
                'global_std': calibrated_variances_features.std(),
                'quantiles': calibrated_variances_features.quantile([0.25, 0.5, 0.75])
            }
        
        results_dir = Path(config['paths']['output_dir']) / f'results{suffix}'
        
        with pd.ExcelWriter(results_dir / 'summary_statistics.xlsx') as writer:
            for stat_type, stats in summary_stats.items():
                for stat_name, stat_value in stats.items():
                    stat_value.to_excel(writer, sheet_name=f'{stat_type}_{stat_name}')
        
        logger.info("Summary statistics generated successfully")
        return summary_stats
        
    except Exception as e:
        logger.error(f"Summary statistics generation failed: {str(e)}")
        raise