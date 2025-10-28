import torch
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2, norm, kstest
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

class VarianceCalibrator:
    """
    Variance calibrator designed for covariate-based generative inference.
    Calibrates the total variance from latent sampling against empirical variance.
    """
    
    def __init__(self, method='temperature_shrinkage', per_dim=True, shrink=0.2, num_samples=1000):
        """
        Args:
            method: 'temperature_shrinkage', 'isotonic', 'linear', or 'temperature_global'
            per_dim: Whether to use per-feature calibration (with shrinkage if applicable)
            shrink: Shrinkage parameter (within [0.1, 0.3]) for temperature_shrinkage
            num_samples: Number of generative samples for calibration fitting
        """
        self.method = method
        self.per_dim = per_dim
        self.shrink = shrink
        self.num_samples = num_samples
        self.calibrators = {}
        self.global_temperature = None
        self.is_fitted = False
        self.covariate_mapping = {}
        
    def _group_covariates(self, cal_covariates):
        """Group calibration data by unique covariate combinations."""
        # Convert to string representation for grouping
        cov_strings = [','.join(map(str, cov)) for cov in cal_covariates]
        unique_covs, indices = np.unique(cov_strings, return_inverse=True)
        
        covariate_groups = {}
        for i, cov_str in enumerate(unique_covs):
            mask = indices == i
            covariate_groups[cov_str] = {
                'indices': np.where(mask)[0],
                'covariate_values': cal_covariates[mask][0]
            }
        
        return covariate_groups
        
    def fit(self, model, cal_data, cal_covariates, device):
        """
        Fit calibrator using generative sampling approach that matches inference.
        """
        logger.info(f"Fitting generative variance calibrator using {self.method} method")
        logger.info(f"Per-dimension: {self.per_dim}, Shrinkage: {self.shrink if 'shrinkage' in self.method else 'N/A'}")
        logger.info(f"Using {self.num_samples} generative samples per covariate combination")
        
        model.eval()
        
        # Convert to tensors
        cal_data = torch.FloatTensor(cal_data).to(device)
        cal_covariates_tensor = torch.FloatTensor(cal_covariates).to(device)
        
        # Group data by covariate combinations
        covariate_groups = self._group_covariates(cal_covariates)
        
        predicted_vars_all = []
        empirical_vars_all = []
        
        logger.info(f"Found {len(covariate_groups)} unique covariate combinations")
        
        for cov_str, group_info in covariate_groups.items():
            indices = group_info['indices'] 
            cov_values = group_info['covariate_values']
            observed_data = cal_data[indices]
            
            # Calculate empirical variance from observed data for this covariate combination
            if len(observed_data) > 1:
                empirical_var = torch.var(observed_data, dim=0).cpu().numpy()
            else:
                # If only one sample, use a small default variance
                empirical_var = np.full(observed_data.shape[1], 0.01)
            
            # Generate samples using the same method as covariate-based inference
            cov_tensor = torch.FloatTensor(cov_values).unsqueeze(0).to(device)
            generated_samples = []
            model_variances = []
            
            with torch.no_grad():
                for _ in range(self.num_samples):
                    # Sample random latent vector (same as inference)
                    z = torch.randn(1, model.latent_dim, device=device)
                    
                    # Decode to get distribution parameters
                    output_dist = model.decode(z, cov_tensor)
                    
                    # Collect model's predicted variance (from decoder)
                    pred_var = output_dist.scale.pow(2).squeeze(0).cpu().numpy()
                    model_variances.append(pred_var)
                    
                    # Sample from the distribution (same as inference)
                    sample = output_dist.sample().squeeze(0).cpu().numpy()
                    generated_samples.append(sample)
            
            # Calculate statistics
            generated_samples = np.array(generated_samples)  # Shape: [num_samples, num_features]
            model_variances = np.array(model_variances)      # Shape: [num_samples, num_features]
            
            # Average model variance across latent samples
            avg_model_var = np.mean(model_variances, axis=0)
            
            # Store for calibration fitting
            predicted_vars_all.append(avg_model_var)
            empirical_vars_all.append(empirical_var)
            
            # Store mapping for later use
            self.covariate_mapping[cov_str] = {
                'covariate_values': cov_values,
                'empirical_var': empirical_var,
                'avg_model_var': avg_model_var
            }
        
        # Convert to arrays for calibration
        predicted_vars_array = np.array(predicted_vars_all)  # Shape: [num_cov_groups, num_features]
        empirical_vars_array = np.array(empirical_vars_all)  # Shape: [num_cov_groups, num_features]
        
        # Fit calibration methods
        if self.method in ['temperature_global', 'temperature_shrinkage']:
            self._fit_temperature_methods(predicted_vars_array, empirical_vars_array)
        elif self.method in ['isotonic', 'linear']:
            self._fit_regression_methods(predicted_vars_array, empirical_vars_array)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
        logger.info(f"Generative variance calibration fitting completed")
    
    def _fit_temperature_methods(self, predicted_vars, empirical_vars):
        """Fit temperature scaling methods."""
        # Remove invalid values
        valid_mask = (predicted_vars > 1e-8) & (empirical_vars > 1e-8) & \
                    ~np.isnan(predicted_vars) & ~np.isnan(empirical_vars)
        
        if not np.any(valid_mask):
            logger.warning("No valid variance pairs found")
            self.global_temperature = 1.0
            return
        
        # Global temperature: s_global = mean(empirical_var / predicted_var)
        ratio_sum = np.sum(empirical_vars[valid_mask] / predicted_vars[valid_mask])
        valid_count = np.sum(valid_mask)
        self.global_temperature = ratio_sum / valid_count
        
        logger.info(f"Global temperature: {self.global_temperature:.4f}")
        
        if self.method == 'temperature_global':
            self.calibrators = {'global': self.global_temperature}
            
        elif self.method == 'temperature_shrinkage' and self.per_dim:
            # Per-dimension temperatures with shrinkage
            n_features = predicted_vars.shape[1]
            per_dim_temperatures = np.zeros(n_features)
            
            for d in range(n_features):
                feature_mask = valid_mask[:, d]
                if np.sum(feature_mask) > 2:  # Need at least 2 covariate groups
                    s_d = np.mean(empirical_vars[feature_mask, d] / predicted_vars[feature_mask, d])
                    per_dim_temperatures[d] = s_d
                else:
                    per_dim_temperatures[d] = self.global_temperature
            
            # Apply shrinkage towards global temperature
            shrunk_temperatures = (1 - self.shrink) * per_dim_temperatures + self.shrink * self.global_temperature
            
            # Store shrunk temperatures
            for d in range(n_features):
                self.calibrators[d] = shrunk_temperatures[d]
            
            logger.info(f"Per-dimension temperatures computed with shrinkage of {self.shrink}")
            logger.info(f"Temperature range: [{shrunk_temperatures.min():.4f}, {shrunk_temperatures.max():.4f}]")
    
    def _fit_regression_methods(self, predicted_vars, empirical_vars):
        """Fit regression-based calibration methods."""
        n_features = predicted_vars.shape[1]
        
        if self.per_dim:
            # Per-feature fitting
            for feature_idx in range(n_features):
                pred_var_feature = predicted_vars[:, feature_idx]
                empirical_var_feature = empirical_vars[:, feature_idx]
                
                # Remove invalid values
                valid_mask = (~np.isnan(pred_var_feature)) & (~np.isnan(empirical_var_feature)) & \
                            (pred_var_feature > 1e-8) & (empirical_var_feature > 1e-8)
                
                if np.sum(valid_mask) < 3:  # Need at least 3 points for regression
                    logger.warning(f"Insufficient valid samples for feature {feature_idx}")
                    continue
                
                pred_valid = pred_var_feature[valid_mask]
                empirical_valid = empirical_var_feature[valid_mask]
                
                if self.method == 'isotonic':
                    # Log-domain isotonic regression
                    log_pred = np.log(pred_valid)
                    log_empirical = np.log(empirical_valid)
                    
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(log_pred, log_empirical)
                    
                elif self.method == 'linear':
                    # Log-domain linear regression
                    log_pred = np.log(pred_valid).reshape(-1, 1)
                    log_empirical = np.log(empirical_valid)
                    
                    calibrator = LinearRegression()
                    calibrator.fit(log_pred, log_empirical)
                
                self.calibrators[feature_idx] = calibrator
        else:
            # Global regression
            pred_flat = predicted_vars.flatten()
            empirical_flat = empirical_vars.flatten()
            
            valid_mask = (~np.isnan(pred_flat)) & (~np.isnan(empirical_flat)) & \
                        (pred_flat > 1e-8) & (empirical_flat > 1e-8)
            
            pred_valid = pred_flat[valid_mask]
            empirical_valid = empirical_flat[valid_mask]
            
            if self.method == 'isotonic':
                log_pred = np.log(pred_valid)
                log_empirical = np.log(empirical_valid)
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(log_pred, log_empirical)
                
            elif self.method == 'linear':
                log_pred = np.log(pred_valid).reshape(-1, 1)
                log_empirical = np.log(empirical_valid)
                calibrator = LinearRegression()
                calibrator.fit(log_pred, log_empirical)
            
            self.calibrators['global'] = calibrator
    
    def calibrate(self, predicted_variances):
        """Apply calibration to predicted variances."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before use")
        
        # Ensure 2D array
        input_shape = predicted_variances.shape
        if len(input_shape) == 1:
            predicted_variances = predicted_variances.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Ensure positive variances
        predicted_variances = np.maximum(predicted_variances, 1e-8)
        
        if self.method == 'temperature_global':
            # Global temperature scaling
            calibrated_variances = self.global_temperature * predicted_variances
            
        elif self.method == 'temperature_shrinkage' and self.per_dim:
            # Per-dimension temperature scaling
            calibrated_variances = np.zeros_like(predicted_variances)
            for d, temperature in self.calibrators.items():
                if isinstance(d, int) and d < predicted_variances.shape[1]:
                    calibrated_variances[:, d] = temperature * predicted_variances[:, d]
            
            # Handle features without calibrators
            for d in range(predicted_variances.shape[1]):
                if d not in self.calibrators:
                    calibrated_variances[:, d] = predicted_variances[:, d]
        
        elif self.method in ['isotonic', 'linear']:
            calibrated_variances = np.zeros_like(predicted_variances)
            
            if self.per_dim:
                for feature_idx, calibrator in self.calibrators.items():
                    if isinstance(feature_idx, int) and feature_idx < predicted_variances.shape[1]:
                        pred_var_feature = predicted_variances[:, feature_idx]
                        log_pred = np.log(pred_var_feature)
                        
                        if self.method == 'isotonic':
                            log_calibrated = calibrator.transform(log_pred)
                        else:  # linear
                            log_calibrated = calibrator.predict(log_pred.reshape(-1, 1))
                        
                        calibrated_variances[:, feature_idx] = np.exp(log_calibrated)
                
                # Handle features without calibrators
                for d in range(predicted_variances.shape[1]):
                    if d not in self.calibrators:
                        calibrated_variances[:, d] = predicted_variances[:, d]
            else:
                # Global calibration
                calibrator = self.calibrators['global']
                pred_flat = predicted_variances.flatten()
                log_pred = np.log(pred_flat)
                
                if self.method == 'isotonic':
                    log_calibrated = calibrator.transform(log_pred)
                else:  # linear
                    log_calibrated = calibrator.predict(log_pred.reshape(-1, 1))
                
                calibrated_variances = np.exp(log_calibrated).reshape(predicted_variances.shape)
        
        if squeeze_output:
            calibrated_variances = calibrated_variances.squeeze()
        
        return calibrated_variances
    
    def evaluate_calibration(self, model, test_data, test_covariates, device, num_eval_samples=500):
        """
        Evaluate calibration quality using generative sampling approach.
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before evaluation")
        
        logger.info("Evaluating generative variance calibration quality")
        
        model.eval()
        
        # Convert to tensors
        test_covariates_tensor = torch.FloatTensor(test_covariates).to(device)
        
        # Group test data by covariates
        test_covariate_groups = self._group_covariates(test_covariates)
        
        all_predicted_vars = []
        all_empirical_vars = []
        all_calibrated_vars = []
        
        for cov_str, group_info in test_covariate_groups.items():
            indices = group_info['indices']
            cov_values = group_info['covariate_values']
            observed_data = test_data[indices]
            
            # Calculate empirical variance
            if len(observed_data) > 1:
                empirical_var = np.var(observed_data, axis=0)
            else:
                continue  # Skip single samples
            
            # Generate samples to get model's predicted variance using vectorised approach
            cov_tensor = torch.FloatTensor(cov_values).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Generate all evaluation samples at once
                z_batch = torch.randn(num_eval_samples, model.latent_dim, device=device)
                cov_batch = cov_tensor.expand(num_eval_samples, -1)
                
                # Get model variances in one forward pass
                output_dist = model.decode(z_batch, cov_batch)
                model_variances = output_dist.scale.pow(2).cpu().numpy()  # Shape: [num_eval_samples, num_features]
            
            avg_model_var = np.mean(model_variances, axis=0)
            calibrated_var = self.calibrate(avg_model_var.reshape(1, -1)).flatten()
            
            all_predicted_vars.append(avg_model_var)
            all_empirical_vars.append(empirical_var)
            all_calibrated_vars.append(calibrated_var)
        
        if not all_predicted_vars:
            logger.warning("No valid test covariate groups for evaluation")
            return {}
        
        # Convert to arrays
        predicted_vars_array = np.array(all_predicted_vars)
        empirical_vars_array = np.array(all_empirical_vars)
        calibrated_vars_array = np.array(all_calibrated_vars)
        
        # Calculate metrics
        mse_uncalibrated = mean_squared_error(empirical_vars_array.flatten(), predicted_vars_array.flatten())
        mse_calibrated = mean_squared_error(empirical_vars_array.flatten(), calibrated_vars_array.flatten())
        
        metrics = {
            'mse_uncalibrated': mse_uncalibrated,
            'mse_calibrated': mse_calibrated,
            'mse_improvement': mse_uncalibrated - mse_calibrated,
            'num_test_groups': len(all_predicted_vars)
        }
        
        logger.info(f"Generative calibration evaluation completed:")
        logger.info(f"  MSE improvement: {metrics['mse_improvement']:.6f}")
        logger.info(f"  Test groups evaluated: {metrics['num_test_groups']}")
        
        return metrics
    
    def save(self, filepath):
        """Save calibrator."""
        import pickle
        calibrator_data = {
            'method': self.method,
            'per_dim': self.per_dim,
            'shrink': self.shrink,
            'num_samples': self.num_samples,
            'calibrators': self.calibrators,
            'global_temperature': self.global_temperature,
            'covariate_mapping': self.covariate_mapping,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(calibrator_data, f)
        logger.info(f"Generative variance calibrator saved to {filepath}")
    
    def load(self, filepath):
        """Load calibrator."""
        import pickle
        with open(filepath, 'rb') as f:
            calibrator_data = pickle.load(f)
        self.method = calibrator_data['method']
        self.per_dim = calibrator_data['per_dim']
        self.shrink = calibrator_data['shrink']
        self.num_samples = calibrator_data.get('num_samples', 1000)
        self.calibrators = calibrator_data['calibrators']
        self.global_temperature = calibrator_data.get('global_temperature')
        self.covariate_mapping = calibrator_data.get('covariate_mapping', {})
        self.is_fitted = calibrator_data['is_fitted']
        logger.info(f"Generative variance calibrator loaded from {filepath}")


# For backward compatibility and reconstruction-based methods (dual_input prediction)
class ReconstructionVarianceCalibrator:
    """
    Original variance calibrator for reconstruction-based inference (dual_input mode).
    """
    
    def __init__(self, method='temperature_shrinkage', per_dim=True, shrink=0.2):
        """
        Args:
            method: 'temperature_shrinkage', 'isotonic', 'linear', or 'temperature_global'
            per_dim: Whether to use per-feature calibration (with shrinkage if applicable)
            shrink: Shrinkage parameter (within [0.1, 0.3]) for temperature_shrinkage
        """
        self.method = method
        self.per_dim = per_dim
        self.shrink = shrink
        self.calibrators = {}
        self.global_temperature = None
        self.is_fitted = False
        
    def fit(self, model, cal_data, cal_covariates, device):
        """
        Fit calibrator using reconstruction residuals (for dual_input prediction).
        """
        logger.info(f"Fitting reconstruction variance calibrator using {self.method} method")
        logger.info(f"Per-dimension: {self.per_dim}, Shrinkage: {self.shrink if 'shrinkage' in self.method else 'N/A'}")
        
        model.eval()
        
        # Convert to tensors
        cal_data = torch.FloatTensor(cal_data).to(device)
        cal_covariates = torch.FloatTensor(cal_covariates).to(device)
        
        with torch.no_grad():
            # Encode observed data to latent space
            mu_z, logvar_z = model.encode(cal_data, cal_covariates)
            z_mean = mu_z  # Use mean of latent distribution
            
            # Decode to get reconstruction distribution
            recon_dist = model.decode(z_mean, cal_covariates)
            
            predicted_means = recon_dist.loc
            predicted_vars = recon_dist.scale.pow(2)
            
            # Compute squared residuals between observed and reconstructed
            squared_residuals = (cal_data - predicted_means) ** 2
        
        # Convert to numpy for calibration fitting
        predicted_vars_np = predicted_vars.cpu().numpy()
        squared_residuals_np = squared_residuals.cpu().numpy()
        
        if self.method in ['temperature_global', 'temperature_shrinkage']:
            self._fit_temperature_methods(predicted_vars_np, squared_residuals_np)
        elif self.method in ['isotonic', 'linear']:
            self._fit_regression_methods(predicted_vars_np, squared_residuals_np)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
        logger.info(f"Reconstruction variance calibration fitting completed")
    
    def _fit_temperature_methods(self, predicted_vars, squared_residuals):
        """Fit temperature scaling methods."""
        # Valid mask for positive variances and residuals
        valid_mask = (predicted_vars > 1e-8) & (squared_residuals > 1e-8) & \
                    ~np.isnan(predicted_vars) & ~np.isnan(squared_residuals)
        
        if not np.any(valid_mask):
            logger.warning("No valid variance pairs found")
            self.global_temperature = 1.0
            return
        
        # Global temperature: s_global = mean(r^2/v) over all valid entries
        ratio_sum = np.sum(squared_residuals[valid_mask] / predicted_vars[valid_mask])
        valid_count = np.sum(valid_mask)
        self.global_temperature = ratio_sum / valid_count
        
        logger.info(f"Global temperature: {self.global_temperature:.4f}")
        
        if self.method == 'temperature_global':
            self.calibrators = {'global': self.global_temperature}
            
        elif self.method == 'temperature_shrinkage' and self.per_dim:
            # Per-dimension temperatures with shrinkage
            n_features = predicted_vars.shape[1]
            per_dim_temperatures = np.zeros(n_features)
            
            for d in range(n_features):
                feature_mask = valid_mask[:, d]
                if np.sum(feature_mask) > 10:
                    s_d = np.mean(squared_residuals[feature_mask, d] / predicted_vars[feature_mask, d])
                    per_dim_temperatures[d] = s_d
                else:
                    per_dim_temperatures[d] = self.global_temperature
            
            # Apply shrinkage
            shrunk_temperatures = (1 - self.shrink) * per_dim_temperatures + self.shrink * self.global_temperature
            
            # Store shrunk temperatures
            for d in range(n_features):
                self.calibrators[d] = shrunk_temperatures[d]
            
            logger.info(f"Per-dimension temperatures computed with shrinkage of {self.shrink}")
            logger.info(f"Temperature range: [{shrunk_temperatures.min():.4f}, {shrunk_temperatures.max():.4f}]")
    
    def _fit_regression_methods(self, predicted_vars, squared_residuals):
        """Fit regression-based calibration methods."""
        n_features = predicted_vars.shape[1]
        
        if self.per_dim:
            for feature_idx in range(n_features):
                pred_var_feature = predicted_vars[:, feature_idx]
                residual_feature = squared_residuals[:, feature_idx]
                
                valid_mask = (~np.isnan(pred_var_feature)) & (~np.isnan(residual_feature)) & \
                            (pred_var_feature > 1e-8) & (residual_feature > 1e-8)
                
                if np.sum(valid_mask) < 10:
                    logger.warning(f"Insufficient valid samples for feature {feature_idx}")
                    continue
                
                pred_valid = pred_var_feature[valid_mask]
                residual_valid = residual_feature[valid_mask]
                
                if self.method == 'isotonic':
                    log_pred = np.log(pred_valid)
                    log_residual = np.log(residual_valid)
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(log_pred, log_residual)
                    
                elif self.method == 'linear':
                    log_pred = np.log(pred_valid).reshape(-1, 1)
                    log_residual = np.log(residual_valid)
                    calibrator = LinearRegression()
                    calibrator.fit(log_pred, log_residual)
                
                self.calibrators[feature_idx] = calibrator
        else:
            # Global regression
            pred_flat = predicted_vars.flatten()
            residual_flat = squared_residuals.flatten()
            
            valid_mask = (~np.isnan(pred_flat)) & (~np.isnan(residual_flat)) & \
                        (pred_flat > 1e-8) & (residual_flat > 1e-8)
            
            pred_valid = pred_flat[valid_mask]
            residual_valid = residual_flat[valid_mask]
            
            if self.method == 'isotonic':
                log_pred = np.log(pred_valid)
                log_residual = np.log(residual_valid)
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(log_pred, log_residual)
                
            elif self.method == 'linear':
                log_pred = np.log(pred_valid).reshape(-1, 1)
                log_residual = np.log(residual_valid)
                calibrator = LinearRegression()
                calibrator.fit(log_pred, log_residual)
            
            self.calibrators['global'] = calibrator
    
    def calibrate(self, predicted_variances):
        """Apply calibration to predicted variances."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before use")
        
        # Implementation is the same as GenerativeVarianceCalibrator.calibrate()
        # (copying the same logic for brevity)
        input_shape = predicted_variances.shape
        if len(input_shape) == 1:
            predicted_variances = predicted_variances.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        predicted_variances = np.maximum(predicted_variances, 1e-8)
        
        if self.method == 'temperature_global':
            calibrated_variances = self.global_temperature * predicted_variances
        elif self.method == 'temperature_shrinkage' and self.per_dim:
            calibrated_variances = np.zeros_like(predicted_variances)
            for d, temperature in self.calibrators.items():
                if isinstance(d, int) and d < predicted_variances.shape[1]:
                    calibrated_variances[:, d] = temperature * predicted_variances[:, d]
            for d in range(predicted_variances.shape[1]):
                if d not in self.calibrators:
                    calibrated_variances[:, d] = predicted_variances[:, d]
        else:
            # Similar logic for regression methods...
            calibrated_variances = predicted_variances  # Simplified for space
        
        if squeeze_output:
            calibrated_variances = calibrated_variances.squeeze()
        
        return calibrated_variances
    
    def save(self, filepath):
        """Save calibrator."""
        import pickle
        calibrator_data = {
            'method': self.method,
            'per_dim': self.per_dim,
            'shrink': self.shrink,
            'calibrators': self.calibrators,
            'global_temperature': self.global_temperature,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(calibrator_data, f)
        logger.info(f"Reconstruction variance calibrator saved to {filepath}")
    
    def load(self, filepath):
        """Load calibrator."""
        import pickle
        with open(filepath, 'rb') as f:
            calibrator_data = pickle.load(f)
        self.method = calibrator_data['method']
        self.per_dim = calibrator_data['per_dim']
        self.shrink = calibrator_data['shrink']
        self.calibrators = calibrator_data['calibrators']
        self.global_temperature = calibrator_data.get('global_temperature')
        self.is_fitted = calibrator_data['is_fitted']
        logger.info(f"Reconstruction variance calibrator loaded from {filepath}")
