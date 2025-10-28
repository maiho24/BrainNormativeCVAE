import optuna
import multiprocessing
import torch
import torch.utils.data as data
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from pathlib import Path
import logging
import yaml
import numpy as np
import pickle
import os

from ..models.cvae import cVAE
from ..utils.data import MyDataset
from ..utils.logger import Logger, plot_losses
from .train import train_model

logger = logging.getLogger(__name__)


class OptunaTrainer:
    def __init__(self, train_data, train_covariates, val_data, val_covariates, config, cal_data=None, cal_covariates=None):
        """
        Initialise the Optuna trainer.
        
        Args:
            train_data: Training data numpy array
            train_covariates: Training covariates numpy array
            val_data: Validation data numpy array
            val_covariates: Validation covariates numpy array
            cal_data: Validation data numpy array (Optional)
            cal_covariates: Validation covariates numpy array (Optional)
            config: Configuration dictionary
        """
        self.train_data = train_data
        self.train_covariates = train_covariates
        self.val_data = val_data
        self.val_covariates = val_covariates
        self.cal_data = cal_data
        self.cal_covariates = cal_covariates
        self.config = config
        
        self.best_trial_number = None
        self.best_trial_score = float('inf')
        self.best_trial_logger = None
        self.best_trial_params = None
        self.early_stopping_patience = config['training']['early_stopping_patience']
        self.max_epochs = config['training']['epochs']
        
        # Cross-validation settings
        self.use_cv = config.get('cross_validation', {}).get('enabled', False)
        self.cv_folds = config.get('cross_validation', {}).get('n_folds', 5)
        self.cv_stratified = config.get('cross_validation', {}).get('stratified', False)
        self.cv_random_state = config.get('cross_validation', {}).get('random_state', 42)
        
        if self.use_cv:
            logger.info(f"Cross-validation enabled: {self.cv_folds}-fold {'stratified' if self.cv_stratified else 'standard'} CV")
        else:
            logger.info("Cross-validation disabled: using standard train/validation split")
        
        storage = self.config['optuna'].get('storage', None)
        self.n_jobs = self._determine_n_jobs()
        
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, 
            n_warmup_steps=20,
            interval_steps=1
        )
        
        sampler = optuna.samplers.TPESampler(seed=self.config.get('seed', 42))
        
        self.study = optuna.create_study(
            study_name=self.config['optuna']['study_name'],
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            pruner=pruner,
            sampler=sampler
        )
    
    def _determine_n_jobs(self):
        """Determine number of parallel trials."""
        user_n_jobs = self.config.get('optuna', {}).get('n_jobs', -1)
        gpu_count = torch.cuda.device_count() if self.config.get('device', {}).get('gpu', True) else 0
        
        if user_n_jobs == -1:  # Auto mode
            if gpu_count > 0:
                suggested_n_jobs = gpu_count
                logger.info(f"Auto mode: Using {suggested_n_jobs} parallel trials (1 per GPU)")
            else:
                suggested_n_jobs = 1
                logger.info(f"Auto mode: Using {suggested_n_jobs} parallel trial (CPU only)")
            return suggested_n_jobs
        else:
            logger.info(f"Using user-specified {user_n_jobs} parallel trials")
            return user_n_jobs
        
    def _get_device_for_trial(self):
        """Select the most appropriate device for a trial with better memory management."""
        try:
            if torch.cuda.is_available() and self.config.get('device', {}).get('gpu', True):
                if not hasattr(self, '_gpu_assignment_count'):
                    self._gpu_assignment_count = {i: 0 for i in range(torch.cuda.device_count())}
                    
                free_mem = []
                for i in range(torch.cuda.device_count()):
                    try:
                        total_mem = torch.cuda.get_device_properties(i).total_memory
                        allocated_mem = torch.cuda.memory_allocated(i)
                        free_mem_bytes = total_mem - allocated_mem
                        free_mem_gb = free_mem_bytes / (1024**3)
                        free_mem.append((free_mem_gb, self._gpu_assignment_count[i], i))
                    except Exception as e:
                        logger.warning(f"Error querying GPU {i}: {e}")
                        continue
                
                if free_mem:
                    # Sort by: 1) most free memory 2) least assigned trials
                    free_mem.sort(key=lambda x: (-x[0], x[1]))
                    best_free_mem, _, best_gpu = free_mem[0]
                    
                    # Check against minimum memory threshold
                    min_gpu_mem_gb = self.config.get('device', {}).get('min_gpu_memory_gb', 1.0)
                    
                    if best_free_mem >= min_gpu_mem_gb:
                        self._gpu_assignment_count[best_gpu] += 1
                        device = torch.device(f"cuda:{best_gpu}")
                        return device
                    else:
                        logger.info(f"Best GPU has only {best_free_mem:.2f}GB free memory (below {min_gpu_mem_gb}GB threshold)")
        except Exception as e:
            logger.warning(f"Error during device selection: {e}")
        
        return torch.device("cpu")
    
    def _parse_hidden_dim(self, hidden_dim_str):
        """Parse string representation of hidden dimensions into list of integers."""
        try:
            return [int(x) for x in hidden_dim_str.split('_')]
        except (ValueError, AttributeError) as e:
            logger.error(f"Error parsing hidden dimensions {hidden_dim_str}: {str(e)}")
            raise ValueError(f"Invalid hidden dimension format: {hidden_dim_str}. Expected format: 'dim1_dim2' or 'dim1'")

    def _create_cv_splits(self):
        """Create cross-validation splits."""
        if self.cv_stratified:
            # For stratified CV, create stratification labels
            stratify_labels = []
            for cov in self.train_covariates:
                # Create a hash-based label for stratification
                label = hash(tuple(cov.astype(str))) % 100
                stratify_labels.append(label)
            
            cv_splitter = StratifiedKFold(
                n_splits=self.cv_folds, 
                shuffle=True, 
                random_state=self.cv_random_state
            )
            splits = cv_splitter.split(self.train_data, stratify_labels)
        else:
            cv_splitter = KFold(
                n_splits=self.cv_folds, 
                shuffle=True, 
                random_state=self.cv_random_state
            )
            splits = cv_splitter.split(self.train_data)
        
        return list(splits)

    def create_model(self, trial):
        """Create model with parameters suggested by Optuna."""
        # Suggest hyperparameters
        hidden_dim_str = trial.suggest_categorical('hidden_dim', 
            self.config['optuna']['search_space']['hidden_dim']['choices'])
        hidden_dim = self._parse_hidden_dim(hidden_dim_str)
        latent_dim = trial.suggest_categorical('latent_dim', 
            self.config['optuna']['search_space']['latent_dim']['choices'])
        learning_rate = trial.suggest_float('learning_rate',
            float(self.config['optuna']['search_space']['learning_rate']['min']),
            float(self.config['optuna']['search_space']['learning_rate']['max']),
            log=True)
        batch_size = trial.suggest_categorical('batch_size',
            self.config['optuna']['search_space']['batch_size']['choices'])
            
        if 'beta' in self.config['optuna']['search_space']:
            beta = trial.suggest_categorical('beta', self.config['optuna']['search_space']['beta']['choices'])
        else:
            beta = 1
        
        variance_type = self.config['model'].get('variance_type', 'two_heads')
        if variance_type == "covariate_specific":
            if 'varnet_hidden_dim' in self.config['optuna']['search_space']:
                varnet_hidden_dim_str = trial.suggest_categorical('varnet_hidden_dim',
                    self.config['optuna']['search_space']['varnet_hidden_dim']['choices'])
                varnet_hidden_dim = self._parse_hidden_dim(varnet_hidden_dim_str)
            else:
                varnet_hidden_dim = [32]
        else:
            varnet_hidden_dim = None
        
        # Create model with suggested parameters    
        model = cVAE(
            input_dim=self.train_data.shape[1],
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            c_dim=self.train_covariates.shape[1],
            beta=beta,
            non_linear=self.config['model']['non_linear'],
            variance_type=variance_type,
            variance_network_hidden_dim=varnet_hidden_dim
        )
        
        return model, batch_size, learning_rate

    def _train_single_fold(self, model, train_indices, val_indices, batch_size, learning_rate, device, trial, fold_idx):
        """Train model on a single fold."""
        # Create fold-specific data
        fold_train_data = self.train_data[train_indices]
        fold_train_covariates = self.train_covariates[train_indices]
        fold_val_data = self.train_data[val_indices]
        fold_val_covariates = self.train_covariates[val_indices]
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        
        train_dataset = MyDataset(fold_train_data, fold_train_covariates)
        val_dataset = MyDataset(fold_val_data, fold_val_covariates)
        
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                     pin_memory=True if device.type == 'cuda' else False)
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                   pin_memory=True if device.type == 'cuda' else False)

        best_val_loss = float('inf')
        patience_counter = 0
        fold_logger = Logger()
        fold_logger.on_train_init(['Total Loss', 'KL Divergence', 'Reconstruction Loss'])
        fold_logger.on_val_init(['Total Loss', 'KL Divergence', 'Reconstruction Loss'])
        
        for epoch in range(self.max_epochs):
            # Training phase
            model.train()
            train_losses = {"Total Loss": 0, "KL Divergence": 0, "Reconstruction Loss": 0}
            num_batches = 0
            
            for batch_data, batch_cov in train_loader:
                batch_data = batch_data.to(device)
                batch_cov = batch_cov.to(device)
                
                fwd_rtn = model.forward(batch_data, batch_cov)
                loss = model.loss_function(batch_data, fwd_rtn)
                
                optimizer.zero_grad()
                loss['Total Loss'].backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses["Total Loss"] += loss['Total Loss'].item()
                train_losses["KL Divergence"] += loss['KL Divergence'].item()
                train_losses["Reconstruction Loss"] += loss['Reconstruction Loss'].item()
                num_batches += 1
            
            avg_train_losses = {k: v/num_batches for k, v in train_losses.items()}

            # Validation phase
            model.eval()
            val_losses = {"Total Loss": 0, "KL Divergence": 0, "Reconstruction Loss": 0}
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_data, batch_cov in val_loader:
                    batch_data = batch_data.to(device)
                    batch_cov = batch_cov.to(device)
                    
                    fwd_rtn = model.forward(batch_data, batch_cov)
                    loss = model.loss_function(batch_data, fwd_rtn)
                    
                    val_losses["Total Loss"] += loss['Total Loss'].item()
                    val_losses["KL Divergence"] += loss['KL Divergence'].item()
                    val_losses["Reconstruction Loss"] += loss['Reconstruction Loss'].item()
                    num_val_batches += 1

            avg_val_losses = {k: v/num_val_batches for k, v in val_losses.items()}
            current_val_loss = avg_val_losses["Total Loss"]
            scheduler.step(current_val_loss)

            # Early stopping check
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break
            
            fold_logger.on_train_step(avg_train_losses)
            fold_logger.on_val_step(avg_val_losses)
            
        return best_val_loss, fold_logger

    def objective(self, trial):
        """Optuna objective function with optional cross-validation."""
        device = torch.device(self._get_device_for_trial()) 
        logger.info(f"Trial {trial.number} running on {device}")

        model, batch_size, learning_rate = self.create_model(trial)
        model = model.to(device)
        
        if self.use_cv:
            # Cross-validation approach
            cv_splits = self._create_cv_splits()
            fold_losses = []
            fold_loggers = []
            
            logger.info(f"Trial {trial.number}: Starting {self.cv_folds}-fold cross-validation")
            
            for fold_idx, (train_indices, val_indices) in enumerate(cv_splits):
                logger.info(f"Trial {trial.number}, Fold {fold_idx + 1}/{self.cv_folds}")
                
                # Reinitialize model for each fold
                fold_model, _, _ = self.create_model(trial)
                fold_model = fold_model.to(device)
                
                try:
                    fold_loss, fold_logger = self._train_single_fold(
                        fold_model, train_indices, val_indices, batch_size, learning_rate, 
                        device, trial, fold_idx
                    )
                    fold_losses.append(fold_loss)
                    fold_loggers.append(fold_logger)
                    
                except optuna.exceptions.TrialPruned:
                    logger.info(f"Trial {trial.number} pruned during fold {fold_idx + 1}")
                    raise
                
                logger.info(f"Trial {trial.number}, Fold {fold_idx + 1} completed with loss: {fold_loss:.4f}")
            
            # Calculate mean CV score
            mean_cv_score = np.mean(fold_losses)
            std_cv_score = np.std(fold_losses)
            
            logger.info(f"Trial {trial.number} CV completed: {mean_cv_score:.4f}  {std_cv_score:.4f}")
            trial.report(mean_cv_score, 0)
            
            # Save best trial info if this is the best so far
            if mean_cv_score < self.best_trial_score:
                self.best_trial_score = mean_cv_score
                self.best_trial_number = trial.number
                self.best_trial_params = {
                    'hidden_dim': trial.params['hidden_dim'],
                    'latent_dim': trial.params['latent_dim'],
                    'learning_rate': trial.params['learning_rate'],
                    'batch_size': trial.params['batch_size'],
                    'beta': trial.params.get('beta', 1)
                }
                
                # Save results for this best trial
                model_dir = Path(self.config['paths']['model_dir'])
                checkpoint_dir = model_dir / 'checkpoints'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Choose representative logger (e.g., median performing fold)
                fold_losses_sorted = sorted(enumerate(fold_losses), key=lambda x: x[1])
                median_fold_idx = fold_losses_sorted[len(fold_losses) // 2][0]
                self.best_trial_logger = fold_loggers[median_fold_idx]
                
                # Save CV results for the best trial
                cv_results = {
                    'trial_number': trial.number,
                    'mean_cv_score': mean_cv_score,
                    'std_cv_score': std_cv_score,
                    'fold_scores': fold_losses,
                    'n_folds': self.cv_folds,
                    'best_fold_idx': int(np.argmin(fold_losses)),
                    'median_fold_idx': median_fold_idx
                }
                
                cv_results_path = checkpoint_dir / f"best_trial_{trial.number}_cv_results.pkl"
                with open(cv_results_path, 'wb') as f:
                    pickle.dump(cv_results, f)
                
                logger_path = checkpoint_dir / f"best_trial_{trial.number}_logger.pkl"
                with open(logger_path, 'wb') as f:
                    pickle.dump(self.best_trial_logger, f)
                
                logger.info(f"New best trial: #{trial.number} with CV score {mean_cv_score:.4f}")
            
            return mean_cv_score
            
        else:
            # Standard train/validation approach (using the reduced validation set if calibration is enabled)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
            
            train_dataset = MyDataset(self.train_data, self.train_covariates)
            val_dataset = MyDataset(self.val_data, self.val_covariates)
            
            train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                         pin_memory=True if device.type == 'cuda' else False)
            val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                       pin_memory=True if device.type == 'cuda' else False)

            patience_counter = 0
            logger_trial = Logger()
            logger_trial.on_train_init(['Total Loss', 'KL Divergence', 'Reconstruction Loss'])
            logger_trial.on_val_init(['Total Loss', 'KL Divergence', 'Reconstruction Loss'])
            
            model_dir = Path(self.config['paths']['model_dir'])
            checkpoint_dir = model_dir / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            for epoch in self.max_epochs:
                # Training phase
                model.train()
                train_losses = {"Total Loss": 0, "KL Divergence": 0, "Reconstruction Loss": 0}
                num_batches = 0
                
                for batch_data, batch_cov in train_loader:
                    batch_data = batch_data.to(device)
                    batch_cov = batch_cov.to(device)
                    
                    fwd_rtn = model.forward(batch_data, batch_cov)
                    loss = model.loss_function(batch_data, fwd_rtn)
                    
                    optimizer.zero_grad()
                    loss['Total Loss'].backward()
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_losses["Total Loss"] += loss['Total Loss'].item()
                    train_losses["KL Divergence"] += loss['KL Divergence'].item()
                    train_losses["Reconstruction Loss"] += loss['Reconstruction Loss'].item()
                    num_batches += 1
                
                avg_train_losses = {k: v/num_batches for k, v in train_losses.items()}

                # Validation phase
                model.eval()
                val_losses = {"Total Loss": 0, "KL Divergence": 0, "Reconstruction Loss": 0}
                num_val_batches = 0
                
                with torch.no_grad():
                    for batch_data, batch_cov in val_loader:
                        batch_data = batch_data.to(device)
                        batch_cov = batch_cov.to(device)
                        
                        fwd_rtn = model.forward(batch_data, batch_cov)
                        loss = model.loss_function(batch_data, fwd_rtn)
                        
                        val_losses["Total Loss"] += loss['Total Loss'].item()
                        val_losses["KL Divergence"] += loss['KL Divergence'].item()
                        val_losses["Reconstruction Loss"] += loss['Reconstruction Loss'].item()
                        num_val_batches += 1

                avg_val_losses = {k: v/num_val_batches for k, v in val_losses.items()}
                current_val_loss = avg_val_losses["Total Loss"]
                scheduler.step(current_val_loss)

                # Early stopping check
                if current_val_loss < self.best_trial_score:
                    patience_counter = 0
                    
                    self.best_trial_score = current_val_loss
                    self.best_trial_number = trial.number
                    self.best_trial_logger = logger_trial
                    self.best_trial_params = {
                        'hidden_dim': trial.params['hidden_dim'],
                        'latent_dim': trial.params['latent_dim'],
                        'learning_rate': trial.params['learning_rate'],
                        'batch_size': trial.params['batch_size'],
                        'beta': trial.params.get('beta', 1)
                    }
                    
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'trial_number': trial.number,
                        'loss': current_val_loss
                    }
                    torch.save(checkpoint, checkpoint_dir / f"trial_{trial.number}_checkpoint.pt")
                    logger_path = checkpoint_dir / f"trial_{trial.number}_logger.pkl"
                    with open(logger_path, 'wb') as f:
                        pickle.dump(logger_trial, f)
                        
                    logger.info(f"New best trial: #{trial.number} with validation loss {current_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break
                
                logger_trial.on_train_step(avg_train_losses)
                logger_trial.on_val_step(avg_val_losses)
                
                trial.report(current_val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return current_val_loss

    def run_optimization(self):
        """Run the full optimisation process."""
        cv_text = f" with {self.cv_folds}-fold cross-validation" if self.use_cv else ""
        calibration_text = " (calibration data held out)" if self.cal_data is not None else ""
        logger.info(f"Starting Optuna optimisation{cv_text}{calibration_text} with {self.config['optuna']['n_trials']} trials")
        
        try:
            logger.info(f"Running optimisation with {self.n_jobs} parallel workers")
            
            self.study.optimize(
                self.objective, 
                n_trials=self.config['optuna']['n_trials'],
                n_jobs=self.n_jobs,
                show_progress_bar=True
            )
            
            best_params = self.study.best_params
            best_value = self.study.best_value
            best_trial = self.study.best_trial
            
            # Verify our tracking matches Optuna's
            if self.best_trial_number != best_trial.number:
                logger.warning(f"Tracking mismatch: Our best trial {self.best_trial_number} vs Optuna's {best_trial.number}")
                # Use Optuna's choice as authoritative
                self.best_trial_number = best_trial.number
                self.best_trial_score = best_value
            
            if self.use_cv:
                logger.info(f"Best trial (#{self.best_trial_number}) achieved CV score: {self.best_trial_score:.4f}")
            else:
                logger.info(f"Best trial (#{self.best_trial_number}) achieved validation loss: {self.best_trial_score:.4f}")
            
            logger.info("Best hyperparameters:")
            for key, value in best_params.items():
                logger.info(f"\t{key}: {value}")
            
            model_dir = Path(self.config['paths']['model_dir'])
            with open(model_dir / 'best_params.yaml', 'w') as f:
                yaml.dump(best_params, f)
                
            df_study = self.study.trials_dataframe()
            df_study.to_csv(model_dir / 'optuna_study_results.csv', index=False)
            
            # Plot and save training curves for best trial
            checkpoint_dir = model_dir / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            if self.use_cv:
                # For CV, use our tracked logger
                if self.best_trial_logger is not None:
                    plot_losses(self.best_trial_logger, model_dir, '_best_trial_cv')
                    logger.info(f"Saved CV loss plots for best trial to {model_dir}")
            else:
                # For non-CV, try to load from checkpoint
                best_logger_path = checkpoint_dir / f"trial_{self.best_trial_number}_logger.pkl"
                if os.path.exists(best_logger_path):
                    with open(best_logger_path, 'rb') as f:
                        self.best_trial_logger = pickle.load(f)
                    plot_losses(self.best_trial_logger, model_dir, '_best_trial')
                    logger.info(f"Saved loss plots for best trial to {model_dir}")
            
            # Remove checkpoints from non-best trials
            logger.info(f"Removing non-best trials from checkpoints/")
            best_checkpoint_filename = f"trial_{best_trial.number}_checkpoint.pt"
            best_cv_results_filename = f"trial_{best_trial.number}_cv_results.pkl"
            best_checkpoint_path = checkpoint_dir / best_checkpoint_filename
            best_logger_path = checkpoint_dir / f"trial_{best_trial.number}_logger.pkl"
            best_cv_results_path = checkpoint_dir / best_cv_results_filename
            
            for file in checkpoint_dir.glob("trial_*_checkpoint.pt"):
                if file.name != best_checkpoint_filename:
                    file.unlink()
            for file in checkpoint_dir.glob("trial_*_logger.pkl"):
                if file.name != f"trial_{best_trial.number}_logger.pkl":
                    file.unlink()
            for file in checkpoint_dir.glob("trial_*_cv_results.pkl"):
                if file.name != best_cv_results_filename:
                    file.unlink()
                    
            logger.info(f"Best checkpoint saved as {best_checkpoint_path}")
            if self.use_cv and best_cv_results_path.exists():
                logger.info(f"Best CV results saved as {best_cv_results_path}")
            
            # Create training config with best parameters
            train_config = self.config.copy()
            if 'model' not in train_config:
                train_config['model'] = {}
            train_config['model'].update({
                'hidden_dim': best_params['hidden_dim'],
                'latent_dim': best_params['latent_dim'],
                'learning_rate': best_params['learning_rate'],
                'beta': best_params.get('beta', 1)
            })
            train_config['training']['batch_size'] = best_params['batch_size']
            if 'varnet_hidden_dim' in best_params:
                train_config['model']['varnet_hidden_dim'] = best_params['varnet_hidden_dim']
            elif train_config['model'].get('variance_type') == 'covariate_specific':
                train_config['model']['varnet_hidden_dim'] = '32'
            
            best_config_file = model_dir / 'best_training_config.yaml'
            with open(best_config_file, 'w') as f:
                yaml.dump(train_config, f)
            logger.info(f"Saved configuration with best params to {best_config_file}")
            
            # Retrain model with best params using the training + validation data
            logger.info(f"Retraining model with best parameters...")
            
            # For final training, we combine train + val data, but keep calibration data separate
            final_train_data = np.vstack([self.train_data, self.val_data])
            final_train_covariates = np.vstack([self.train_covariates, self.val_covariates])
            
            # Use calibration data as validation for final training (if available)
            if self.cal_data is not None:
                final_val_data = self.cal_data
                final_val_covariates = self.cal_covariates
                final_cal_data = self.cal_data  # Same data used for both validation and calibration in final training
                final_cal_covariates = self.cal_covariates
                logger.info("Using calibration data as validation set for final training")
            else:
                # If no calibration, need to split again for final validation (very small portion)
                final_train_data, final_val_data, final_train_covariates, final_val_covariates = train_test_split(
                    final_train_data, final_train_covariates, 
                    test_size=0.05, random_state=43
                )
                final_cal_data, final_cal_covariates = None, None
                logger.info("Split final training data for validation (no calibration enabled)")
            
            final_model = train_model(
                final_train_data,
                final_train_covariates,
                final_val_data,
                final_val_covariates,
                train_config,
                cal_data=final_cal_data,
                cal_covariates=final_cal_covariates
            )
            
            return final_model, best_params
            
        except Exception as e:
            logger.error(f"Optimisation failed: {str(e)}")
            raise