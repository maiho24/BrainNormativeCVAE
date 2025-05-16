import optuna
import multiprocessing
import torch
import torch.utils.data as data
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    def __init__(self, train_data, train_covariates, val_data, val_covariates, config):
        """
        Initialize the Optuna trainer.
        
        Args:
            train_data: Training data numpy array
            train_covariates: Training covariates numpy array
            val_data: Validation data numpy array
            val_covariates: Validation covariates numpy array
            config: Configuration dictionary
        """
        self.train_data = train_data
        self.train_covariates = train_covariates
        self.val_data = val_data
        self.val_covariates = val_covariates
        self.config = config
        self.best_trial_logger = None
        
        storage = self.config['optuna'].get('storage', None)
        cores_per_trial = self.config.get('optuna', {}).get('cores_per_trial', 1)
        self.n_jobs = self._determine_n_jobs(cores_per_trial)
        
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
    
    def _determine_n_jobs(self, cores_per_trial=1):
        """Determine optimal number of parallel jobs based on hardware resources."""
        user_n_jobs = self.config.get('optuna', {}).get('n_jobs', -1)
        cpu_count = multiprocessing.cpu_count()
        gpu_count = torch.cuda.device_count() if self.config.get('device', {}).get('gpu', True) else 0
        
        cores_reserved = self.config.get('optuna', {}).get('cores_reserved', 2)
        max_safe_parallel = max(1, (cpu_count - cores_reserved) // cores_per_trial)
        
        logger.info(f"Detected {cpu_count} CPUs and {gpu_count} GPUs.")
        
        if gpu_count > 0:
            if user_n_jobs <= 0:
                optimal_n_jobs = gpu_count
                logger.info(f"Using {optimal_n_jobs} GPUs.")
            else:
                optimal_n_jobs = user_n_jobs
                if user_n_jobs > gpu_count + max_safe_parallel:
                    logger.warning(
                        f"User-specified n_jobs={user_n_jobs} may oversubscribe resources. "
                        f"Consider reducing to {gpu_count + max_safe_parallel} for better performance."
                    )
        else:
            if self.config.get('device', {}).get('gpu', True):
                logger.info(f"Detected 0 GPUs -> Using {cpu_count} CPUs instead.")
            if user_n_jobs <= 0:
                optimal_n_jobs = max_safe_parallel
                logger.info(f"Using {optimal_n_jobs} CPUs.")
            else:
                optimal_n_jobs = min(user_n_jobs, max_safe_parallel)
                if optimal_n_jobs < user_n_jobs:
                    logger.info(f"Limiting user-specified n_jobs={user_n_jobs} to {optimal_n_jobs} to prevent CPU oversubscription")
        
        return max(1, optimal_n_jobs)
        
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
        """
        Parse string representation of hidden dimensions into list of integers.
        
        Args:
            hidden_dim_str: String representation of hidden dimensions (e.g., "64_32")
            
        Returns:
            List of integers representing hidden dimensions
        """
        try:
            return [int(x) for x in hidden_dim_str.split('_')]
        except (ValueError, AttributeError) as e:
            logger.error(f"Error parsing hidden dimensions {hidden_dim_str}: {str(e)}")
            raise ValueError(f"Invalid hidden dimension format: {hidden_dim_str}. Expected format: 'dim1_dim2' or 'dim1'")

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
        
        variance_type = self.config['model'].get('variance_type', 'global_learnable')
        if variance_type == "covariate_specific":
            # Check if varnet_hidden_dim is in the search space
            if 'varnet_hidden_dim' in self.config['optuna']['search_space']:
                varnet_hidden_dim_str = trial.suggest_categorical('varnet_hidden_dim',
                    self.config['optuna']['search_space']['varnet_hidden_dim']['choices'])
                varnet_hidden_dim = self._parse_hidden_dim(varnet_hidden_dim_str)
            else:
                # Use default value of [32] if not in search space
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

    def objective(self, trial):
        """Optuna objective function."""
        device = torch.device(self._get_device_for_trial()) 
        logger.info(f"Trial {trial.number} running on {device}")

        model, batch_size, learning_rate = self.create_model(trial)
        model = model.to(device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-6
        )
        
        train_dataset = MyDataset(self.train_data, self.train_covariates)
        val_dataset = MyDataset(self.val_data, self.val_covariates)
        
        train_loader = data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True if device.type == 'cuda' else False
        )

        best_val_loss = float('inf')
        patience_counter = 0
        logger_trial = Logger()
        logger_trial.on_train_init(['Total Loss', 'KL Divergence', 'Reconstruction Loss'])
        logger_trial.on_val_init(['Total Loss', 'KL Divergence', 'Reconstruction Loss'])
        
        model_dir = Path(self.config['paths']['model_dir'])
        checkpoint_dir = model_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        #------ TRAINING & VALIDATION ------#
        for epoch in range(self.config['training']['epochs']):
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
            
            current_lr = optimizer.param_groups[0]['lr']
            avg_losses = {k: v/num_batches for k, v in train_losses.items()}

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
                self.best_trial_logger = logger_trial
                
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'trial_number': trial.number,
                    'loss': best_val_loss
                }
                torch.save(checkpoint, checkpoint_dir / f"trial_{trial.number}_checkpoint.pt")
                logger_path = checkpoint_dir / f"trial_{trial.number}_logger.pkl"
                with open(logger_path, 'wb') as f:
                    pickle.dump(logger_trial, f)
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    break
            
            logger_trial.on_train_step(avg_losses)
            logger_trial.on_val_step(avg_val_losses)
            
            trial.report(current_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_loss

    def run_optimization(self):
        """Run the full optimization process."""
        logger.info(f"Starting Optuna optimization with {self.config['optuna']['n_trials']} trials")
        
        try:
            logger.info(f"Running optimization with {self.n_jobs} parallel workers")
            
            self.study.optimize(
                self.objective, 
                n_trials=self.config['optuna']['n_trials'],
                n_jobs=self.n_jobs,
                show_progress_bar=True
            )
            
            best_params = self.study.best_params
            best_value = self.study.best_value
            best_trial = self.study.best_trial
            
            logger.info(f"Best trial (#{best_trial.number}) achieved validation loss: {best_value}")
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
            best_logger_path = checkpoint_dir / f"trial_{best_trial.number}_logger.pkl"
            if os.path.exists(best_logger_path):
                with open(best_logger_path, 'rb') as f:
                    self.best_trial_logger = pickle.load(f)
                plot_losses(self.best_trial_logger, model_dir, '_best_trial')
                logger.info(f"Saved loss plots for best trial to {model_dir}")
            
            # Remove checkpoints from non-best trials
            logger.info(f"Removing non-best trials from checkpoints/")
            best_checkpoint_filename = f"trial_{best_trial.number}_checkpoint.pt"
            best_checkpoint_path = checkpoint_dir / best_checkpoint_filename
            best_logger_path = checkpoint_dir / f"trial_{best_trial.number}_logger.pkl"
            for file in checkpoint_dir.glob("trial_*_checkpoint.pt"):
                if file.name != best_checkpoint_filename:
                    file.unlink()
            for file in checkpoint_dir.glob("trial_*_logger.pkl"):
                if file.name != best_logger_path:
                    file.unlink()
            logger.info(f"Best checkpoint saved as {best_checkpoint_path}")
            
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
            
            # Retrain model with best params
            logger.info("Retraining model with best parameters...")
            final_model = train_model(
                self.train_data,
                self.train_covariates,
                self.val_data,
                self.val_covariates,
                train_config
            )
            
            return final_model, best_params
            
        except Exception as e:
            logger.error(f"Optimisation failed: {str(e)}")
            raise