import optuna
import torch
import torch.utils.data as data
from pathlib import Path
import logging
import yaml
import numpy as np
import warnings

warnings.filterwarnings('ignore', 
    message='Choices for a categorical distribution should be a tuple of None, bool, int, float and str.*',
    category=UserWarning,
    module='optuna.distributions')

from ..models.cvae import cVAE
from ..utils.data import MyDataset
from ..utils.logger import Logger, plot_losses

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
        self.device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
        
        # Set up Optuna study
        self.study = optuna.create_study(direction="minimize")

    def create_model(self, trial):
        """Create model with parameters suggested by Optuna."""
        # Suggest hyperparameters
        hidden_dim = trial.suggest_categorical('hidden_dim', 
            self.config['optuna']['search_space']['hidden_dim']['choices'])
        latent_dim = trial.suggest_categorical('latent_dim', 
            self.config['optuna']['search_space']['latent_dim']['choices'])
        learning_rate = trial.suggest_float('learning_rate',
            float(self.config['optuna']['search_space']['learning_rate']['min']),
            float(self.config['optuna']['search_space']['learning_rate']['max']),
            log=True)
        batch_size = trial.suggest_categorical('batch_size',
            self.config['optuna']['search_space']['batch_size']['choices'])
        beta = trial.suggest_categorical('beta',
            self.config['optuna']['search_space']['beta']['choices'])
        
        # Create model with suggested parameters
        model = cVAE(
            input_dim=self.config['model']['input_dim'],
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            c_dim=self.train_covariates.shape[1],
            learning_rate=learning_rate,
            beta=beta,
            non_linear=self.config['model']['non_linear']
        )
        return model, batch_size

    def objective(self, trial):
        """Optuna objective function."""
        # Create model and get batch size
        model, batch_size = self.create_model(trial)
        model = model.to(self.device)

        # Create data loaders
        train_dataset = MyDataset(self.train_data, self.train_covariates)
        val_dataset = MyDataset(self.val_data, self.val_covariates)
        
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        logger_trial = Logger()
        logger_trial.on_train_init(['total_loss', 'KL', 'neg_LL'])
        logger_trial.on_val_init(['total_loss', 'KL', 'neg_LL'])
        
        for epoch in range(self.config['training']['epochs']):
            # Training phase
            model.train()
            train_losses = {"total_loss": 0, "KL": 0, "neg_LL": 0}
            num_batches = 0
            
            for batch_data, batch_cov in train_loader:
                batch_data = batch_data.to(self.device)
                batch_cov = batch_cov.to(self.device)
                
                fwd_rtn = model.forward(batch_data, batch_cov)
                loss = model.loss_function(batch_data, fwd_rtn)
                
                model.optimizer.zero_grad()
                loss['Total Loss'].backward()
                model.optimizer.step()
                
                train_losses["total_loss"] += loss['Total Loss'].item()
                train_losses["KL"] += loss['KL Divergence'].item()
                train_losses["neg_LL"] += loss['Reconstruction Loss'].item()
                num_batches += 1

            # Calculate average losses
            avg_losses = {k: v/num_batches for k, v in train_losses.items()}
            logger_trial.on_train_step(avg_losses)

            # Validation phase
            model.eval()
            val_losses = {"total_loss": 0, "KL": 0, "neg_LL": 0}
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_data, batch_cov in val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_cov = batch_cov.to(self.device)
                    
                    fwd_rtn = model.forward(batch_data, batch_cov)
                    loss = model.loss_function(batch_data, fwd_rtn)
                    
                    val_losses["total_loss"] += loss['Total Loss'].item()
                    val_losses["KL"] += loss['KL Divergence'].item()
                    val_losses["neg_LL"] += loss['Reconstruction Loss'].item()
                    num_val_batches += 1

            # Calculate average validation losses
            avg_val_losses = {k: v/num_val_batches for k, v in val_losses.items()}
            logger_trial.on_val_step(avg_val_losses)
            
            current_val_loss = avg_val_losses["total_loss"]

            # Early stopping check
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    break

            # Report to Optuna
            trial.report(current_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_loss

    def run_optimization(self):
        """Run the full optimization process."""
        logger.info(f"Starting Optuna optimization with {self.config['optuna']['n_trials']} trials")
        
        try:
            self.study.optimize(
                self.objective, 
                n_trials=self.config['optuna']['n_trials'],
                show_progress_bar=True
            )
            
            # Get best parameters
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            logger.info(f"Best trial achieved validation loss: {best_value}")
            logger.info("Best hyperparameters:")
            for key, value in best_params.items():
                logger.info(f"\t{key}: {value}")
            
            # Save best parameters
            with open(self.models_dir / 'best_params.yaml', 'w') as f:
                yaml.dump(best_params, f)
            logger.info(f"Saved best parameters to {self.models_dir / 'best_params.yaml'}")
            
            # Train final model with best parameters
            trial = optuna.trial.FixedTrial(best_params)
            final_model, _ = self.create_model(trial)
            final_model = final_model.to(self.device)
            
            # Save study statistics in models directory
            df_study = self.study.trials_dataframe()
            df_study.to_csv(self.models_dir / 'optuna_study_results.csv', index=False)
            logger.info(f"Saved study results to {self.models_dir / 'optuna_study_results.csv'}")
            
            return final_model, best_params
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise