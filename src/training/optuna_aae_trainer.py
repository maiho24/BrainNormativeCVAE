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

from ..models.aae import AAE
from ..utils.data import MyDataset
from ..utils.logger import Logger, plot_losses

logger = logging.getLogger(__name__)

class AAEOptunaTrainer:
    def __init__(self, train_data, train_covariates, val_data, val_covariates, config):
        """
        Initialize the AAE Optuna trainer.
        
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
        self.device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
        
        # Set up Optuna study
        self.study = optuna.create_study(direction="minimize")

    def create_model(self, trial):
        """Create AAE model with parameters suggested by Optuna."""
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
        
        # Create model with suggested parameters
        model = AAE(
            input_dim=self.train_data.shape[1],
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            condition_dim=self.train_covariates.shape[1],
            learning_rate=learning_rate,
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
        loss_keys = ['total_loss', 'reconstruction', 'discriminator', 'generator']
        logger_trial.on_train_init(loss_keys)
        logger_trial.on_val_init(loss_keys)

        # Training loop
        base_lr = learning_rate
        max_lr = self.config['model']['max_learning_rate']
        gamma = self.config['model']['gamma']
        
        for epoch in range(self.config['training']['epochs']):
            # Training phase
            model.train()
            train_losses = self._train_epoch(model, train_loader, epoch, base_lr, max_lr, gamma)
            logger_trial.on_train_step(train_losses)

            # Validation phase
            model.eval()
            val_losses = self._validate(model, val_loader)
            logger_trial.on_val_step(val_losses)
            
            current_val_loss = val_losses["total_loss"]

            # Early stopping check
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                self.best_trial_logger = logger_trial
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    break

            # Report to Optuna
            trial.report(current_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_loss

    def _train_epoch(self, model, train_loader, epoch, base_lr, max_lr, gamma):
        """Train for one epoch with cyclical learning rate."""
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction': 0.0,
            'discriminator': 0.0,
            'generator': 0.0
        }
        
        n_samples = len(train_loader.dataset)
        step_size = 2 * np.ceil(n_samples / train_loader.batch_size)
        
        for batch_idx, (data, covariates) in enumerate(train_loader):
            # Update learning rate
            global_step = epoch * len(train_loader) + batch_idx
            cycle = np.floor(1 + global_step / (2 * step_size))
            x_lr = np.abs(global_step / step_size - 2 * cycle + 1)
            clr = base_lr + (max_lr - base_lr) * max(0, 1 - x_lr) * (gamma ** cycle)
            
            for param_group in model.enc_dec_optimizer.param_groups:
                param_group['lr'] = clr
            for param_group in model.disc_optimizer.param_groups:
                param_group['lr'] = clr
            
            # Forward pass
            data = data.to(self.device)
            covariates = covariates.to(self.device)
            
            losses = model.training_step(data, covariates)
            
            for k, v in losses.items():
                epoch_losses[k] += v.item()
        
        return {k: v/len(train_loader) for k, v in epoch_losses.items()}

    def _validate(self, model, val_loader):
        """Validate the model."""
        model.eval()
        val_losses = {
            'total_loss': 0.0,
            'reconstruction': 0.0,
            'discriminator': 0.0,
            'generator': 0.0
        }
        
        with torch.no_grad():
            for data, covariates in val_loader:
                data = data.to(self.device)
                covariates = covariates.to(self.device)
                
                # Forward pass without gradient updates
                x_recon, z = model(data, covariates)
                losses = {
                    'reconstruction': model.reconstruction_criterion(x_recon, data),
                    'discriminator': model.adversarial_criterion(
                        model.discriminate(z),
                        torch.zeros_like(model.discriminate(z))
                    ),
                    'generator': model.adversarial_criterion(
                        model.discriminate(z),
                        torch.ones_like(model.discriminate(z))
                    )
                }
                losses['total_loss'] = losses['reconstruction'] + losses['generator']
                
                for k, v in losses.items():
                    val_losses[k] += v.item()
        
        return {k: v/len(val_loader) for k, v in val_losses.items()}

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
            model_dir = Path(self.config['paths']['model_dir'])
            with open(model_dir / 'best_params.yaml', 'w') as f:
                yaml.dump(best_params, f)
                
            # Save study statistics
            df_study = self.study.trials_dataframe()
            df_study.to_csv(model_dir / 'optuna_study_results.csv', index=False)
            
            # Plot and save training curves for best trial
            if self.best_trial_logger is not None:
                plot_losses(self.best_trial_logger, model_dir, '_best_trial')
                logger.info(f"Saved loss plots for best trial to {model_dir}")
            
            # Create training config with best parameters
            train_config = self.config.copy()
            train_config['model'].update({
                'hidden_dim': best_params['hidden_dim'],
                'latent_dim': best_params['latent_dim'],
                'learning_rate': best_params['learning_rate']
            })
            train_config['training']['batch_size'] = best_params['batch_size']
            
            # Create and train final model with best parameters
            final_model = AAE(
                input_dim=self.train_data.shape[1],
                hidden_dim=best_params['hidden_dim'],
                latent_dim=best_params['latent_dim'],
                condition_dim=self.train_covariates.shape[1],
                learning_rate=best_params['learning_rate'],
                non_linear=self.config['model']['non_linear']
            ).to(self.device)
            
            # Train the final model
            trainer = AAETrainer(train_config)
            final_model = trainer.train_model(
                self.train_data,
                self.train_covariates,
                self.val_data,
                self.val_covariates
            )
            
            return final_model, best_params
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise