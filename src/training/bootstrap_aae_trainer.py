"""
Bootstrap Training Implementation for CAAE with proper logging
"""

import torch
import numpy as np
from pathlib import Path
import logging
from torch.utils.data import DataLoader
from ..models.aae import AAE
from ..utils.data import MyDataset
from ..utils.logger import Logger, plot_losses

logger = logging.getLogger(__name__)

class BootstrapAAETrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if config['device']['gpu'] and 
                                 torch.cuda.is_available() else "cpu")
        self.n_bootstrap = config['bootstrap']['n_samples']
        
    def create_bootstrap_sample(self, data, covariates):
        """Create a bootstrap sample with replacement."""
        n_samples = len(data)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return data[indices], covariates[indices]
        
    def train_bootstrap_models(self, data, covariates, output_dir):
        """Train multiple AAE models using bootstrap samples."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        condition_dim = covariates.shape[1]
        
        for i in range(self.n_bootstrap):
            logger.info(f"Training bootstrap model {i+1}/{self.n_bootstrap}")
            
            # Create loss logger for this bootstrap model
            loss_logger = Logger()
            loss_keys = ['total_loss', 'reconstruction', 'discriminator', 'generator']
            loss_logger.on_train_init(loss_keys)
            
            # Create bootstrap sample
            boot_data, boot_covariates = self.create_bootstrap_sample(data, covariates)
            
            # Initialize model
            model = AAE(
                input_dim=boot_data.shape[1],
                hidden_dim=self.config['model']['hidden_dim'],
                latent_dim=self.config['model']['latent_dim'],
                condition_dim=condition_dim,
                learning_rate=self.config['model']['learning_rate'],
                non_linear=self.config['model']['non_linear']
            ).to(self.device)
            
            # Create data loader
            dataset = MyDataset(boot_data, boot_covariates)
            loader = DataLoader(
                dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True
            )
            
            # Training loop with cyclical learning rate
            self.train_single_model(model, loader, i, output_dir, loss_logger)
            
            # Save model
            model_dir = output_dir / f"bootstrap_{i+1}"
            model_dir.mkdir(exist_ok=True)
            
            # Save model and plot losses
            torch.save(model.state_dict(), model_dir / "model.pt")
            torch.save(model, model_dir / "model.pkl")
            plot_losses(loss_logger, model_dir, title=f'_bootstrap_{i+1}', val_plot=False)
            
    def train_single_model(self, model, loader, bootstrap_idx, output_dir, loss_logger):
        """Train a single AAE model with cyclical learning rate."""
        n_samples = len(loader.dataset)
        step_size = 2 * np.ceil(n_samples / self.config['training']['batch_size'])
        base_lr = self.config['model']['learning_rate']
        max_lr = self.config['model']['max_learning_rate']
        gamma = self.config['model']['gamma']
        
        for epoch in range(self.config['training']['epochs']):
            model.train()
            epoch_losses = {
                'total_loss': 0.0,
                'reconstruction': 0.0,
                'discriminator': 0.0,
                'generator': 0.0
            }
            
            for batch_idx, (data, covariates) in enumerate(loader):
                # Update learning rate
                global_step = epoch * len(loader) + batch_idx
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
            
            # Calculate average losses for the epoch
            avg_losses = {k: v/len(loader) for k, v in epoch_losses.items()}
            
            # Log losses
            loss_logger.on_train_step(avg_losses)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Bootstrap {bootstrap_idx+1}, Epoch {epoch+1}: "
                    f"Total={avg_losses['total_loss']:.4f}, "
                    f"Recon={avg_losses['reconstruction']:.4f}, "
                    f"Disc={avg_losses['discriminator']:.4f}, "
                    f"Gen={avg_losses['generator']:.4f}"
                )