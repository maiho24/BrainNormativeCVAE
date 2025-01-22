"""
CAAE Training Implementation with proper logging
"""

import torch
import numpy as np
from pathlib import Path
import logging
from torch.utils.data import DataLoader
from ..models.aae import AAE
from ..utils.data import MyDataset
from ..utils.logger import Logger, plot_losses_aae

logger = logging.getLogger(__name__)

class AAETrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if config['device']['gpu'] and 
                                 torch.cuda.is_available() else "cpu")
        self.loss_logger = Logger()
        
    def setup_model(self, input_dim, condition_dim):
        """Initialize the AAE model with configured parameters."""
        return AAE(
            input_dim=input_dim,
            hidden_dim=self.config['model']['hidden_dim'],
            latent_dim=self.config['model']['latent_dim'],
            condition_dim=condition_dim,
            learning_rate=self.config['model']['learning_rate'],
            non_linear=self.config['model']['non_linear']
        ).to(self.device)

    def train_model(self, train_data, train_covariates, val_data, val_covariates):
        """Train the AAE model using processed data."""
        # Initialize datasets and dataloaders
        train_dataset = MyDataset(train_data, train_covariates)
        val_dataset = MyDataset(val_data, val_covariates)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )

        # Initialize model
        model = self.setup_model(
            input_dim=train_data.shape[1],
            condition_dim=train_covariates.shape[1]
        )

        # Initialize logger
        loss_keys = ['total_loss', 'reconstruction', 'discriminator', 'generator']
        self.loss_logger.on_train_init(loss_keys)
        self.loss_logger.on_val_init(loss_keys)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        base_lr = self.config['model']['learning_rate']
        max_lr = self.config['model']['max_learning_rate']
        gamma = self.config['model']['gamma']
        
        for epoch in range(self.config['training']['epochs']):
            # Train epoch
            model.train()
            train_losses = self._train_epoch(model, train_loader, epoch, base_lr, max_lr, gamma)
            val_losses = self._validate(model, val_loader)
            
            # Logging
            self.loss_logger.on_train_step(train_losses)
            self.loss_logger.on_val_step(val_losses)
            
            # Early stopping check
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                patience_counter = 0
                
                # Save best model
                model_path = Path(self.config['paths']['model_dir']) / 'best_model.pt'
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                    
            logger.info(f"Epoch {epoch + 1}/{self.config['training']['epochs']}: "
                       f"Train Loss = {train_losses['total_loss']:.4f}, "
                       f"Val Loss = {val_losses['total_loss']:.4f}")
                       
        # Plot losses at the end of training
        plot_losses_aae(self.loss_logger, self.config['paths']['model_dir'], title='_direct_training')
        return model

    def _train_epoch(self, model, train_loader, epoch, base_lr, max_lr, gamma):
        """Train for one epoch with cyclical learning rate."""
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction': 0.0,
            'discriminator': 0.0,
            'generator': 0.0
        }
        
        n_samples = len(train_loader.dataset)
        step_size = 2 * np.ceil(n_samples / self.config['training']['batch_size'])
        
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