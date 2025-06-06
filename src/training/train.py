import torch
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import yaml
from pathlib import Path
import pickle

from ..utils.data import MyDataset
from ..utils.logger import Logger, plot_losses
from ..models.cvae import cVAE, VarianceType

logger = logging.getLogger(__name__)

            
def parse_hidden_dim(hidden_dim):
    """
    Parse hidden dimensions from string format.
    
    Args:
        hidden_dim: String in format "dim1_dim2" or "dim1"
        
    Returns:
        List of integers representing hidden dimensions
    """
    if not isinstance(hidden_dim, str):
        raise ValueError(f"Hidden dimensions must be a string (e.g., '64_32'), got {type(hidden_dim)}")
    
    try:
        return [int(x) for x in hidden_dim.split('_')]
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing hidden dimensions {hidden_dim}: {str(e)}")
        raise ValueError(f"Invalid hidden dimension format: {hidden_dim}. Expected format: 'dim1_dim2' or 'dim1'")

def train_model(train_data, train_covariates, val_data, val_covariates, config):
    """Train the cVAE model with the given configurations."""
    device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
    
    train_dataset = MyDataset(train_data, train_covariates)
    val_dataset = MyDataset(val_data, val_covariates)
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    variance_type = config['model'].get('variance_type', 'global_learnable')
    
    if variance_type == "covariate_specific":
        varnet_hidden_dim = parse_hidden_dim(config['model']['varnet_hidden_dim'])
    else:
        varnet_hidden_dim = None
    
    model = cVAE(
        input_dim=train_data.shape[1],
        hidden_dim=parse_hidden_dim(config['model']['hidden_dim']),
        latent_dim=config['model']['latent_dim'],
        c_dim=train_covariates.shape[1],
        beta=config['model'].get('beta', 1),
        non_linear=config['model']['non_linear'],
        variance_type=variance_type,
        variance_network_hidden_dim=varnet_hidden_dim
    )
    model.to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['model']['learning_rate']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=1e-6
    )

    train_logger = Logger()
    train_logger.on_train_init(['Total Loss', 'KL Divergence', 'Reconstruction Loss'])
    train_logger.on_val_init(['Total Loss', 'KL Divergence', 'Reconstruction Loss'])
    
    best_loss = float('inf')
    patience_counter = 0
    
    models_dir = Path(config['paths']['model_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)

    #------ TRAINING & VALIDATION ------#
    for epoch in range(config['training']['epochs']):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses["Total Loss"] += loss['Total Loss'].item()
            train_losses["KL Divergence"] += loss['KL Divergence'].item()
            train_losses["Reconstruction Loss"] += loss['Reconstruction Loss'].item()
            num_batches += 1

        avg_train_losses = {k: v/num_batches for k, v in train_losses.items()}
        current_train_loss = avg_train_losses["Total Loss"]
        
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
        if current_val_loss < best_loss:
            best_loss = current_val_loss
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }
            best_checkpoint_path = models_dir / "best_model_checkpoint.pt"
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f'Epoch {epoch}: New best model saved')
            
            logger_path = models_dir / "training_logger.pkl"
            with open(logger_path, 'wb') as f:
                pickle.dump(train_logger, f)
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        train_logger.on_train_step(avg_train_losses)
        train_logger.on_val_step(avg_val_losses)

        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss = {avg_train_losses['Total Loss']:.4f} "
            f"(KL: {avg_train_losses['KL Divergence']:.4f}, "
            f"NegLogLik: {avg_train_losses['Reconstruction Loss']:.4f}), "
            f"Val Loss = {avg_val_losses['Total Loss']:.4f} "
            f"(KL: {avg_val_losses['KL Divergence']:.4f}, "
            f"NegLogLik: {avg_val_losses['Reconstruction Loss']:.4f})"
        )

    plot_losses(train_logger, models_dir, '_training_validation')
    
    best_checkpoint_path = models_dir / f"best_model_checkpoint.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model with validation loss: {checkpoint['loss']}")
    else:
        logger.warning("No best model checkpoint found. Returning the final model state.")
        
    return model