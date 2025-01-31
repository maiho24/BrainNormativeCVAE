import torch
import torch.utils.data as data
import logging
import yaml
from pathlib import Path

from ..utils.data import MyDataset
from ..utils.logger import Logger, plot_losses
from ..models.cvae import cVAE

logger = logging.getLogger(__name__)

def validate_model(model, generator_val, device):
    """Run validation step."""
    total_val_loss = 0
    val_kl_loss = 0
    val_ll_loss = 0
    num_batches = 0

    with torch.no_grad():
        for data_val, cov_val in generator_val:
            data_val, cov_val = data_val.to(device), cov_val.to(device)
            fwd_rtn_val = model.forward(data_val, cov_val)
            val_loss = model.loss_function(data_val, fwd_rtn_val)

            total_val_loss += val_loss['Total Loss'].item()
            val_kl_loss += val_loss['KL Divergence'].item()
            val_ll_loss += val_loss['Reconstruction Loss'].item()
            num_batches += 1

    return (total_val_loss / num_batches,
            val_kl_loss / num_batches,
            val_ll_loss / num_batches)
            
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
    # Setup device
    device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create datasets and dataloaders
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

    # Initialize model
    model = cVAE(
        input_dim=train_data.shape[1],
        hidden_dim=parse_hidden_dim(config['model']['hidden_dim']),
        latent_dim=config['model']['latent_dim'],
        c_dim=train_covariates.shape[1],
        learning_rate=config['model']['learning_rate'],
        beta=config['model']['beta'],
        non_linear=config['model']['non_linear']
    )
    model.to(device)

    # Initialize logger and training variables
    train_logger = Logger()
    train_logger.on_train_init(['total_loss', 'KL', 'neg_LL'])
    train_logger.on_val_init(['total_loss', 'KL', 'neg_LL'])
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Save paths
    models_dir = Path(config['paths']['model_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        total_train_loss = 0
        total_kl_loss = 0
        total_ll_loss = 0
        num_batches = 0
        
        for batch_data, batch_cov in train_loader:
            batch_data = batch_data.to(device)
            batch_cov = batch_cov.to(device)
            
            fwd_rtn = model.forward(batch_data, batch_cov)
            loss = model.loss_function(batch_data, fwd_rtn)
            
            model.optimizer.zero_grad()
            loss['Total Loss'].backward()
            model.optimizer.step()
            
            total_train_loss += loss['Total Loss'].item()
            total_kl_loss += loss['KL Divergence'].item()
            total_ll_loss += loss['Reconstruction Loss'].item()
            num_batches += 1

        # Calculate average training losses
        avg_train_loss = total_train_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        avg_ll_loss = total_ll_loss / num_batches

        # Validation phase
        model.eval()
        avg_val_loss, avg_val_kl_loss, avg_val_ll_loss = validate_model(
            model, val_loader, device
        )

        # Early stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model.state_dict()
            patience_counter = 0
            logger.info(f'Epoch {epoch}: New best model saved')
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Log losses
        train_logger.on_train_step({
            'total_loss': avg_train_loss,
            'KL': avg_kl_loss,
            'neg_LL': avg_ll_loss
        })
        train_logger.on_val_step({
            'total_loss': avg_val_loss,
            'KL': avg_val_kl_loss,
            'neg_LL': avg_val_ll_loss
        })

        logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Plot and save training curves
    plot_losses(train_logger, models_dir, '_training_validation')
    
    # Load and return best model
    model.load_state_dict(best_model)
    return model