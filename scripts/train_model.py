import argparse
import yaml
from pathlib import Path
import sys
import logging
import torch
import pandas as pd
import numpy as np

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.train import train_model
from src.training.optuna_trainer import OptunaTrainer
from src.models.cvae import cVAE
from src.utils.data import MyDataset_labels
from src.utils.visualization import Logger, plot_losses

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def process_covariates(covariates_df):
    """Process covariates into the format needed by the model."""
    # Extract continuous variables
    age_icv = covariates_df[['Age', 'wb_EstimatedTotalIntraCranial_Vol']].values
    
    # Process categorical variables
    one_hot_sex = pd.get_dummies(covariates_df['Sex'], prefix='Sex').values
    one_hot_diabetes = pd.get_dummies(covariates_df['Diabetes Status'], prefix='Diabetes').values
    one_hot_smoking = pd.get_dummies(covariates_df['Smoking Status'], prefix='Smoking').values
    one_hot_hypercholesterolemia = pd.get_dummies(covariates_df['Hypercholesterolemia Status'], prefix='Hypercholesterolemia').values
    one_hot_obesity = pd.get_dummies(covariates_df['Obesity Status'], prefix='Obesity').values
    
    # Combine all covariates
    return np.hstack((
        age_icv, 
        one_hot_sex,
        one_hot_diabetes, 
        one_hot_smoking,
        one_hot_hypercholesterolemia, 
        one_hot_obesity
    ))

def load_and_preprocess_data(config):
    """Load and preprocess training and validation data."""
    logger.info("Loading data...")
    
    # Load raw data
    data_path = Path(config['paths']['data_dir'])
    train_data = pd.read_csv(data_path / 'train_data_subset.csv')
    train_covariates = pd.read_csv(data_path / 'train_covariates_subset.csv')
    test_data = pd.read_csv(data_path / 'test_data_subset.csv')
    test_covariates = pd.read_csv(data_path / 'test_covariates_subset.csv')
    
    # Process covariates
    train_covariates_processed = process_covariates(train_covariates)
    test_covariates_processed = process_covariates(test_covariates)
    
    # Convert data to numpy arrays
    train_data_np = train_data.to_numpy()
    test_data_np = test_data.to_numpy()
    
    # Split training data into train and validation sets
    val_size = config['training']['validation_split']
    indices = np.arange(len(train_data_np))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(len(indices) * (1 - val_size))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create train and validation sets
    train_data_split = train_data_np[train_indices]
    train_cov_split = train_covariates_processed[train_indices]
    val_data_split = train_data_np[val_indices]
    val_cov_split = train_covariates_processed[val_indices]
    
    return (train_data_split, train_cov_split, 
            val_data_split, val_cov_split,
            test_data_np, test_covariates_processed)

def main():
    parser = argparse.ArgumentParser(description="Train Normative cVAE Model")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, help='Override data directory from config')
    parser.add_argument('--output_dir', type=str, help='Override output directory from config')
    parser.add_argument('--mode', type=str, choices=['direct', 'optuna'], default='direct',
                      help='Training mode: direct training or Optuna optimization')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    config['device']['gpu'] = args.gpu

    # Create output directories
    output_dir = Path(config['paths']['output_dir'])
    model_dir = Path(config['paths']['model_dir'])
    for directory in [output_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Save the configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    try:
        # Load and preprocess data
        (train_data, train_covariates, 
         val_data, val_covariates,
         test_data, test_covariates) = load_and_preprocess_data(config)

        # Set device
        device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if args.mode == 'optuna':
            logger.info("Starting Optuna hyperparameter optimization...")
            trainer = OptunaTrainer(
                train_data=train_data,
                train_covariates=train_covariates,
                val_data=val_data,
                val_covariates=val_covariates,
                config=config
            )
            model, best_params = trainer.run_optimization()
            
            # Save best parameters
            with open(output_dir / 'best_params.yaml', 'w') as f:
                yaml.dump(best_params, f)
                
        else:
            logger.info("Starting direct training with provided configuration...")
            # Create model
            model = cVAE(
                input_dim=train_data.shape[1],
                hidden_dim=config['model']['hidden_dim'],
                latent_dim=config['model']['latent_dim'],
                c_dim=train_covariates.shape[1],
                learning_rate=config['model']['learning_rate'],
                beta=config['model']['beta'],
                non_linear=config['model']['non_linear']
            )
            model = model.to(device)
            
            # Train model
            model = train_model(
                train_data=train_data,
                train_covariates=train_covariates,
                val_data=val_data,
                val_covariates=val_covariates,
                config=config
            )

        # Save final model
        torch.save(model, model_dir / 'final_model.pkl')
        logger.info("Training completed successfully")
        
        # Optional: Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_data_tensor = torch.FloatTensor(test_data).to(device)
            test_covariates_tensor = torch.FloatTensor(test_covariates).to(device)
            _, test_recon_var = model.pred_recon(test_data_tensor, test_covariates_tensor, device)
            
            # Save test reconstruction variances
            np.savetxt(
                output_dir / 'test_reconstruction_vars.csv',
                test_recon_var,
                delimiter=','
            )

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()