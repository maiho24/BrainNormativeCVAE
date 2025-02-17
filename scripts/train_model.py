import argparse
import yaml
from pathlib import Path
import sys

def create_parser():
    """Create argument parser with detailed help messages."""
    parser = argparse.ArgumentParser(
        description="""
        Brain Normative cVAE Model Training
        
        This script trains a conditional Variational Autoencoder (cVAE) for normative modeling 
        of brain imaging data. It supports both direct training with specified parameters and 
        hyperparameter optimization using Optuna.
        
        Example usage:
          brain-cvae-train --config configs/my_config.yaml --mode direct --gpu
          brain-cvae-train --config configs/optuna_config.yaml --mode optuna
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default_config.yaml',
        help='Path to configuration file (default: configs/default_config.yaml)'
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str,
        help='Override data directory specified in config file'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str,
        help='Override output directory specified in config file'
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['direct', 'optuna'],
        default='direct',
        help='''Training mode:
        direct: Train with parameters specified in config file
        optuna: Perform hyperparameter optimization (default: direct)'''
    )
    
    parser.add_argument(
        '--gpu', 
        action='store_true',
        help='Use GPU for training if available'
    )
    
    return parser

def run_training(args, config):
    """Separate function containing all training-related code and imports."""
    import logging
    import torch
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    sys.path.append(str(Path(__file__).parent.parent))
    
    from src.training.train import train_model
    from src.training.optuna_trainer import OptunaTrainer
    from src.models.cvae import cVAE
    from src.utils.data import MyDataset, process_covariates, load_train_data
    from src.utils.logger import Logger, plot_losses, setup_logging

    output_dir = Path(config['paths']['output_dir'])
    model_dir = output_dir / 'models'
    config['paths']['model_dir'] = str(model_dir)
    for directory in [output_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, 'training')
    
    config_file = model_dir / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Saved configuration to {config_file}")

    try:
        data_path = Path(config['paths']['data_dir'])
        val_size = config['training']['validation_split']
        (train_data, train_covariates, 
         val_data, val_covariates,
         test_data, test_covariates) = load_train_data(data_path, val_size, logger)

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
                
        else:
            logger.info("Starting direct training with provided configuration...")
            model = train_model(
                train_data=train_data,
                train_covariates=train_covariates,
                val_data=val_data,
                val_covariates=val_covariates,
                config=config
            )

        model_file = Path(config['paths']['model_dir']) / 'final_model.pkl'
        torch.save(model, model_file)
        logger.info(f"Saved final model to {model_file}")
        logger.info("Training completed successfully")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
        
def main():
    parser = create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()

    if len(sys.argv) > 1 and not (len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        if args.data_dir:
            config['paths']['data_dir'] = args.data_dir
        if args.output_dir:
            config['paths']['output_dir'] = args.output_dir
        config['device']['gpu'] = args.gpu

        run_training(args, config)

if __name__ == '__main__':
    main()