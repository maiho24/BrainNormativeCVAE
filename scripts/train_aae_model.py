#!/usr/bin/env python3
"""
Script to train the Adversarial Autoencoder (AAE) model.
Supports both direct training and bootstrap training modes.
"""

import argparse
import yaml
from pathlib import Path
import sys
import logging
import torch
import numpy as np
import random
import pandas as pd
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

def create_parser():
    """Create argument parser with detailed help messages."""
    parser = argparse.ArgumentParser(
        description="""
        Brain Normative AAE Model Training
        
        This script trains an Adversarial Autoencoder (AAE) for normative modeling 
        of brain imaging data. It supports both direct training and bootstrap training
        for robustness assessment.
        
        Example usage:
          train-aae --config configs/aae_config.yaml --mode direct
          train-aae --config configs/bootstrap_aae_config.yaml --mode bootstrap
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/aae_config.yaml',
        help='Path to configuration file'
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
        choices=['direct', 'bootstrap'],
        default='direct',
        help='Training mode: direct (single model) or bootstrap (multiple models)'
    )
    
    parser.add_argument(
        '--gpu', 
        action='store_true',
        help='Use GPU for training if available'
    )
    
    return parser

def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_training(args, config):
    """Execute the training process based on the specified mode."""
    # Import project modules
    sys.path.append(str(Path(__file__).parent.parent))
    from src.training.aae_trainer import AAETrainer
    from src.training.bootstrap_aae_trainer import BootstrapAAETrainer
    from src.utils.data import load_train_data
    from src.utils.logger import setup_logging
    
    # Create output directories
    output_dir = Path(config['paths']['output_dir'])
    model_dir = output_dir / 'models'
    config['paths']['model_dir'] = str(model_dir)
    for directory in [output_dir, model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(output_dir, 'aae_training')
    
    # Save configuration
    config_file = model_dir / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    logger.info(f"Saved configuration to {config_file}")

    try:
        # Set random seeds
        set_random_seeds(config.get('bootstrap', {}).get('random_seed', 42))
        
        # Load data
        logger.info("Loading training data...")
        data_path = Path(config['paths']['data_dir'])
        val_size = config['training'].get('validation_split', 0.2)
        
        (train_data, train_covariates, 
         val_data, val_covariates,
         test_data, test_covariates) = load_train_data(data_path, val_size, logger)

        # Set device
        device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        config['device']['gpu'] = device.type == 'cuda'

        if args.mode == 'bootstrap' or config['training']['mode'] == 'bootstrap':
            logger.info("Starting bootstrap training...")
            trainer = BootstrapAAETrainer(config)
            trainer.train_bootstrap_models(
                train_data, 
                train_covariates,
                model_dir / 'bootstrap'
            )
            logger.info("Bootstrap training completed")
            
        else:  # direct mode
            logger.info("Starting direct training...")
            trainer = AAETrainer(config)
            model = trainer.train_model(
                train_data, 
                train_covariates,
                val_data,
                val_covariates
            )
            
            # Save final model
            model_path = model_dir / 'final_model.pt'
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved final model to {model_path}")
            
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

        # Override config with command line arguments
        if args.data_dir:
            config['paths']['data_dir'] = args.data_dir
        if args.output_dir:
            config['paths']['output_dir'] = args.output_dir
        config['device']['gpu'] = args.gpu

        run_training(args, config)

if __name__ == '__main__':
    main()