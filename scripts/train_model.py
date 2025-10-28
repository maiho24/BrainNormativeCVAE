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
        hyperparameter optimization using Optuna. When using Optuna mode, k-fold cross-validation
        can be enabled for more robust hyperparameter selection.
        
        Example usage:
          brain-cvae-train --config configs/my_config.yaml --mode direct --gpu
          brain-cvae-train --config configs/optuna_config.yaml --mode optuna
          brain-cvae-train --config configs/optuna_cv_config.yaml --mode optuna  # with cross-validation
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
        optuna: Perform hyperparameter optimization with optional k-fold CV (default: direct)'''
    )
    
    parser.add_argument(
        '--gpu', 
        action='store_true',
        help='Use GPU for training if available'
    )
    
    # Calibration arguments with mutually exclusive group
    calibration_group = parser.add_mutually_exclusive_group()
    calibration_group.add_argument(
        '--calibration',
        action='store_true',
        default=False,
        help='Enable post-training calibration (default: False)'
    )
    
    calibration_group.add_argument(
        '--no-calibration',
        action='store_false',
        dest='calibration',
        help='Disable post-training calibration'
    )
    
    parser.add_argument(
        '--suffix', 
        type=str,
        default='',
        help="Suffix for the output folder's name"
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
    model_dir = output_dir / f'model{args.suffix}'
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
        
        # Check if calibration is enabled for proper data loading
        enable_calibration = config.get('calibration', {}).get('enabled', False)
        calibration_split = config.get('calibration', {}).get('calibration_split', 0.3)
        
        # Load data with automatic calibration split
        data_splits = load_train_data(data_path, val_size, logger, enable_calibration, calibration_split)
        
        if enable_calibration:
            (train_data, train_covariates, 
             val_data, val_covariates,
             test_data, test_covariates,
             cal_data, cal_covariates) = data_splits
            logger.info(f"Calibration enabled: Using {len(cal_data)} samples for variance calibration")
        else:
            (train_data, train_covariates, 
             val_data, val_covariates,
             test_data, test_covariates) = data_splits
            cal_data, cal_covariates = None, None
            logger.info("Calibration disabled: Using standard training approach")

        device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        if args.mode == 'optuna':
            # Check if cross-validation is enabled
            cv_enabled = config.get('cross_validation', {}).get('enabled', False)
            if cv_enabled:
                cv_folds = config.get('cross_validation', {}).get('n_folds', 5)
                cv_type = "stratified" if config.get('cross_validation', {}).get('stratified', False) else "standard"
                logger.info(f"Starting Optuna hyperparameter optimisation with {cv_folds}-fold {cv_type} cross-validation...")
            else:
                logger.info("Starting Optuna hyperparameter optimisation...")
                
            trainer = OptunaTrainer(
                train_data=train_data,
                train_covariates=train_covariates,
                val_data=val_data,
                val_covariates=val_covariates,
                config=config,
                cal_data=cal_data,
                cal_covariates=cal_covariates
            )
            model, best_params = trainer.run_optimization()
                
        else:
            logger.info("Starting direct training with provided configuration...")
            model = train_model(
                train_data=train_data,
                train_covariates=train_covariates,
                val_data=val_data,
                val_covariates=val_covariates,
                config=config,
                cal_data=cal_data,
                cal_covariates=cal_covariates
            )

        model_file = Path(config['paths']['model_dir']) / 'final_model.pkl'
        torch.save(model, model_file)
        logger.info(f"Saved final model to {model_file}")
        
        # Log calibration status
        calibrator_path = Path(config['paths']['model_dir']) / 'variance_calibrator.pkl'
        if calibrator_path.exists():
            logger.info(f"Training completed successfully with variance calibration")
        else:
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
        if 'device' not in config:
            config['device'] = {}
            config['device']['gpu'] = args.gpu
        if 'calibration' not in config:
            config['calibration'] = {}
            config['calibration']['enabled'] = args.calibration

        run_training(args, config)

if __name__ == '__main__':
    main()