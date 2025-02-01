import argparse
import sys
import yaml
from pathlib import Path


class ConditionalRequiredAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        
def create_parser():
    """Create argument parser with detailed help messages."""
    parser = argparse.ArgumentParser(
        description="""
        Brain Normative cVAE Model Inference
        
        This script runs inference on a trained conditional Variational Autoencoder (cVAE) model.
        Two prediction methods are available:
        1. Covariate: Standard normative prediction using only covariates (default)
        2. Dual-input: Prediction using both observed data and covariates
        
        Example usage:
          brain-cvae-inference --model_path models/final_model.pkl --config configs/my_config.yaml --prediction_type covariate
          brain-cvae-inference --model_path models/final_model.pkl --num_samples 2000 --num_bootstraps 1500 --prediction_type dual_input
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration file. If not provided, --data_dir and --output_dir must be specified'
    )
    
    parser.add_argument(
        '--model_path', 
        type=str,
        required=True,
        help='Path to trained model checkpoint (required)'
    )
    
    parser.add_argument(
        '--data_dir', 
        type=str,
        action=ConditionalRequiredAction,
        help='Directory containing input data (required if --config not provided)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str,
        action=ConditionalRequiredAction,
        help='Directory for output files (required if --config not provided)'
    )
    
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=1000,
        help='Number of samples for bootstrap analysis (default: 1000)'
    )
    
    parser.add_argument(
        '--num_bootstraps', 
        type=int, 
        default=1000,
        help='Number of bootstrap iterations (default: 1000)'
    )
    
    parser.add_argument(
        '--prediction_type',
        type=str,
        choices=['covariate', 'dual_input'],
        default='covariate',
        help='Method for prediction: covariate (using only covariates), or dual_input (using both observed data and covariates)'
    )
    
    parser.add_argument(
        '--gpu', 
        action='store_true',
        help='Use GPU for inference if available'
    )
    
    return parser

def validate_args(args, parser):
    """Validate that required arguments are present based on conditions."""
    if not args.config and (not args.data_dir or not args.output_dir):
        parser.error("If --config is not provided, both --data_dir and --output_dir are required")
    return args

def create_default_config(args):
    """Create a default configuration dictionary from command line arguments."""
    return {
        'paths': {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir
        },
        'device': {
            'gpu': args.gpu
        }
    }
    
def run_inference(args, config):
    """Separate function containing all inference-related code and imports."""
    import torch
    import logging
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # Add project root to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    # Import project-specific modules
    from src.inference.bootstrap import (
        generate_bootstrap_stats_by_covariates, 
        generate_bootstrap_stats_from_encoded, 
        compute_feature_importance, 
        generate_summary_statistics
    )
    from src.models.cvae import cVAE
    from src.utils.data import load_test_data, process_covariates
    from src.utils.logger import setup_logging

    # Create output directory structure
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, 'inference')
    
    try:
        # Load model
        logger.info("Loading model...")
        model = torch.load(args.model_path, map_location=torch.device('cpu'))
        model.eval()

        # Load and process test data
        logger.info("Loading and processing test data...")
        test_data, test_covariates_raw, test_covariates_processed = load_test_data(config['paths']['data_dir'], logger)
        
        # Extract features' column names
        feature_cols = test_data.columns.tolist()
        
        # Run predictions based on specified method
        if args.prediction_type == 'covariate':
            logger.info("Starting covariate-based predictions...")
            results = generate_bootstrap_stats_by_covariates(
                model=model,
                covariates=test_covariates_processed,
                covariates_df=test_covariates_raw,
                feature_cols=feature_cols,
                config=config,
                num_samples=args.num_samples,
                num_bootstraps=args.num_bootstraps
            )
        else:  # dual_input
            logger.info("Starting dual-input predictions...")
            results = generate_bootstrap_stats_from_encoded(
                model=model,
                features=test_data,
                covariates=test_covariates_processed,
                covariates_raw=test_covariates_raw,
                feature_cols=feature_cols,
                config=config,
                num_samples=args.num_samples,
                num_bootstraps=args.num_bootstraps
            )
            
        # Compute feature importance
        logger.info(f"Computing feature importance for {args.prediction_type} predictions...")
        feature_variability, covariate_sensitivity = compute_feature_importance(
            bootstrap_results=results,
            covariates_df=test_covariates_raw,
            feature_cols=feature_cols,
            config=config,
            prediction_type=args.prediction_type
        )
        
        # Generate summary statistics
        logger.info(f"Generating summary statistics for {args.prediction_type} predictions...")
        summary_stats = generate_summary_statistics(
            bootstrap_results=results,
            feature_cols=feature_cols,
            config=config,
            prediction_type=args.prediction_type
        )
        
        logger.info("Inference completed successfully")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise
    
def main():
    parser = create_parser()
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    args = validate_args(args, parser)

    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Override config with command line arguments if provided
        if args.data_dir:
            config['paths']['data_dir'] = args.data_dir
        if args.output_dir:
            config['paths']['output_dir'] = args.output_dir
    else:
        # Create default config from command line arguments
        config = create_default_config(args)

    config['device']['gpu'] = args.gpu
    
    run_inference(args, config)

if __name__ == '__main__':
    main()
