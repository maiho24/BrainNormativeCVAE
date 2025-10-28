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
        
        Bootstrap analysis can be enabled or disabled. When disabled, simple sampling is used
        to obtain conditional means and variances without confidence intervals.
        
        Example usage:
          brain-cvae-inference --model_path models/final_model.pkl --config configs/my_config.yaml --prediction_type covariate
          brain-cvae-inference --model_path models/final_model.pkl --num_samples 2000 --bootstrap --num_bootstraps 1500 --prediction_type dual_input
          brain-cvae-inference --model_path models/final_model.pkl --num_samples 1000 --no-bootstrap --prediction_type covariate
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
        help='Number of samples for sampling analysis (default: 1000)'
    )
    
    # Bootstrap arguments with mutually exclusive group
    bootstrap_group = parser.add_mutually_exclusive_group()
    bootstrap_group.add_argument(
        '--bootstrap',
        action='store_true',
        default=True,
        help='Enable bootstrap analysis (default: enabled)'
    )
    
    bootstrap_group.add_argument(
        '--no-bootstrap',
        action='store_false',
        dest='bootstrap',
        help='Disable bootstrap analysis and use simple sampling instead'
    )
    
    parser.add_argument(
        '--num_bootstraps', 
        type=int, 
        default=1000,
        help='Number of bootstrap iterations (default: 1000, only used when bootstrap is enabled)'
    )
    
    parser.add_argument(
        '--confidence_level',
        type=float,
        default=0.95,
        help='Confidence level for bootstrap confidence intervals (default: 0.95, only used when bootstrap is enabled)'
    )
    
    parser.add_argument(
        '--prediction_type',
        type=str,
        choices=['covariate', 'dual_input'],
        default='covariate',
        help='Method for prediction: covariate (using only covariates), or dual_input (using both observed data and covariates)'
    )
    
    parser.add_argument(
        '--calibrator_path', 
        type=str,
        default=None,
        help='Path to the fitted calibrator (Optional)'
    )
    
    parser.add_argument(
        '--gpu', 
        action='store_true',
        help='Use GPU for inference if available'
    )
    
    parser.add_argument(
        '--summary_report', 
        action='store_true',
        help='Generate summary reports'
    )
    
    parser.add_argument(
        '--suffix', 
        type=str,
        default='',
        help="Suffix for the output folder's name"
    )

    return parser

def validate_args(args, parser):
    """Validate that required arguments are present based on conditions."""
    if not args.config and (not args.data_dir or not args.output_dir):
        parser.error("If --config is not provided, both --data_dir and --output_dir are required")
    
    # Validate confidence level
    if args.bootstrap and not (0 < args.confidence_level < 1):
        parser.error("Confidence level must be between 0 and 1 (exclusive)")
    
    return args

def create_default_config(args):
    """Create a default configuration dictionary from command line arguments."""
    config = {
        'paths': {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'calibrator_path': args.calibrator_path
        },
        'device': {
            'gpu': args.gpu
        },
        'inference': {
            'num_samples': args.num_samples,
            'bootstrap': args.bootstrap
        }
    }
    
    # Only add bootstrap-specific parameters if bootstrap is enabled
    if args.bootstrap:
        config['inference']['num_bootstraps'] = args.num_bootstraps
        config['inference']['confidence_level'] = args.confidence_level
    
    return config
    
def run_inference(args, config):
    """Separate function containing all inference-related code and imports."""
    import torch
    import logging
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    sys.path.append(str(Path(__file__).parent.parent))
    
    from src.inference.bootstrap import (
        generate_bootstrap_stats_by_covariates, 
        generate_bootstrap_stats_from_encoded, 
        generate_simple_stats_by_covariates,
        generate_simple_stats_from_encoded,
        compute_feature_importance, 
        generate_summary_statistics
    )
    from src.models.cvae import cVAE
    from src.utils.data import load_test_data, process_covariates
    from src.utils.logger import setup_logging

    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, 'inference')
    
    # Update config with command line bootstrap setting
    config['inference']['bootstrap'] = args.bootstrap
    
    try:
        logger.info("Loading model...")
        model = torch.load(args.model_path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()

        logger.info("Loading and processing test data...")
        test_data, test_covariates_raw, test_covariates_processed = load_test_data(config['paths']['data_dir'], logger)
        
        feature_cols = test_data.columns.tolist()
        
        # Determine which analysis method to use
        use_bootstrap = config['inference']['bootstrap']
        
        if use_bootstrap:
            logger.info(f"Bootstrap analysis enabled with {config['inference']['num_bootstraps']} iterations")
            if args.prediction_type == 'covariate':
                logger.info("Starting covariate-based predictions with bootstrap...")
                results = generate_bootstrap_stats_by_covariates(
                    model=model,
                    covariates=test_covariates_processed,
                    covariates_df=test_covariates_raw,
                    feature_cols=feature_cols,
                    config=config,
                    num_samples=config['inference']['num_samples'],
                    num_bootstraps=config['inference']['num_bootstraps'],
                    confidence_level=config['inference'].get('confidence_level', 0.95),
                    suffix=args.suffix
                )
            else:  # dual_input
                logger.info("Starting dual-input predictions with bootstrap...")
                results = generate_bootstrap_stats_from_encoded(
                    model=model,
                    features=test_data,
                    covariates=test_covariates_processed,
                    covariates_raw=test_covariates_raw,
                    feature_cols=feature_cols,
                    config=config,
                    num_samples=config['inference']['num_samples'],
                    num_bootstraps=config['inference']['num_bootstraps'],
                    confidence_level=config['inference'].get('confidence_level', 0.95),
                    suffix=args.suffix
                )
        else:
            logger.info("Bootstrap analysis disabled - using simple sampling")
            if args.prediction_type == 'covariate':
                logger.info("Starting covariate-based predictions with simple sampling...")
                results = generate_simple_stats_by_covariates(
                    model=model,
                    covariates=test_covariates_processed,
                    covariates_df=test_covariates_raw,
                    feature_cols=feature_cols,
                    config=config,
                    num_iterations=config['inference']['num_samples'],
                    suffix=args.suffix
                )
            else:  # dual_input
                logger.info("Starting dual-input predictions with simple sampling...")
                results = generate_simple_stats_from_encoded(
                    model=model,
                    features=test_data,
                    covariates=test_covariates_processed,
                    covariates_raw=test_covariates_raw,
                    feature_cols=feature_cols,
                    config=config,
                    num_iterations=config['inference']['num_samples'],
                    suffix=args.suffix
                )
        if args.summary_report:    
            logger.info(f"Computing feature importance for {args.prediction_type} predictions...")
            feature_variability, covariate_sensitivity = compute_feature_importance(
                results=results,
                covariates_df=test_covariates_raw,
                feature_cols=feature_cols,
                config=config,
                suffix=args.suffix
            )
            
            logger.info(f"Generating summary statistics for {args.prediction_type} predictions...")
            summary_stats = generate_summary_statistics(
                results=results,
                feature_cols=feature_cols,
                config=config,
                suffix=args.suffix
            )
        
        # Log completion message with analysis type
        analysis_type = "bootstrap" if use_bootstrap else "simple sampling"
        logger.info(f"Inference completed successfully using {analysis_type} analysis")
        
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

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Override config with command line arguments if provided
        if args.data_dir:
            config['paths']['data_dir'] = args.data_dir
        if args.output_dir:
            config['paths']['output_dir'] = args.output_dir
        if args.calibrator_path:
            config['paths']['calibrator_path'] = args.calibrator_path
        if 'inference' not in config:
            config['inference'] = {}
        config['inference']['num_samples'] = args.num_samples
        config['inference']['bootstrap'] = args.bootstrap
        if args.bootstrap:
            config['inference']['num_bootstraps'] = args.num_bootstraps
            config['inference']['confidence_level'] = args.confidence_level
    else:
        config = create_default_config(args)
    
    # Ensure device config exists
    if 'device' not in config:
        config['device'] = {}
    config['device']['gpu'] = args.gpu
    
    run_inference(args, config)

if __name__ == '__main__':
    main()