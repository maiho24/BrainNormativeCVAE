import argparse
import yaml
from pathlib import Path
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.inference.inference import run_inference
from src.inference.bootstrap import generate_bootstrap_stats_by_covariates
from src.models.cvae import load_model
from src.utils.data import load_test_data, process_covariates

def create_parser():
    """Create argument parser with detailed help messages."""
    parser = argparse.ArgumentParser(
        description="""
        Brain Normative cVAE Model Inference
        
        This script runs inference on a trained conditional Variational Autoencoder (cVAE) model.
        It performs two main tasks:
        1. Computes reconstruction variances for test data
        2. Generates bootstrap statistics for different covariate combinations
        
        Example usage:
          brain-cvae-inference --model_path models/final_model.pkl --config configs/my_config.yaml
          brain-cvae-inference --model_path models/final_model.pkl --num_samples 2000 --num_bootstraps 1500
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
        '--model_path', 
        type=str,
        required=True,
        help='Path to trained model checkpoint (required)'
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
        '--gpu', 
        action='store_true',
        help='Use GPU for inference if available'
    )
    
    return parser

def setup_logging(output_dir, script_name):
    """Set up logging with timestamp in filename."""
    # Create logs directory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{script_name}_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file))
        ]
    )
    return logging.getLogger(__name__)

def main():
    # Create parser with detailed help
    parser = create_parser()
    
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
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

    # Create output directory structure
    output_dir = Path(config['paths']['output_dir'])
    results_dir = output_dir / 'results'
    for directory in [output_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, 'inference')
    
    try:
        # Load model
        logger.info("Loading model...")
        model = load_model(args.model_path)

        # Load and process test data
        logger.info("Loading and processing test data...")
        test_data, test_covariates_raw = load_test_data(config['paths']['data_dir'])
        test_covariates_processed = process_covariates(test_covariates_raw)

        # Run inference with processed covariates
        logger.info("Computing reconstruction variances...")
        recon_vars = run_inference(model, test_data, test_covariates_processed, config)
        np.savetxt(results_dir / 'reconstruction_variances.csv', recon_vars, delimiter=',')

        # Run bootstrap analysis
        logger.info("Starting bootstrap analysis...")
        bootstrap_results = generate_bootstrap_stats_by_covariates(
            model=model,
            covariates_df=test_covariates_raw,
            config=config,
            num_samples=args.num_samples,
            num_bootstraps=args.num_bootstraps
        )

        # Save bootstrap results
        for filename, df in bootstrap_results.items():
            df.to_csv(results_dir / filename, index=False)

        logger.info("Inference and bootstrap analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == '__main__':
    main()