import argparse
import yaml
from pathlib import Path
import sys
import logging

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.inference import run_inference
from src.inference.bootstrap import generate_bootstrap_stats_by_covariates
from src.models.cvae import load_model
from src.utils.data import load_test_data, process_covariates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Inference with Normative cVAE Model")
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, help='Override data directory from config')
    parser.add_argument('--output_dir', type=str, help='Override output directory from config')
    parser.add_argument('--num_samples', type=int, default=1000,
                      help='Number of samples for bootstrap analysis')
    parser.add_argument('--num_bootstraps', type=int, default=1000,
                      help='Number of bootstrap iterations')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir

    # Create output directory
    Path(config['paths']['output_dir']).mkdir(parents=True, exist_ok=True)

    try:
        # Load model
        logger.info("Loading model...")
        model = load_model(args.model_path)

        # Load test data
        logger.info("Loading test data...")
        test_data, test_covariates = load_test_data(config['paths']['data_dir'])

        # Get reconstruction variances
        logger.info("Computing reconstruction variances...")
        run_inference(model, test_data, test_covariates, config)

        # Run bootstrap analysis
        logger.info("Starting bootstrap analysis...")
        covariates_df = pd.read_csv(
            Path(config['paths']['data_dir']) / 'test_covariates_subset.csv'
        )
        
        # Select relevant columns for covariates
        covariates_df = covariates_df[['Age', 'wb_EstimatedTotalIntraCranial_Vol', 
                                      'Sex', 'Diabetes Status', 'Smoking Status',
                                      'Hypercholesterolemia Status', 'Obesity Status']]
        
        bootstrap_results = generate_bootstrap_stats_by_covariates(
            model=model,
            covariates_df=covariates_df,
            config=config,
            num_samples=args.num_samples,
            num_bootstraps=args.num_bootstraps
        )
        
        logger.info("Inference and bootstrap analysis completed successfully")

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == '__main__':
    main()