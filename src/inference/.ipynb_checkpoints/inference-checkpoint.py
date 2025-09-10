import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def get_reconstruction_vars(model, test_data, test_covariates, device):
    """
    Get reconstruction variances for test data.
    
    Args:
        model: Trained cVAE model
        test_data: Test data tensor
        test_covariates: Test covariates tensor
        device: Torch device
    
    Returns:
        numpy array: Reconstruction variances
    """
    model.eval()
    with torch.no_grad():
        _, test_recon_var = model.pred_recon(test_data, test_covariates, device)
    return test_recon_var

def run_inference(model, test_data, test_covariates, config):
    """Simple inference to get reconstruction variances."""
    device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    try:
        test_data = pd.DataFrame(test_data)
        
        test_recon_var = get_reconstruction_vars(
            model, 
            test_data, 
            test_covariates, 
            device
        )
        
        results_dir = Path(config['paths']['output_dir']) / 'results'
        np.savetxt(
            results_dir / 'test_reconstruction_vars.csv', 
            test_recon_var, 
            delimiter=','
        )
        
        logger.info("Reconstruction variances saved successfully")
        return test_recon_var
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise