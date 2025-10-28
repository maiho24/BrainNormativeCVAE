import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    def __init__(self, data, labels, indices=False, transform=None):
        self.data = data
        self.labels = labels
        if isinstance(data, list) or isinstance(data, tuple):
            self.data = [torch.from_numpy(d).float() if isinstance(d, np.ndarray) else d for d in self.data]
            self.N = len(self.data[0])
            self.shape = np.shape(self.data[0])
        elif isinstance(data, np.ndarray):
            self.data = torch.from_numpy(self.data).float()
            self.N = len(self.data)
            self.shape = np.shape(self.data)
            
        self.labels = torch.from_numpy(self.labels).float()
        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if isinstance(self.data, list):
            x = [d[index] for d in self.data]
        else:
            x = self.data[index]

        if self.transform:
            x = self.transform(x)
        t = self.labels[index]
        if self.indices:
            return x, t, index
        return x, t

    def __len__(self):
        return self.N
        
        
def process_covariates(covariates_df):
    """Process covariates into the format needed by the model."""
    try:
        # Continuous variables
        age_icv = covariates_df[['Age', 'eTIV', 'Sex', 'Diabetes Status', 'Smoking Status', 
                                 'Hypercholesterolemia Status', 'Obesity Status']].values
        
        # Categorical variables
        categorical_cols = []
        one_hot_encodings = []
        
        for col in categorical_cols:
            if col not in covariates_df.columns:
                raise KeyError(f"Column '{col}' not found in covariates DataFrame")
            one_hot = pd.get_dummies(covariates_df[col], prefix=col).values
            one_hot_encodings.append(one_hot)
        
        # Combine all covariates
        return np.hstack([age_icv] + one_hot_encodings)
    except KeyError as e:
        raise KeyError(f"Missing required column: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing covariates: {str(e)}")
    
def load_train_data(data_path, val_size, logger, calibration_enabled=True, calibration_split=0.3):
    """
    Load and process training and validation data with automatic calibration data handling.
    
    This function creates a proper three-way split:
    1. Training data (for model training)
    2. Validation data (for early stopping during hyperparameter search)
    3. Calibration data (completely held out for post-training variance calibration)
    
    Args:
        data_path: Path to data directory
        val_size: Proportion of data to reserve for validation + calibration
        logger: Logger instance
        calibration_enabled: Whether calibration is enabled
        calibration_split: Proportion of validation data to use for calibration
        
    Returns:
        Tuple of (train_data, train_cov, val_data, val_cov, test_data, test_cov, [cal_data, cal_cov])
    """
    logger.info("Loading training data with automatic calibration split...")
    
    # Load raw data
    train_data = pd.read_csv(data_path / 'train_data.csv')
    train_covariates = pd.read_csv(data_path / 'train_covariates.csv')
    test_data = pd.read_csv(data_path / 'test_data.csv')
    test_covariates = pd.read_csv(data_path / 'test_covariates.csv')
    
    # Process covariates
    train_covariates_processed = process_covariates(train_covariates)
    test_covariates_processed = process_covariates(test_covariates)
    
    # Convert data to numpy arrays
    train_data_np = train_data.to_numpy()
    test_data_np = test_data.to_numpy()
    
    # Create indices for splitting
    indices = np.arange(len(train_data_np))
    np.random.seed(42)
    
    if calibration_enabled:
        # Three-way split: train / val / calibration
        # First split: separate training from (validation + calibration)
        train_indices, val_cal_indices = train_test_split(
            indices, test_size=val_size, random_state=42, shuffle=True
        )
        
        # Second split: separate validation from calibration
        val_cal_data = train_data_np[val_cal_indices]
        val_cal_cov = train_covariates_processed[val_cal_indices]
        
        val_indices, cal_indices = train_test_split(
            np.arange(len(val_cal_indices)), 
            test_size=calibration_split, 
            random_state=43,
            shuffle=True
        )
        
        # Map back to original indices
        actual_val_indices = val_cal_indices[val_indices]
        actual_cal_indices = val_cal_indices[cal_indices]
        
        # Create final splits
        train_data_split = train_data_np[train_indices]
        train_cov_split = train_covariates_processed[train_indices]
        val_data_split = train_data_np[actual_val_indices]
        val_cov_split = train_covariates_processed[actual_val_indices]
        cal_data_split = train_data_np[actual_cal_indices]
        cal_cov_split = train_covariates_processed[actual_cal_indices]
        
        logger.info(f"Data split - Training: {len(train_data_split)}, "
                   f"Validation: {len(val_data_split)}, "
                   f"Calibration: {len(cal_data_split)}, "
                   f"Test: {len(test_data_np)}")
        
        return (train_data_split, train_cov_split, 
                val_data_split, val_cov_split,
                test_data_np, test_covariates_processed,
                cal_data_split, cal_cov_split)
    
    else:
        # Two-way split: train / val (no calibration)
        train_indices, val_indices = train_test_split(
            indices, test_size=val_size, random_state=42, shuffle=True
        )
        
        # Create final splits
        train_data_split = train_data_np[train_indices]
        train_cov_split = train_covariates_processed[train_indices]
        val_data_split = train_data_np[val_indices]
        val_cov_split = train_covariates_processed[val_indices]
        
        logger.info(f"Data split - Training: {len(train_data_split)}, "
                   f"Validation: {len(val_data_split)}, "
                   f"Test: {len(test_data_np)}")
        
        return (train_data_split, train_cov_split, 
                val_data_split, val_cov_split,
                test_data_np, test_covariates_processed)

def load_test_data(data_path, logger):
    """Load and process test data."""
    logger.info("Loading test data...")
    
    test_data = pd.read_csv(f"{data_path}/test_data.csv")
    test_covariates = pd.read_csv(f"{data_path}/test_covariates.csv")
    
    # Process covariates
    processed_covariates = process_covariates(test_covariates)
    
    return test_data, test_covariates, processed_covariates