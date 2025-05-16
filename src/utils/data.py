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
        age_icv = covariates_df[['Age', 'wb_EstimatedTotalIntraCranial_Vol']].values
        
        # Categorical variables
        categorical_cols = ['Sex', 'Diabetes Status', 'Smoking Status', 
                          'Hypercholesterolemia Status', 'Obesity Status']
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
    
def load_train_data(data_path, val_size, logger):
    """Load and process training and validation data."""
    logger.info("Loading training data...")
    
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
    
    # Split training data into train and validation sets
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

def load_test_data(data_path, logger):
    """Load and process test data."""
    logger.info("Loading test data...")
    
    test_data = pd.read_csv(f"{data_path}/test_data.csv")
    test_covariates = pd.read_csv(f"{data_path}/test_covariates.csv")
    
    # Process covariates
    processed_covariates = process_covariates(test_covariates)
    
    return test_data, test_covariates, processed_covariates