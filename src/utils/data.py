import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class MyDataset_labels(Dataset):
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
            
        self.labels = torch.from_numpy(self.labels).long()
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
    # Continuous variables
    age_icv = covariates_df[['Age', 'wb_EstimatedTotalIntraCranial_Vol']].values
    
    # Categorical variables
    categorical_cols = ['Sex', 'Diabetes Status', 'Smoking Status', 
                       'Hypercholesterolemia Status', 'Obesity Status']
    one_hot_encodings = []
    
    for col in categorical_cols:
        one_hot = pd.get_dummies(covariates_df[col], prefix=col).values
        one_hot_encodings.append(one_hot)
    
    # Combine all covariates
    return np.hstack([age_icv] + one_hot_encodings)

def load_data(data_path, val_split=0.15):
    """Load and process training and validation data."""
    # Load raw data
    train_data = pd.read_csv(f"{data_path}/train_data_subset.csv")
    train_covariates = pd.read_csv(f"{data_path}/train_covariates_subset.csv")
    
    # Process covariates
    processed_covariates = process_covariates(train_covariates)
    
    # Split into train and validation
    train_data_np, val_data_np, train_cov_np, val_cov_np = train_test_split(
        train_data.to_numpy(), 
        processed_covariates,
        test_size=val_split,
        random_state=42
    )
    
    return train_data_np, train_cov_np, val_data_np, val_cov_np

def load_test_data(data_path):
    """Load and process test data."""
    test_data = pd.read_csv(f"{data_path}/test_data_subset.csv")
    test_covariates = pd.read_csv(f"{data_path}/test_covariates_subset.csv")
    
    # Process covariates
    processed_covariates = process_covariates(test_covariates)
    
    return test_data.to_numpy(), processed_covariates