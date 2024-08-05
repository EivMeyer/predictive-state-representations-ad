# utils/dataset_utils.py

import torch
from torch.utils.data import Dataset
from experiment_setup import load_config
from pathlib import Path
import numpy as np

class EnvironmentDataset(Dataset):
    def __init__(self, data=None):
        # Initialize with data if provided, else start with an empty list
        if data is not None:
            self.data = data
        else:
            self.data = []

    def add_episode(self, observations, actions, ego_states, next_observations, next_actions, dones):
        # Convert lists to numpy arrays before storing
        self.data.append((
            np.array(observations, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(ego_states, dtype=np.float32),
            np.array(next_observations, dtype=np.float32),
            np.array(next_actions, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @classmethod
    def load_from_file(cls, file_path):
        # Load the dataset from a file and pass it to the constructor
        data = torch.load(file_path)
        return cls(data)


def get_data_dimensions(dataset):
    """
    Extracts the dimensions of observations, actions, and ego states from the dataset.
    
    Args:
    dataset (EnvironmentDataset): The dataset to examine

    Returns:
    obs_dim (int): Total dimension of a single observation
    action_dim (int): Dimension of the action space
    ego_state_dim (int): Dimension of the ego state
    """
    # Get the first item from the dataset
    first_item = dataset[0]
    
    # Unpack the first item
    observations, actions, ego_states, _, _, _ = first_item
    
    # Get dimensions
    obs_shape = observations[0].shape  # Shape of a single observation
    obs_dim = np.prod(obs_shape)  # Total dimension of observation
    action_dim = len(actions[0])  # Dimension of a single action
    ego_state_dim = len(ego_states[0])  # Dimension of a single ego state
    
    print(f"Observation shape: {obs_shape}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Ego state dimension: {ego_state_dim}")
    
    return obs_dim, action_dim, ego_state_dim