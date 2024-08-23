# utils/dataset_utils.py

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
import os
import cv2

class EnvironmentDataset:
    def __init__(self, data_dir, downsample_factor=1):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.episode_files = []
        self.downsample_factor = downsample_factor
        self.load_existing_episodes()

    def load_existing_episodes(self):
        self.episode_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith("episode_") and f.endswith(".pt")])
        self.episode_count = len(self.episode_files)

    def add_episode(self, observations, actions, ego_states, next_observations, next_actions, dones):
        episode_data = {
            'observations': np.array(observations, dtype=np.uint8),  # Assuming 8-bit color images
            'actions': np.array(actions, dtype=np.float32),
            'ego_states': np.array(ego_states, dtype=np.float32),
            'next_observations': np.array(next_observations, dtype=np.uint8),  # Assuming 8-bit color images
            'next_actions': np.array(next_actions, dtype=np.float32),
            'dones': np.array(dones, dtype=bool)
        }
        episode_filename = f"episode_{self.episode_count}.pt"
        torch.save(episode_data, self.data_dir / episode_filename)
        self.episode_files.append(episode_filename)
        self.episode_count += 1
        return episode_filename

    def __len__(self):
        return self.episode_count
    
    def downsample_image(self, image):
        if self.downsample_factor > 1:
            return cv2.resize(image, (image.shape[1] // self.downsample_factor, 
                                      image.shape[0] // self.downsample_factor), 
                              interpolation=cv2.INTER_AREA)
        return image

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.episode_count:
            raise IndexError("Episode index out of range")
        episode_path = self.data_dir / self.episode_files[idx]
        data = torch.load(episode_path)
        
        # Apply downsampling to observations and next_observations
        data['observations'] = np.array([self.downsample_image(obs) for obs in data['observations']])
        data['next_observations'] = np.array([self.downsample_image(obs) for obs in data['next_observations']])
        
        # Convert to PyTorch tensors and adjust format
        # From [time, height, width, channels] to [time, channels, height, width]
        data['observations'] = torch.from_numpy(data['observations']).float().permute(0, 3, 1, 2).contiguous() / 255.0
        data['next_observations'] = torch.from_numpy(data['next_observations']).float().permute(0, 3, 1, 2).contiguous() / 255.0

        # Convert other data to tensors
        data['actions'] = torch.from_numpy(data['actions']).float()
        data['ego_states'] = torch.from_numpy(data['ego_states']).float()
        data['next_actions'] = torch.from_numpy(data['next_actions']).float()
        data['dones'] = torch.from_numpy(data['dones']).float()
        
        return data

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4, device='cpu'):
        from torch.utils.data import DataLoader

        def collate_fn(batch):
            return {
                'observations': torch.stack([torch.from_numpy(item['observations']) for item in batch]).to(device),
                'actions': torch.stack([torch.from_numpy(item['actions']) for item in batch]).to(device),
                'ego_states': torch.stack([torch.from_numpy(item['ego_states']) for item in batch]).to(device),
                'next_observations': torch.stack([torch.from_numpy(item['next_observations']) for item in batch]).to(device),
                'next_actions': torch.stack([torch.from_numpy(item['next_actions']) for item in batch]).to(device),
                'dones': torch.stack([torch.from_numpy(item['dones']) for item in batch]).to(device)
            }

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    

def collate_fn(batch, device='cpu'):
    return {
        'observations': torch.from_numpy(np.stack([item['observations'] for item in batch])).to(device),
        'actions': torch.from_numpy(np.stack([item['actions'] for item in batch])).to(device),
        'ego_states': torch.from_numpy(np.stack([item['ego_states'] for item in batch])).to(device),
        'next_observations': torch.from_numpy(np.stack([item['next_observations'] for item in batch])).to(device),
        'next_actions': torch.from_numpy(np.stack([item['next_actions'] for item in batch])).to(device),
        'dones': torch.from_numpy(np.stack([item['dones'] for item in batch])).to(device)
    }


def get_dataloader(dataset_dir, batch_size=32, shuffle=True, num_workers=4, device='cpu'):
    dataset = EnvironmentDataset(dataset_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn(batch, device)
    )


def create_data_loaders(dataset, batch_size, device, train_ratio=0.8):
    """
    Split the dataset into training and validation sets, then create DataLoaders.
    
    Args:
    dataset (Dataset): The full dataset
    batch_size (int): Batch size for the DataLoaders
    train_ratio (float): Ratio of data to use for training (default: 0.8)

    Returns:
    train_loader (DataLoader): DataLoader for the training set
    val_loader (DataLoader): DataLoader for the validation set
    """
    # Calculate the size of each split
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders with custom collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda b: collate_fn(b, device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, device))

    return train_loader, val_loader



def get_data_dimensions(dataset):
    """
    Extracts the dimensions of observations, actions, and ego states from the dataset.
    
    Args:
    dataset (EnvironmentDataset): The dataset to examine

    Returns:
    obs_shape (int): Shape of a single observation
    action_dim (int): Dimension of the action space
    ego_state_dim (int): Dimension of the ego state
    """
    # Get the first item from the dataset
    first_item = dataset[0]
    
    # Unpack the first item
    observations, actions, ego_states = first_item['observations'], first_item['actions'], first_item['ego_states']
    
    # Get dimensions
    obs_shape = observations[0].shape  # Shape of a single observation
    action_dim = len(actions[0])  # Dimension of a single action
    ego_state_dim = len(ego_states[0])  # Dimension of a single ego state
    
    print(f"Observation shape: {obs_shape}")
    print(f"Action dimension: {action_dim}")
    print(f"Ego state dimension: {ego_state_dim}")
    
    return obs_shape, action_dim, ego_state_dim