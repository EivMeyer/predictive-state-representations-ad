import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
import os
import cv2
import logging

class EnvironmentDataset(Dataset):
    def __init__(self, data_dir, downsample_factor=1, batch_size=32):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.downsample_factor = downsample_factor
        self.batch_size = batch_size
        self.current_batch = []
        self.episode_files = []
        self.load_existing_batches()

    def load_existing_batches(self):
        self.episode_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith("batch_") and f.endswith(".pt")])
        self.batch_count = len(self.episode_files)

    def add_episode(self, observations, actions, ego_states, next_observations, next_actions, dones):
        # Preprocess images as uint8 for storage
        observations = np.stack([self._preprocess_image(obs) for obs in observations])
        next_observations = np.stack([self._preprocess_image(obs) for obs in next_observations])

        # Convert data to tensors at this stage
        episode_data = {
            'observations': torch.from_numpy(observations).to(torch.uint8),
            'actions': torch.from_numpy(np.array(actions)).to(torch.float32),
            'ego_states': torch.from_numpy(np.array(ego_states)).to(torch.float32),
            'next_observations': torch.from_numpy(next_observations).to(torch.uint8),
            'next_actions': torch.from_numpy(np.array(next_actions)).to(torch.float32),
            'dones': torch.from_numpy(np.array(dones)).to(torch.bool)
        }
        self.current_batch.append(episode_data)

        if len(self.current_batch) >= self.batch_size:
            self._save_current_batch()

    def _preprocess_image(self, image):
        if self.downsample_factor > 1:
            image = cv2.resize(image, (image.shape[1] // self.downsample_factor, 
                                       image.shape[0] // self.downsample_factor), 
                               interpolation=cv2.INTER_AREA)
        return image.transpose(2, 0, 1)  # [C, H, W] format

    def _save_current_batch(self):
        batch_data = {
            'observations': [],
            'actions': [],
            'ego_states': [],
            'next_observations': [],
            'next_actions': [],
            'dones': []
        }

        for episode in self.current_batch:
            batch_data['observations'].append(episode['observations'])
            batch_data['actions'].append(episode['actions'])
            batch_data['ego_states'].append(episode['ego_states'])
            batch_data['next_observations'].append(episode['next_observations'])
            batch_data['next_actions'].append(episode['next_actions'])
            batch_data['dones'].append(episode['dones'])

        # Stack the data along the batch dimension
        for key in batch_data:
            if key in ['observations', 'next_observations']:
                batch_data[key] = np.stack(batch_data[key])  # uint8 storage
            else:
                batch_data[key] = torch.stack(batch_data[key])

        batch_index = len(self.episode_files)
        batch_filename = f"batch_{batch_index}.pt"
        torch.save(batch_data, self.data_dir / batch_filename)
        
        self.episode_files.append(batch_filename)
        self.batch_count += 1
        self.current_batch = []

    def __len__(self):
        return self.batch_count

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.batch_count:
            raise IndexError("Batch index out of range")
        
        # Try loading files until a valid one is found or we run out of files
        for i in range(idx, self.batch_count):
            file = self.episode_files[i]
            try:
                data = self._load_and_process_file(file)
                return data
            except Exception as e:
                logging.warning(f"Error loading file {file}: {str(e)}. Skipping to next file.")
        
        # If we've gone through all files without success, raise an exception
        raise RuntimeError("No valid files found after the specified index.")
    
    def _load_and_process_file(self, file):
        file_path = self.data_dir / file
        data = torch.load(file_path)

        # Convert images from uint8 to float32 and normalize
        data['observations'] = torch.tensor(data['observations']).float() / 255.0
        data['next_observations'] = torch.tensor(data['next_observations']).float() / 255.0

        return data

    @staticmethod
    def collate_fn(batch):
        # Determine the total number of samples across all subbatches
        total_samples = sum(subbatch['observations'].size(0) for subbatch in batch)
        
        # Get the shapes of the tensors
        first_subbatch = batch[0]
        shapes = {key: first_subbatch[key].shape[1:] for key in first_subbatch.keys()}
        
        # Pre-allocate tensors for the merged batch on CPU
        merged_batch = {
            key: torch.empty((total_samples, *shapes[key]),
                            dtype=first_subbatch[key].dtype,
                            pin_memory=True)  # Use pinned memory for faster transfers
            for key in first_subbatch.keys()
        }
        
        # Fill the pre-allocated tensors
        start_idx = 0
        for subbatch in batch:
            batch_size = subbatch['observations'].size(0)
            for key in merged_batch.keys():
                merged_batch[key][start_idx:start_idx+batch_size].copy_(subbatch[key])
            start_idx += batch_size
        
        return merged_batch

def create_data_loaders(dataset, batch_size, train_ratio, prefetch_factor, num_workers, pin_memory):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,  # Each item is a pre-batched set of episodes
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=EnvironmentDataset.collate_fn,
        prefetch_factor=prefetch_factor,  # Prefetch 2 batches per worker
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,  # Each item is a pre-batched set of episodes
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=EnvironmentDataset.collate_fn,
        prefetch_factor=prefetch_factor,  # Prefetch 2 batches per worker
    )

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
    # Get the first batch from the dataset
    first_batch = dataset[0]
    
    # Unpack the first item
    observations = first_batch['observations']
    actions = first_batch['actions']
    ego_states = first_batch['ego_states']
    
    # Get dimensions
    obs_shape = observations.shape[1:]  # Shape of a single observation
    action_dim = actions.shape[-1]  # Dimension of the action space
    ego_state_dim = ego_states.shape[-1]  # Dimension of the ego state
    
    return obs_shape, action_dim, ego_state_dim

def move_batch_to_device(batch, device):
    # Move entire batch to GPU at once
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return batch