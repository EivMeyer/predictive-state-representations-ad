import torch
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
from pathlib import Path
import numpy as np
import os
import cv2
import logging

class EnvironmentDataset(Dataset):
    def __init__(self, data_dir, downsample_factor=1, storage_batch_size=32):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.downsample_factor = downsample_factor
        self.storage_batch_size = storage_batch_size
        self.current_batch = []
        self.episode_files = []
        self.update_file_list()

    def update_file_list(self):
        self.episode_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith("batch_") and f.endswith(".pt")])
        self.batch_count = len(self.episode_files)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.batch_count:
            raise IndexError("Batch index out of range")
        
        file = self.episode_files[idx]
        try:
            data = self._load_and_process_file(file)
            return data
        except Exception as e:
            logging.error(f"Error loading file {file}: {str(e)}. Removing it from the dataset.")
            self._remove_corrupted_file(idx)
            
            # Update the file list and batch count
            self.update_file_list()
            
            # Recursively try the next file
            if idx < self.batch_count:
                return self.__getitem__(idx)
            else:
                raise RuntimeError("No valid files found after the specified index.")

    def _load_and_process_file(self, file):
        file_path = self.data_dir / file
        data = torch.load(file_path, map_location='cpu')

        # Convert images from uint8 to float32 and normalize if they are numpy arrays
        for key in ['observations', 'next_observations']:
            if isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key]).float().div(255.0)
            elif isinstance(data[key], torch.Tensor):
                if data[key].dtype == torch.uint8:
                    data[key] = data[key].float().div(255.0)

        return data

    def _remove_corrupted_file(self, idx):
        file_to_remove = self.episode_files[idx]
        file_path = self.data_dir / file_to_remove
        try:
            os.remove(file_path)
        except OSError as e:
            logging.error(f"Error removing file {file_to_remove}: {str(e)}")
        
        del self.episode_files[idx]
        self.batch_count -= 1

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

        if len(self.current_batch) >= self.storage_batch_size:
            self._save_current_batch()

    def _preprocess_image(self, image):
        # Ensure image is a numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Downsample if necessary
        if self.downsample_factor > 1:
            image = cv2.resize(image, (image.shape[1] // self.downsample_factor,
                                       image.shape[0] // self.downsample_factor),
                               interpolation=cv2.INTER_AREA)

        # Check if the image needs to be transposed
        if len(image.shape) == 3 and image.shape[2] in [1, 3, 4]:  # HWC format
            image = image.transpose(2, 0, 1)  # Convert to CHW format
        elif len(image.shape) == 2:  # Grayscale image
            image = np.expand_dims(image, axis=0)  # Add channel dimension

        # Ensure the image is uint8
        image = np.asarray(image, dtype=np.uint8)

        return image

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

        # Perform sanity checks
        try:
            self._sanity_check(batch_data)
        except AssertionError as e:
            logging.error(f"Sanity check failed: {str(e)}. Discarding this batch.")
            self.current_batch = []
            return

        batch_index = self.batch_count
        batch_filename = f"batch_{batch_index}.pt"
        temp_filename = f"temp_{batch_filename}"

        # Save to a temporary file first
        torch.save(batch_data, self.data_dir / temp_filename)

        # Verify the saved file
        try:
            loaded_data = torch.load(self.data_dir / temp_filename)
            self._sanity_check(loaded_data)
        except Exception as e:
            logging.error(f"Failed to verify saved batch: {str(e)}. Discarding this batch.")
            os.remove(self.data_dir / temp_filename)
            self.current_batch = []
            return

        # If verification passes, rename the file
        os.rename(self.data_dir / temp_filename, self.data_dir / batch_filename)
        
        self.episode_files.append(batch_filename)
        self.batch_count += 1
        self.current_batch = []

    def _sanity_check(self, batch_data):
        assert all(key in batch_data for key in ['observations', 'actions', 'ego_states', 'next_observations', 'next_actions', 'dones']), "Missing keys in batch data"
        assert all(len(batch_data[key]) == len(batch_data['observations']) for key in batch_data), "Inconsistent batch sizes"
        assert batch_data['observations'].shape[0] == self.storage_batch_size, f"Incorrect batch size: {batch_data['observations'].shape[0]} vs {self.storage_batch_size}"

    def __len__(self):
        return self.batch_count

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
    

class SubsetRandomSampler(Sampler):
    def __init__(self, indices, num_samples=None):
        self.indices = indices
        self._num_samples = num_samples

    @property
    def num_samples(self):
        return len(self.indices) if self._num_samples is None else self._num_samples

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices))[:self.num_samples])

    def __len__(self):
        return self.num_samples



def create_data_loaders(dataset, batch_size, val_size, prefetch_factor, num_workers, pin_memory, batches_per_epoch=None):
    dataset_size = len(dataset)
    if isinstance(val_size, float):
        val_size = int(val_size * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if batches_per_epoch is not None:
        train_sampler = SubsetRandomSampler(range(len(train_dataset)), num_samples=batches_per_epoch)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=EnvironmentDataset.collate_fn,
        prefetch_factor=prefetch_factor,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=EnvironmentDataset.collate_fn,
        prefetch_factor=prefetch_factor,
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