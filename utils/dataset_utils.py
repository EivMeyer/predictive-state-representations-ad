import torch
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
from pathlib import Path
import numpy as np
import os
import logging
from utils.transformations import polar_transform
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from omegaconf import OmegaConf


def sample_minibatches(batch, minibatch_size, episode_length_exp=1.0):
    # First identify episode boundaries using dones
    episodes = []
    episode_start = 0
    dones = batch['dones']
    
    for i in range(len(dones)):
        if dones[i].any():  # End of episode
            episodes.append((episode_start, i + 1))
            episode_start = i + 1
    
    # Add final episode if exists
    if episode_start < len(dones):
        episodes.append((episode_start, len(dones)))

    # Calculate sampling weights based on episode lengths
    lengths = np.array([end - start for start, end in episodes])
    weights = lengths ** episode_length_exp
    weights = weights / weights.sum()

    # Create array of selected indices same size as original batch
    total_samples = len(dones)
    selected_indices = []
    
    for i in range(total_samples):
        # Sample an episode based on weights
        episode_idx = np.random.choice(len(episodes), p=weights)
        start, end = episodes[episode_idx]
        # Random sample from the episode
        sample_idx = np.random.randint(start, end)
        selected_indices.append(sample_idx)

    # Return reordered batch with same structure
    return {
        key: value[selected_indices] for key, value in batch.items()
    }

class EnvironmentDataset(Dataset):
    def __init__(self, cfg):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        self.data_dir = Path(cfg.project_dir) / Path(cfg.dataset_dir)
        self.preprocessed_dir = self.data_dir / "preprocessed"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        self.downsample_factor = cfg.training.downsample_factor
        self.storage_batch_size = cfg.dataset.storage_batch_size
        self.use_polar_transform = cfg.dataset.use_polar_transform
        self.current_batch = []
        self.episode_files = []
        
        if cfg.dataset.preprocess_existing and not cfg.dataset.preprocess_online:
            self.preprocess_dataset()
        
        self.update_file_list()

        self.episode_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith("batch_") and f.endswith(".pt")])
        self.batch_count = len(self.episode_files)
        
        if self.batch_count == 1:
            print("Loading single dataset file into memory...")
            self.data = self._load_and_process_file(self.episode_files[0])
            self.num_samples = self.data['observations'].shape[0]
            print(f"Dataset loaded. Total samples: {self.num_samples}")
        else:
            self.data = None
            self.num_samples = None


    def update_file_list(self):
        if self.cfg.dataset.preprocess_existing and not self.cfg.dataset.preprocess_online:
            self.episode_files = sorted([f for f in os.listdir(self.preprocessed_dir) if f.startswith("batch_") and f.endswith(".pt")])
            self.use_preprocessed = True
        else:
            self.episode_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith("batch_") and f.endswith(".pt")])
            self.use_preprocessed = False
        self.batch_count = len(self.episode_files)

    def preprocess_dataset(self):
        print("Preprocessing existing dataset...")
        preprocessed_dir = self.data_dir / "preprocessed"
        preprocessed_dir.mkdir(exist_ok=True)
        
        files_to_process = [f for f in self.data_dir.glob("batch_*.pt") if not (preprocessed_dir / f.name).exists()]
        
        if not files_to_process:
            print("No files to preprocess or all files are already preprocessed.")
            return
        
        num_workers = self.cfg.dataset.preprocess_workers if self.cfg.dataset.preprocess_workers > 0 else cpu_count()
        
        if num_workers > 1:
            with Pool(num_workers) as pool:
                list(tqdm(pool.imap(self.preprocess_file, files_to_process), total=len(files_to_process)))
        else:
            for file in tqdm(files_to_process):
                self.preprocess_file(file)
        
        print("Preprocessing complete.")

    def preprocess_file(self, file_path):
        try:
            data = torch.load(file_path)
            
            # Preprocess observations and next_observations
            for key in ['observations', 'next_observations']:
                data[key] = torch.stack([self.preprocess_image(img) for img in data[key]])
            
            # Save preprocessed data
            preprocessed_path = self.data_dir / "preprocessed" / file_path.name
            torch.save(data, preprocessed_path)
        
        except Exception as e:
            print(f"Error preprocessing file {file_path}: {str(e)}")

    def preprocess_image(self, image, normalize=True):
        # Ensure image is a numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # If the input is a single image, add a batch dimension
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        # Downsample if necessary
        if self.downsample_factor > 1:
            import cv2
            new_h, new_w = image.shape[2] // self.downsample_factor, image.shape[3] // self.downsample_factor
            image = np.stack([cv2.resize(img.transpose(1, 2, 0), (new_w, new_h), 
                                         interpolation=cv2.INTER_AREA).transpose(2, 0, 1) 
                              for img in image])

        # Convert to float32 and normalize to [0, 1] if necessary
        if normalize and image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Apply polar transform if enabled
        if self.use_polar_transform:
            image = np.stack([polar_transform(img) for img in image])

        return torch.from_numpy(image)

    def __getitem__(self, idx):
        if len(self.episode_files) == 0:
            raise RuntimeError(f"No valid data files found in the dataset directory: {self.data_dir}")

        if idx < 0 or idx >= self.batch_count:
            raise IndexError("Batch index out of range")

        if self.data is not None:
            return {key: value[idx] for key, value in self.data.items()}
        else:
            file = self.episode_files[idx]
            try:
                data = self._load_and_process_file(file)
                return self._sample_minibatches(data) if self.cfg.dataset.sample_by_episode_length else self._shuffle_batch(data)
            except Exception as e:
                logging.error(f"Error loading file {file}: {str(e)}. Removing it from the dataset.")
                self._remove_corrupted_batch(idx)
                
                # Recursively try the next file
                idx = min(idx, self.batch_count - 1)
                if idx < self.batch_count:
                    return self.__getitem__(idx)
                else:
                    raise RuntimeError("No valid files found after the specified index.")

    def _sample_minibatches(self, data):
        return sample_minibatches(
            data,
            minibatch_size=self.cfg.training.minibatch_size,
            episode_length_exp=1.0  # Configurable
        )

    def _shuffle_batch(self, data):
        batch_size = data['observations'].shape[0]
        shuffle_idx = torch.randperm(batch_size)
        
        return {
            key: value[shuffle_idx] for key, value in data.items()
        }

    def _load_and_process_file(self, file):
        if self.use_preprocessed:
            file_path = self.preprocessed_dir / file
        else:
            file_path = self.data_dir / file
        data = torch.load(file_path, map_location='cpu')
        if not self.use_preprocessed and self.cfg.dataset.preprocess_online:
            # Apply preprocessing on-the-fly
            for key in ['observations', 'next_observations']:
                if isinstance(data[key], np.ndarray):
                    data[key] = torch.stack([self.preprocess_image(img) for img in data[key]])
                elif isinstance(data[key], torch.Tensor):
                    if data[key].dtype == torch.uint8:
                        data[key] = torch.stack([self.preprocess_image(img.numpy()) for img in data[key]])
        else:
            # Convert to tensors if necessary
            for key in data:
                if isinstance(data[key], np.ndarray):
                    data[key] = torch.from_numpy(data[key])
                elif isinstance(data[key], torch.Tensor):
                    if data[key].dtype == torch.uint8:
                        data[key] = data[key].to(torch.float32) / 255.0
        # Remove 3rd dimension if present and has length 1
        for key in ['observations', 'next_observations']:
            if data[key].shape[2] == 1:
                data[key] = data[key].squeeze(2)
        # Permute to (seq_len, batch_size, channels, height, width) if necessary
        if data['observations'].ndim > 3 and data['observations'].shape[-1] != data['observations'].shape[-2]:
            for key in ['observations', 'next_observations']:
                if isinstance(data[key], np.ndarray):
                    data[key] = np.transpose(data[key], (0, 1, 4, 2, 3))
                else:
                    data[key] = data[key].permute(0, 1, 4, 2, 3).contiguous()
        return data

    def _remove_corrupted_batch(self, idx):        
        del self.episode_files[idx]
        self.batch_count -= 1

    def add_episode(self, observations, actions, ego_states, next_observations, next_actions, dones):
        # Preprocess images as uint8 for storage
        observations = np.stack([self.preprocess_image(obs, normalize=False) for obs in observations])
        next_observations = np.stack([self.preprocess_image(obs, normalize=False) for obs in next_observations])

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
            if key in ['observations', 'next_observations'] and batch_data[key][0].dtype == torch.uint8:
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
        # Check for required keys
        assert all(key in batch_data for key in ['observations', 'actions', 'ego_states', 'next_observations', 'next_actions', 'dones']), "Missing keys in batch data"
        
        # Check for consistent batch sizes
        assert all(batch_data[key].shape[0] == batch_data['observations'].shape[0] for key in batch_data), "Inconsistent batch sizes"
        
        # Check for correct batch size
        assert batch_data['observations'].shape[0] == self.storage_batch_size, f"Incorrect batch size: {batch_data['observations'].shape[0]} vs {self.storage_batch_size}"

    def __len__(self):
        return self.num_samples if self.num_samples is not None else self.batch_count

    @staticmethod
    def collate_fn(batch):
        # Each item in the batch is already shuffled, so we just need to concatenate
        return {
            key: torch.cat([subbatch[key] for subbatch in batch], dim=0)
            for key in batch[0].keys()
        }
    
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

def create_data_loaders(dataset, batch_size, val_size, prefetch_factor, num_workers, pin_memory, batches_per_epoch=None, subset_size=None):
    if subset_size is not None:
        if subset_size > len(dataset):
            raise ValueError(f"Subset size ({subset_size}) cannot be greater than the dataset size ({len(dataset)})")
        dataset = torch.utils.data.Subset(dataset, range(subset_size))
        
    dataset_size = len(dataset)
    if isinstance(val_size, float):
        val_size = int(val_size * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Length of training dataset: {len(train_dataset)}")
    
    if batches_per_epoch is not None:
        train_sampler = SubsetRandomSampler(range(len(train_dataset)), num_samples=batches_per_epoch * batch_size)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=batches_per_epoch is None,
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

def tensor_memory_usage(tensor):
    """
    Calculates and prints the memory usage of a given PyTorch tensor, 
    whether on GPU or CPU.

    Args:
        tensor (torch.Tensor): The tensor to calculate memory usage for.

    Returns:
        dict: A dictionary containing memory usage in bytes, KB, MB, and GB.
    """
    # Calculate memory usage in bytes
    memory_bytes = tensor.element_size() * tensor.nelement()
    
    # Convert to human-readable formats
    memory_kb = memory_bytes / 1024
    memory_mb = memory_kb / 1024
    memory_gb = memory_mb / 1024

    # Format results
    memory_info = {
        "bytes": memory_bytes,
        "KB": memory_kb,
        "MB": memory_mb,
        "GB": memory_gb,
    }

    # Identify if tensor is on GPU or CPU
    device = "GPU" if tensor.is_cuda else "RAM"
    
    print(f"Tensor memory usage ({device}):")
    for key, value in memory_info.items():
        print(f"{key}: {value:.2f}")

    return memory_info