from omegaconf import DictConfig
import matplotlib.pyplot as plt
from utils.dataset_utils import EnvironmentDataset
from pathlib import Path
import numpy as np
from utils.config_utils import config_wrapper

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

def visualize_episode(episode: Dict[str, Any]) -> None:
    observations, next_observations, dones = episode['observations'], episode['next_observations'], episode['dones']
    
    # Select first batch if batched
    if observations.ndim == 5:
        observations = observations[0]
        next_observations = next_observations[0]
        dones = dones[0]
    
    num_frames = max(len(observations), len(next_observations))
    
    # Create a figure with two rows: one for observations, one for next_observations
    fig, axs = plt.subplots(2, num_frames, figsize=(4*num_frames, 8))
    fig.suptitle('Full Episode Sequence. Observation shape: {}'.format(observations[0].shape))
    
    for i in range(num_frames):
        # Process and display observation
        obs_image = np.squeeze(observations[i]) if i < len(observations) else np.zeros_like(observations[0])
        if obs_image.shape[0] == 3:
            obs_image = np.transpose(obs_image, (1, 2, 0))
        
        if obs_image.ndim == 3:
            axs[0, i].imshow(obs_image)
            axs[0, i].set_title(f'Obs {i+1}')
            axs[0, i].axis('off')
        else:
            axs[0, i].text(0.5, 0.5, 'Invalid shape', ha='center', va='center')
        
        # Process and display next_observation
        next_obs_image = np.squeeze(next_observations[i]) if i < len(next_observations) else np.zeros_like(next_observations[0])
        if next_obs_image.shape[0] == 3:
            next_obs_image = np.transpose(next_obs_image, (1, 2, 0))
        
        if next_obs_image.ndim == 3:
            axs[1, i].imshow(next_obs_image)
            axs[1, i].set_title(f'Next Obs {i+1} ' + '(X)' if dones[i].item() else '')
            axs[1, i].axis('off')
        else:
            axs[1, i].text(0.5, 0.5, 'Invalid shape', ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

@config_wrapper()
def main(cfg: DictConfig) -> None:
    dataset = EnvironmentDataset(cfg)
    
    # Visualize the first 5 episodes as examples
    if len(dataset) > 0:
        for i in range(min(5, len(dataset))):
            visualize_episode(dataset[i])
    else:
        print("No episodes available in the dataset.")

if __name__ == "__main__":
    main()