from omegaconf import DictConfig
import matplotlib.pyplot as plt
from utils.dataset_utils import EnvironmentDataset
from pathlib import Path
import numpy as np
from utils.config_utils import config_wrapper
from typing import Dict, Any

def visualize_episode(episode: Dict[str, Any]) -> None:
    observations = episode['observations']
    next_observations = episode['next_observations']

    # Select first batch if batched
    if observations.ndim == 5:
        observations = observations[0]

    num_frames = len(observations)
    
    # Determine the number of segments
    if isinstance(next_observations, dict):
        num_segments = len(next_observations)
        segment_names = list(next_observations.keys())
    else:
        num_segments = 1
        segment_names = ['next_observations']
        next_observations = {'next_observations': next_observations}

    # Create a figure with rows for observations and each segment
    fig, axs = plt.subplots(1 + num_segments, num_frames, figsize=(4*num_frames, 4*(1 + num_segments)))
    fig.suptitle('Full Episode Sequence')

    def process_and_display_image(ax, image, title):
        image = np.squeeze(image)
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        
        if image.ndim == 3:
            ax.imshow(image)
            ax.set_title(title)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Invalid shape', ha='center', va='center')

    # Display observations
    for i in range(num_frames):
        process_and_display_image(axs[0, i], observations[i], f'Obs {i+1}')

    # Display next_observations for each segment
    for seg_idx, (seg_name, seg_data) in enumerate(next_observations.items(), start=1):
        if seg_data.ndim == 5:
            seg_data = seg_data[0]  # Select first batch if batched
        for i in range(num_frames):
            process_and_display_image(axs[seg_idx, i], seg_data[i], f'{seg_name} {i+1}')

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