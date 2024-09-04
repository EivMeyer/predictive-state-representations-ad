from omegaconf import DictConfig
import matplotlib.pyplot as plt
from utils.dataset_utils import EnvironmentDataset
from pathlib import Path
import numpy as np
from utils.config_utils import config_wrapper


def visualize_episode(episode):
    observations, next_observations = episode['observations'], episode['next_observations']

    # Example to show one image from observations and one from predictions
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Example Episode Images')

    # Show first observation image
    if len(observations) > 0:
        obs_image = np.squeeze(observations[0])  # Remove singleton dimensions
        if obs_image.ndim == 3:  # Check if it's still a valid image shape
            axs[0].imshow(obs_image)
            axs[0].set_title('First Observation')
            axs[0].axis('off')  # Hide axes for images
        else:
            print("Observation image has an unexpected number of dimensions:", obs_image.shape)

    # Show first next observation image
    if len(next_observations) > 0:
        next_obs_image = np.squeeze(next_observations[0])
        if next_obs_image.ndim == 3:
            axs[1].imshow(next_obs_image)
            axs[1].set_title('First Prediction Observation')
            axs[1].axis('off')
        else:
            print("Next observation image has an unexpected number of dimensions:", next_obs_image.shape)

    plt.show()

@config_wrapper()
def main(cfg: DictConfig) -> None:
    dataset_path = Path(cfg.project_dir) / "dataset"
    dataset = EnvironmentDataset(dataset_path)

    # Visualize the first 10 episodes as examples
    if len(dataset) > 0:
        for i in range(10):
            visualize_episode(dataset[i])
    else:
        print("No episodes available in the dataset.")

if __name__ == "__main__":
    main()
