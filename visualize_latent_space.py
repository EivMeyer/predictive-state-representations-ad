import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from models import get_model_class
from utils.dataset_utils import EnvironmentDataset, get_data_dimensions, move_batch_to_device
from utils.config_utils import load_and_merge_config
from utils.file_utils import find_model_path
from pathlib import Path
import argparse
import time
import os

def visualize_latent_space(model, dataset, device, num_samples=1000, perplexity=30, n_iter=1000):
    print(f"Starting latent space visualization with {num_samples} samples...")
    model.eval()
    latent_vectors = []
    labels = []

    print("Encoding samples...")
    start_time = time.time()
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            if i % 100 == 0:
                print(f"Processed {i}/{min(num_samples, len(dataset))} samples...")
            batch = dataset[i]
            batch = move_batch_to_device(batch, device)
            
            latent = model.encode(batch)
            if isinstance(latent, tuple):
                latent = latent[0]
            
            # Handle batch dimension
            latent = latent.cpu().numpy()
            if latent.ndim == 2:
                latent_vectors.extend(latent)
                labels.extend(batch['ego_states'][:, 0, 0].cpu().numpy())
            else:
                latent_vectors.append(latent)
                labels.append(batch['ego_states'][0, 0, 0].item())

    latent_vectors = np.array(latent_vectors)
    labels = np.array(labels)
    encoding_time = time.time() - start_time
    print(f"Encoding completed in {encoding_time:.2f} seconds.")

    print(f"Latent vectors shape: {latent_vectors.shape}, Labels shape: {labels.shape}")

    if len(latent_vectors) != len(labels):
        raise ValueError(f"Mismatch between number of latent vectors ({len(latent_vectors)}) and labels ({len(labels)})")

    print(f"Performing t-SNE with perplexity={perplexity} and n_iter={n_iter}...")
    start_time = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42, verbose=1)
    latent_2d = tsne.fit_transform(latent_vectors)
    tsne_time = time.time() - start_time
    print(f"t-SNE completed in {tsne_time:.2f} seconds.")

    print("Generating visualization...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Ego Vehicle Velocity')
    plt.title('t-SNE Visualization of Latent Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/latent_space_visualization.png')
    plt.close()

    print(f"Latent space visualization saved as 'output/latent_space_visualization.png'")


def main():
    parser = argparse.ArgumentParser(description="Visualize the latent space of a model")
    parser.add_argument('--model-type', type=str, required=True, help='Class of trained model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to use for visualization')
    args = parser.parse_args()

    print("Loading configuration...")
    cfg = load_and_merge_config()
    
    print("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else cfg.device)
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset_path = Path(cfg.project_dir) / "dataset"
    full_dataset = EnvironmentDataset(dataset_path, downsample_factor=cfg.training.downsample_factor)
    print(f"Dataset loaded with {len(full_dataset)} samples.")

    print("Getting data dimensions...")
    obs_shape, action_dim, ego_state_dim = get_data_dimensions(full_dataset)
    print(f"Observation shape: {obs_shape}, Action dim: {action_dim}, Ego state dim: {ego_state_dim}")

    print(f"Initializing model of type {args.model_type}...")
    ModelClass = get_model_class(args.model_type)
    if ModelClass is None:
        raise ValueError(f"Invalid model type: {args.model_type}")
    
    model = ModelClass(obs_shape=obs_shape, action_dim=action_dim, ego_state_dim=ego_state_dim, cfg=cfg).to(device)
    print("Model initialized.")

    print(f"Loading model weights from {args.model_path}...")
    model_path = find_model_path(cfg.project_dir, args.model_path)
    if model_path is None:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'], strict=False)
    print("Model weights loaded successfully.")

    visualize_latent_space(model, full_dataset, device, num_samples=args.num_samples)

if __name__ == "__main__":
    main()