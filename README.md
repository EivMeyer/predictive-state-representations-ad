# Enhancing State Representation Learning through Constant Action Interventions and Survival Analysis for Autonomous Highway Driving

This project develops and evaluates predictive models for autonomous vehicle navigation using deep learning. The goal is to learn ego-centric representations of the future environmental states. It uses [CommonRoad-Geometric](https://github.com/CommonRoad/crgeo) as the autonomous driving software environment.

## Demo Video

Click the thumbnail above to watch the demonstration video on Google Drive.

[![Demo Video](https://via.placeholder.com/800x450.png?text=Click+to+Play+Demo+Video)](https://drive.google.com/file/d/1ZDKHsqMhnGXWziVpE1_HnlSOTkUTk3sG/view?usp=drive_link)
---

## Table of Contents
1. [Overview](#overview)
2. [Setup](#setup)
   - [Docker](#docker)
   - [Local Configuration](#local-configuration)
3. [Usage](#usage)
   - [Dataset Collection](#dataset-collection)
   - [Training Models](#training-models)
4. [Troubleshooting](#troubleshooting)

---

## Overview

This repository includes tools for:
- Collecting datasets for training and evaluation.
- Training predictive state representation models.
- Training reinforcement learning agents based on these representations.

---

## Setup

### Docker

To simplify environment management, we use Docker with GPU support. Follow the steps below to get started:

1. **Pull the Docker Image**  
   Pull the latest pre-built Docker image:
   ```bash
   docker pull ge32luk/psr-ad:latest
   ```

2. **Run the Docker Container**  
   Use the following command to start a Docker container:
   ```bash
   docker run -it --gpus all   -v $(pwd):/app/psr-ad   -v $(pwd)/output:/app/psr-ad/output   -v $(pwd)/scenarios:/app/psr-ad/scenarios -v $(pwd)/../../data:/app/psr-ad/data   -e CUDA_VISIBLE_DEVICES=0   -e WANDB_API_KEY=...   ge32luk/psr-ad:latest
   ```

   **Explanation of Flags**:
   - `--gpus all`: Enables GPU access.
   - `-v $(pwd):/app/psr-ad`: Mounts the current directory as `/app/psr-ad` inside the container.
   - `-e CUDA_VISIBLE_DEVICES=0`: Limits GPU usage to a specific device.
   - `-e WANDB_API_KEY=<your_key>`: Sets the API key for Weights & Biases logging.

3. **Build the Docker Image Locally**  
   If you need to build the image locally:
   ```bash
   docker-compose build
   ```

---

## Local Configuration

Create a `config.local.yaml` file to specify machine-specific settings. Use the provided template:
```bash
cp config.local.template.yaml config.local.yaml
```

Edit the file to match your local setup (e.g., paths, hyperparameters).

---

## Usage

### Dataset Collection

To collect a dataset, ensure your environment is set up (either locally or via Docker). Use the following command:

```bash
python collect_dataset.py
```

Alternatively, for distributed workload using multiple workers:
```bash
./parallel_dataset_collection.sh -e 60 -w 2 -r 60 commonroad.scenario_dir="data"
```

#### **Important Note**
For headless environments (e.g., servers without a display), set the following environment variable:
```bash
export PYGLET_HEADLESS=1
```

### Train Representation Model

Train the predictive state representation model using:
```bash
python train_model.py
```

### Train RL Agent

Once the representation model is trained, train a reinforcement learning agent:
```bash
python train_rl_agent.py
```

---

## Troubleshooting

### Rendering Conflicts

- If you encounter issues related to rendering during dataset collection, ensure the following:
  - Set `export PYGLET_HEADLESS=1` for headless environments.
  - Verify all required directories (e.g., `output`, `scenarios`, `data`) exist.

### Performance Bottlenecks

- For faster dataset collection, use parallel workers:
  ```bash
  ./parallel_dataset_collection.sh -e <num_episodes> -w <num_workers> -r <retries>
  ```

### Common Errors

- **Missing Directories**: Verify paths in `config.local.yaml` or the Docker volumes.
- **Connection Issues with WandB**: Ensure your API key is set correctly using the `WANDB_API_KEY` environment variable.

---

## Contributing

We welcome contributions! If you encounter issues or have feature requests, feel free to open an issue or a pull request.

---
