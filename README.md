# Predictive State Representation Learning for Autonomous Vehicles

## Overview

This project develops and evaluates predictive models for autonomous vehicle navigation using deep learning. The goal is to learn ego-centric representations of the future environmental states. It uses [CommonRoad-Geometric](https://github.com/CommonRoad/crgeo) as the autonomous driving software environment.

## Usage

1. Collect dataset:
   ```
   python collect_dataset.py
   ```

2. Train representation model:
   ```
   python train_model.py
   ```

3. Train downstream RL agent:
   ```
   python train_rl_agent.py
   ```

Modify `config.yaml` to adjust simulation parameters and training settings.

## Local Configuration

For machine-specific settings, create a `config.local.yaml` file based on the `config.local.template.yaml`:

```bash
cp config.local.template.yaml config.local.yaml
