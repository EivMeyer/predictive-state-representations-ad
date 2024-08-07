# Predictive State Representation Learning for Autonomous Vehicles

## Overview

This project develops and evaluates predictive models for autonomous vehicle navigation using deep learning. The goal is to learn ego-centric representations of the future environmental states.

## Usage

1. Collect dataset:
   ```
   python collect_dataset.py
   ```

2. Train model:
   ```
   python train_model.py
   ```

3. Modify `config.yaml` to adjust simulation parameters and training settings.

## Models

- `PredictiveModel`: Main model using sequences of observations, actions, and ego states
- `SimplePredictiveModel`: Debug model reconstructing the last observation
- `SingleStepPredictiveModel`: Debug model predicting the next observation based on the current one
