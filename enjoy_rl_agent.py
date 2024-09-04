import hydra
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
import argparse

from utils.rl_utils import setup_rl_experiment

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig, model_path: str = None):
    # Setup the environment from configuration
    experiment = setup_rl_experiment(cfg)
    env = experiment.make_env(
        scenario_dir=Path(cfg.scenario_dir),
        n_envs=1,
        seed=cfg.seed
    )

    # Load the model
    if model_path is None:
        model_path = sorted(Path(cfg.project_dir).rglob('*.zip'), key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print("Loading model from:", model_path)
    model = PPO.load(model_path, env=env)

    obs = env.reset()
    total_reward = 0.0

    print("Starting interaction with environment...")
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        total_reward += rewards
        print(f"Action: {action}, Reward: {rewards.item()}, Total Reward: {total_reward}")

        if dones:
            obs = env.reset()
            print(f"Episode completed. Total Reward: {total_reward}")
            total_reward = 0

        env.render('rgb_array')

if __name__ == "__main__":
    main()
