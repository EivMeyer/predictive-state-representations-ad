import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from utils.rl_utils import setup_rl_experiment
from commonroad_geometric.simulation.ego_simulation.control_space.keyboard_input import UserAdvanceScenarioInterrupt, UserQuitInterrupt, UserResetInterrupt, get_keyboard_action
from commonroad_geometric.learning.reinforcement.experiment import RLExperiment

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):

    experiment = setup_rl_experiment(cfg)
    env = experiment.make_env(
        scenario_dir=Path(cfg.scenario_dir),
        n_envs=1,
        seed=cfg.seed
    )

    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    obs = env.reset()
    total_reward = 0.0

    while True:
        try:
            action = get_keyboard_action(env.get_attr('renderers')[0][0].viewer)
        except UserResetInterrupt:
            env.env_method('respawn')
            action = np.array([0.0, 0.0], dtype=np.float32)
        except UserAdvanceScenarioInterrupt:
            obs = env.reset()
            total_reward = 0.0
            action = np.array([0.0, 0.0], dtype=np.float32)
        except UserQuitInterrupt:
            print("Quit game")
            return

        obs, reward, done, info = env.step([action])
        total_reward += reward.item()

        msg = f"action: {action}, reward: {reward.item():.3f} ({total_reward:.3f}), low: {info[0]['lowest_reward_computer']} ({info[0]['lowest_reward']:.3f}), high: {info[0]['highest_reward_computer']} ({info[0]['highest_reward']:.3f}), t: {info[0]['time_step']}"
        print(msg)

        if not done:
            env.render('rgb_array')
        else:
            total_reward = 0.0

if __name__ == "__main__":
    main()