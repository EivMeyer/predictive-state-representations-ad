from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np

from utils.config_utils import config_wrapper
from commonroad_geometric.simulation.ego_simulation.control_space.keyboard_input import UserAdvanceScenarioInterrupt, UserQuitInterrupt, UserResetInterrupt, get_keyboard_action
from environments import get_environment

@config_wrapper()
def main(cfg: DictConfig) -> None:

    # Create the environment
    env_class = get_environment(cfg.environment)
    env = env_class().make_env(cfg, n_envs=1, seed=cfg.seed, rl_mode=True)

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

        msg = f"scenario: {env.get_attr('current_scenario_id')[0]}, action: {action}, reward: {reward.item():.3f} ({total_reward:.3f}), low: {info[0]['lowest_reward_computer']} ({info[0]['lowest_reward']:.3f}), high: {info[0]['highest_reward_computer']} ({info[0]['highest_reward']:.3f}), t: {info[0]['time_step']}"
        print(msg)

        if not done:
            env.render('rgb_array')
        else:
            total_reward = 0.0

if __name__ == "__main__":
    main()