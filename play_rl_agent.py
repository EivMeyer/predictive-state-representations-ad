from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np

from utils.config_utils import config_wrapper
from commonroad_geometric.simulation.ego_simulation.control_space.keyboard_input import UserAdvanceScenarioInterrupt, UserQuitInterrupt, UserResetInterrupt, get_keyboard_action
from environments import get_environment
from commonroad_geometric.common.logging import setup_logging
import logging
import sys

class EpisodeTracker:
    def __init__(self):
        self.episode_count = 0
        self.success_count = 0
        self.total_reward = 0
        self.success_rate = 0
        self.avg_reward = 0
        
    def update(self, termination_reason, episode_reward):
        self.episode_count += 1
        if termination_reason in ["ReachedEnd", "ScenarioFinished"]:
            self.success_count += 1
        self.total_reward += episode_reward
        
        self.success_rate = (self.success_count / self.episode_count) * 100
        self.avg_reward = self.total_reward / self.episode_count
        
        print(f"\n=== Episode {self.episode_count} Statistics ===")
        print(f"Success Rate: {self.success_rate:.2f}%")
        print(f"Average Reward: {self.avg_reward:.2f}")
        print(f"Total Episodes: {self.episode_count}")

@config_wrapper()
def main(cfg: DictConfig) -> None:
    if cfg.verbose:
        setup_logging(level=logging.DEBUG)

    # Use completely random seed
    np.random.seed(None)

    env_class = get_environment(cfg.environment)
    env = env_class().make_env(cfg, n_envs=1, seed=cfg.seed, rl_mode=True)

    obs = env.reset()
    episode_reward = 0.0
    tracker = EpisodeTracker()

    while True:
        try:
            action = get_keyboard_action(env.get_attr('renderers')[0][0].viewer)
        except UserResetInterrupt:
            env.env_method('respawn')
            action = np.array([0.0, 0.0], dtype=np.float32)
        except UserAdvanceScenarioInterrupt:
            tracker.update(info[0].get('termination_reason', 'Unknown'), episode_reward)
            obs = env.reset()
            episode_reward = 0.0
            action = np.array([0.0, 0.0], dtype=np.float32)
        except UserQuitInterrupt:
            print("\nFinal Statistics:")
            tracker.update(info[0].get('termination_reason', 'Unknown'), episode_reward)
            print("Quit game")
            return

        obs, reward, done, info = env.step([action])
        episode_reward += reward.item()

        msg = f"scenario: {env.get_attr('current_scenario_id')[0]}, action: {action}, reward: {reward.item():.3f} ({episode_reward:.3f}), low: {info[0]['lowest_reward_computer']} ({info[0]['lowest_reward']:.3f}), high: {info[0]['highest_reward_computer']} ({info[0]['highest_reward']:.3f}), t: {info[0]['time_step']}"
        if cfg.verbose:
            print(msg, end='\r')
            sys.stdout.flush()

        if not done:
            env.render('rgb_array')
        else:
            tracker.update(info[0].get('termination_reason', 'Unknown'), episode_reward)
            episode_reward = 0.0

if __name__ == "__main__":
    main()