from utils.rl_utils import BaseWandbCallback
import wandb
import numpy as np

class CommonRoadWandbCallback(BaseWandbCallback):
    def __init__(self, cfg, verbose: int = 0):
        self.enabled = cfg['wandb']['enabled']
        self.n_calls = 0
        self.n_rollouts = 0
        if not self.enabled:
            return
        super(CommonRoadWandbCallback, self).__init__(verbose)

    def _on_training_start(self) -> None:
        if not self.enabled:
            return
        super()._on_training_start()
        self._termination_reasons = self.training_env.get_attr('termination_reasons')[0]

    def _log_episode_metrics(self, rollout_buffer, n_episodes_done_step, n_steps, last_done_array, last_info):
        if not self.enabled:
            return
        super()._log_episode_metrics(rollout_buffer, n_episodes_done_step, n_steps, last_done_array, last_info)

        termination_criteria_flags = dict.fromkeys(self._termination_reasons, False)
        done_indices = np.where(last_done_array)[0]
        for env_index in done_indices:
            env_info = last_info[env_index]
            termination_reason = env_info.get('termination_reason')
            termination_criteria_flags[termination_reason] = True

            env_reward_component_info = env_info['reward_component_episode_info']
            for reward_component, component_info in env_reward_component_info.items():
                for component_metric, component_value in component_info.items():
                    wandb.log({f"train/reward_{reward_component}_ep_{component_metric}": float(component_value)}, step=self.num_timesteps)

            vehicle_aggregate_stats = env_info['vehicle_aggregate_stats']
            for state, state_info in vehicle_aggregate_stats.items():
                for state_metric, state_value in state_info.items():
                    wandb.log({f"train/vehicle_{state}_ep_{state_metric}": float(state_value)}, step=self.num_timesteps)

            wandb.log({
                "train/ep_num_obstacles": float(env_info.get('total_num_obstacles', 0)),
                "train/ep_cumulative_reward": float(env_info.get('cumulative_reward', 0)),
                "train/next_reset_ready": float(env_info.get('next_reset_ready', False)),
                "train/ep_length": float(env_info.get('time_step', 0))
            }, step=self.num_timesteps)

        for termination_criteria in self._termination_reasons:
            wandb.log({
                f"train/termination_{termination_criteria}": 
                float(termination_criteria_flags[termination_criteria])
            }, step=self.num_timesteps)