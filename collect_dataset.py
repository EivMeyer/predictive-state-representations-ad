from omegaconf import DictConfig, OmegaConf
import numpy as np
from utils.dataset_utils import EnvironmentDataset
from pathlib import Path
from tqdm import tqdm
from utils.config_utils import config_wrapper
from environments import get_environment

def collect_episodes(cfg_dict, env, num_episodes):
    dataset = EnvironmentDataset(
        data_dir=Path(cfg_dict["project_dir"]) / "dataset",
        storage_batch_size=cfg_dict["dataset"]["storage_batch_size"]
    )

    is_vector_env = hasattr(env, 'num_envs')
    num_envs = env.num_envs if is_vector_env else 1
    
    t_obs = cfg_dict['dataset']['t_obs']
    t_pred = cfg_dict['dataset']['t_pred']
    total_steps = t_obs + t_pred
    
    episodes_collected = 0
    with tqdm(total=num_episodes, desc="Collecting episodes") as pbar:
        while episodes_collected < num_episodes:
            obs_sequences = [[] for _ in range(num_envs)]
            action_sequences = [[] for _ in range(num_envs)]
            ego_state_sequences = [[] for _ in range(num_envs)]
            done_sequences = [[] for _ in range(num_envs)]
            
            obs_result = env.reset()
            if isinstance(obs_result, tuple):
                obs, _ = obs_result
            else:
                obs = obs_result
                
            if not is_vector_env:
                obs = [obs]
            
            for t in range(total_steps):
                actions = [env.action_space.sample() for _ in range(num_envs)]
                if not is_vector_env:
                    actions = actions[0]
                
                step_result = env.step(actions)
                
                if is_vector_env:
                    truncateds = None # Assume no truncation
                    if len(step_result) == 5:
                        new_obs, rewards, terminateds, truncateds, infos = step_result
                    else:
                        new_obs, rewards, terminateds, infos = step_result
                    if truncateds is not None:
                        dones = [term or trunc for term, trunc in zip(terminateds, truncateds)]
                    else:
                        dones = terminateds
                else:
                    truncated = None # Assume no truncation
                    if len(step_result) == 5:
                        new_obs, reward, terminated, truncated, info = step_result
                    else:
                        new_obs, reward, terminated, info = step_result
                    if truncated is not None:
                        dones = [terminated or truncated]
                    else:
                        dones = [terminated]
                    new_obs, rewards, terminateds, truncateds, infos = [new_obs], [reward], [terminated], [truncated], [info]

                # Create a placeholder for ego_states (adjust as needed)
                ego_states = [np.zeros(4) for _ in range(num_envs)]  # Assuming 4-dimensional ego state

                for i in range(num_envs):
                    obs_sequences[i].append(obs[i])
                    action_sequences[i].append(actions[i] if is_vector_env else actions)
                    ego_state_sequences[i].append(ego_states[i])
                    done_sequences[i].append(dones[i])
                
                if all(dones):
                    break
                
                obs = new_obs
            
            for i in range(num_envs):
                sequence_length = len(obs_sequences[i])
                if sequence_length < total_steps:
                    # If the episode terminated early, we pad with the last observation
                    pad_length = total_steps - sequence_length
                    last_obs = obs_sequences[i][-1]
                    last_ego_state = ego_state_sequences[i][-1]
                    obs_sequences[i].extend([last_obs] * pad_length)
                    action_sequences[i].extend([np.zeros_like(action_sequences[i][-1])] * pad_length)
                    ego_state_sequences[i].extend([last_ego_state] * pad_length)
                    done_sequences[i].extend([True] * pad_length)
                
                # Prepare data for add_episode
                observations = np.stack(obs_sequences[i])
                actions = np.stack(action_sequences[i])
                ego_states = np.stack(ego_state_sequences[i])
                next_observations = np.stack(obs_sequences[i][1:] + [obs_sequences[i][-1]])
                next_actions = np.stack(action_sequences[i][1:] + [np.zeros_like(action_sequences[i][-1])])
                dones = np.array(done_sequences[i])
                
                dataset.add_episode(
                    observations,
                    actions,
                    ego_states,
                    next_observations,
                    next_actions,
                    dones
                )
                episodes_collected += 1
                pbar.update(1)
                
                if episodes_collected >= num_episodes:
                    break
    
    return dataset

@config_wrapper()
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create the environment
    env_class = get_environment(cfg.environment)
    env = env_class(cfg).make_env(cfg, n_envs=1, seed=cfg.seed)
    
    dataset = collect_episodes(cfg_dict, env, cfg.dataset.num_episodes)
    
    print(f"Dataset collection complete. Total episodes: {len(dataset)}")

if __name__ == "__main__":
    main()