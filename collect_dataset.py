from omegaconf import DictConfig, OmegaConf
import numpy as np
from utils.dataset_utils import EnvironmentDataset
from pathlib import Path
from tqdm import tqdm
from utils.config_utils import config_wrapper
from environments import get_environment

def collect_episodes(cfg_dict, env, num_episodes):
    dataset = EnvironmentDataset(cfg_dict)

    num_envs = env.num_envs if hasattr(env, 'num_envs') else 1
    
    episodes_collected = 0
    with tqdm(total=num_episodes, desc="Collecting episodes") as pbar:
        while episodes_collected < num_episodes:
            obs_sequences, action_sequences, ego_state_sequences = [], [], []
            next_obs_sequences, next_action_sequences, done_sequences = [], [], []
            
            obs = env.reset()
            
            # Collect observations
            for t in range(cfg_dict['dataset']['t_obs'] * (cfg_dict['dataset']['obs_skip_frames'] + 1)):
                actions = [env.action_space.sample() for _ in range(num_envs)]
                step_tuple = env.step(actions)
                if len(step_tuple) == 4:
                    new_obs, rewards, dones, infos = env.step(actions)
                else:
                    new_obs, rewards, dones, _, infos = env.step(actions)
                
                if isinstance(dones, bool):
                    # Put all return values in a list
                    new_obs = [new_obs]
                    rewards = [rewards]
                    dones = [dones]
                    infos = [infos]

                try:
                    ego_states = env.get_attr('ego_vehicle_simulation')
                    ego_states = [np.array([e.ego_vehicle.state.velocity, 
                                            e.ego_vehicle.state.acceleration, 
                                            e.ego_vehicle.state.steering_angle, 
                                            e.ego_vehicle.state.yaw_rate]) for e in ego_states]
                except AttributeError: # THIS IS A HACK FOR THE COMMONROAD ENVIRONMENT. REMOVE THIS LATER. REPLACE WITH A BETTER SOLUTION
                    ego_states = [np.zeros(4) for _ in range(num_envs)]
                
                
                if t % (cfg_dict['dataset']['obs_skip_frames'] + 1) == 0:
                    for i in range(num_envs):
                        if len(obs_sequences) <= i:
                            obs_sequences.append([])
                            action_sequences.append([])
                            ego_state_sequences.append([])
                        
                        obs_sequences[i].append(obs[i])
                        action_sequences[i].append(actions[i])
                        ego_state_sequences[i].append(ego_states[i])
                
                obs = new_obs
            
            # Collect predictions
            for t in range(cfg_dict['dataset']['t_pred'] * (cfg_dict['dataset']['pred_skip_frames'] + 1)):
                actions = [np.zeros(env.action_space.shape) for _ in range(num_envs)]
                
                if len(step_tuple) == 4:
                    new_obs, rewards, dones, infos = env.step(actions)
                else:
                    new_obs, rewards, dones, _, infos = env.step(actions)

                if isinstance(dones, bool):
                    # Put all return values in a list
                    new_obs = [new_obs]
                    rewards = [rewards]
                    dones = [dones]
                    infos = [infos]
                
                if t % (cfg_dict['dataset']['pred_skip_frames'] + 1) == 0:
                    for i in range(num_envs):
                        if len(next_obs_sequences) <= i:
                            next_obs_sequences.append([])
                            next_action_sequences.append([])
                            done_sequences.append([])
                        
                        next_obs_sequences[i].append(new_obs[i])
                        next_action_sequences[i].append(actions[i])
                        done_sequences[i].append(dones[i])
            
            for i in range(num_envs):
                dataset.add_episode(
                    obs_sequences[i], 
                    action_sequences[i], 
                    ego_state_sequences[i],
                    next_obs_sequences[i], 
                    next_action_sequences[i], 
                    done_sequences[i]
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
    env = env_class().make_env(cfg, n_envs=1, seed=cfg.seed)
    
    dataset = collect_episodes(cfg_dict, env, cfg.dataset.num_episodes)
    
    print(f"Dataset collection complete. Total episodes: {len(dataset)}")

if __name__ == "__main__":
    main()