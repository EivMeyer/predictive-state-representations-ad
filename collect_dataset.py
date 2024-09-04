from omegaconf import DictConfig, OmegaConf
import numpy as np
from experiment_setup import setup_base_experiment
from utils.dataset_utils import EnvironmentDataset
from pathlib import Path
from tqdm import tqdm
from utils.config_utils import config_wrapper

def collect_episodes(cfg_dict, env, num_episodes):
    dataset = EnvironmentDataset(
        data_dir=Path(cfg_dict["project_dir"]) / "dataset",
        batch_size=cfg_dict["dataset"]["batch_size"]
    )
    
    episodes_collected = 0
    with tqdm(total=num_episodes, desc="Collecting episodes") as pbar:
        while episodes_collected < num_episodes:
            obs_sequences, action_sequences, ego_state_sequences = [], [], []
            next_obs_sequences, next_action_sequences, done_sequences = [], [], []
            
            obs = env.reset()
            
            # Collect observations
            for t in range(cfg_dict['dataset']['t_obs'] * (cfg_dict['dataset']['obs_skip_frames'] + 1)):
                actions = [env.action_space.sample() for _ in range(env.num_envs)]
                new_obs, rewards, dones, infos = env.step(actions)
                
                ego_states = env.get_attr('ego_vehicle_simulation')
                ego_states = [np.array([e.ego_vehicle.state.velocity, 
                                        e.ego_vehicle.state.acceleration, 
                                        e.ego_vehicle.state.steering_angle, 
                                        e.ego_vehicle.state.yaw_rate]) for e in ego_states]
                
                if t % (cfg_dict['dataset']['obs_skip_frames'] + 1) == 0:
                    for i in range(env.num_envs):
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
                actions = [[0.0, 0.0] for _ in range(env.num_envs)]
                new_obs, rewards, dones, infos = env.step(actions)
                
                if t % (cfg_dict['dataset']['pred_skip_frames'] + 1) == 0:
                    for i in range(env.num_envs):
                        if len(next_obs_sequences) <= i:
                            next_obs_sequences.append([])
                            next_action_sequences.append([])
                            done_sequences.append([])
                        
                        next_obs_sequences[i].append(new_obs[i])
                        next_action_sequences[i].append(actions[i])
                        done_sequences[i].append(dones[i])
            
            for i in range(env.num_envs):
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
    experiment, env = setup_base_experiment(cfg_dict)
    
    dataset = collect_episodes(cfg_dict, env, cfg.dataset.num_episodes)
    
    print(f"Dataset collection complete. Total episodes: {len(dataset)}")

if __name__ == "__main__":
    main()