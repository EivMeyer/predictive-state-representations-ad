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

    t_obs = cfg_dict['dataset']['t_obs']
    t_pred = cfg_dict['dataset']['t_pred']
    obs_skip_frames = cfg_dict['dataset']['obs_skip_frames']
    pred_skip_frames = cfg_dict['dataset']['pred_skip_frames']

    obs = env.reset()

    episodes_collected = 0
    with tqdm(total=num_episodes, desc="Collecting episodes") as pbar:
        while episodes_collected < num_episodes:
            # if env.get_attr('step_counter')[0] > 0:
            #     obs = env.reset()
            
            # Initialize sequences
            obs_sequences = [[] for _ in range(num_envs)]
            action_sequences = [[] for _ in range(num_envs)]
            ego_state_sequences = [[] for _ in range(num_envs)]

            while True:  # Keep trying until enough data is collected from a single episode
                terminated_during_obs = False
                obs_sequences[0].clear()  # Reset sequences for new episode
                action_sequences[0].clear()
                ego_state_sequences[0].clear()

                for t in range(t_obs * (obs_skip_frames + 1)):
                    actions = [env.action_space.sample() for _ in range(num_envs)]
                    next_obs, rewards, dones, infos = env.step(actions)
                    ego_simulation = env.get_attr('ego_vehicle_simulation')[0]

                    if any(dones):
                        terminated_during_obs = True
                        print(f"Episode {episodes_collected} terminated prematurely at step {t} because of reason: {infos[0]['termination_reason']}. Retrying...")
                        break

                    if t % (obs_skip_frames + 1) == 0:
                        # Append observations and actions
                        obs_sequences[0].append(obs[0])
                        action_sequences[0].append(actions[0])

                        # Extract ego state
                        ego_state = np.array([
                            ego_simulation.ego_vehicle.state.velocity,
                            ego_simulation.ego_vehicle.state.acceleration,
                            getattr(ego_simulation.ego_vehicle.state, 'steering_angle', 0.0),
                            getattr(ego_simulation.ego_vehicle.state, 'yaw_rate', 0.0)
                        ])
                        ego_state_sequences[0].append(ego_state)

                    obs = next_obs

                    print(f"Step {t} / {t_obs * (obs_skip_frames + 1)}")

                if not terminated_during_obs:
                    print(f"Collected {len(obs_sequences[0])} observations. Proceeding to prediction.")
                    break 

            # Collect predictions
            next_obs_sequences = [[] for _ in range(num_envs)]
            next_action_sequences = [[] for _ in range(num_envs)]
            done_sequences = [[] for _ in range(num_envs)]

            print("Predicting future observations...")

            t = 0
            is_done = False
            while len(next_obs_sequences[0]) < t_pred:
                actions = [0.0*env.action_space.sample() for _ in range(num_envs)] # Zero actions for prediction
                next_obs, rewards, dones, infos = env.step(actions)

                print(f"Step {t} / {t_pred}")

                if any(dones):
                    is_done = True

                if t % (pred_skip_frames + 1) == 0:
                    print(f"Predicted {len(next_obs_sequences[0])} observations.")
                    for i in range(num_envs):
                        done_sequences[i].append(is_done)
                        if is_done:
                            next_obs_sequences[i].append(np.zeros_like(next_obs[i]))
                            next_action_sequences[i].append(np.zeros_like(actions[i]))

                            # Fill remaining steps with zeros and True for dones
                            while len(next_obs_sequences[i]) < t_pred:
                                next_obs_sequences[i].append(np.zeros_like(next_obs[i]))
                                next_action_sequences[i].append(np.zeros_like(actions[i]))
                                done_sequences[i].append(True)

                        else:
                            next_obs_sequences[i].append(next_obs[i])
                            next_action_sequences[i].append(actions[i])
                            
                t += 1

            print("Episode complete.")

            # Add episodes to dataset
            for i in range(num_envs):
                obs = np.array(obs_sequences[i])
                actions = np.array(action_sequences[i])
                ego_states = np.array(ego_state_sequences[i])
                next_obs = np.array(next_obs_sequences[i])
                next_actions = np.array(next_action_sequences[i])
                dones = np.array(done_sequences[i])

                # Assert shapes
                assert obs.shape == (t_obs, *env.observation_space.shape), f"Unexpected obs shape: {obs.shape}"
                assert actions.shape == (t_obs, *env.action_space.shape), f"Unexpected actions shape: {actions.shape}"
                assert ego_states.shape == (t_obs, 4), f"Unexpected ego_states shape: {ego_states.shape}"
                assert next_obs.shape == (t_pred, *env.observation_space.shape), f"Unexpected next_obs shape: {next_obs.shape}"
                assert next_actions.shape == (t_pred, *env.action_space.shape), f"Unexpected next_actions shape: {next_actions.shape}"
                assert dones.shape == (t_pred,), f"Unexpected dones shape: {dones.shape}"

                # Assert dones consistency
                if np.any(dones):
                    first_done = np.argmax(dones)
                    assert np.all(dones[first_done:]), f"Inconsistent dones after first True at index {first_done}"

                    # Assert zeros after first done
                    assert np.all(next_obs[first_done+1:] == 0), f"Non-zero next_obs after done at index {first_done}"
                    assert np.all(next_actions[first_done+1:] == 0), f"Non-zero next_actions after done at index {first_done}"

                dataset.add_episode(obs, actions, ego_states, next_obs, next_actions, dones)
                
                episodes_collected += 1
                pbar.update(1)

                if episodes_collected >= num_episodes:
                    break

            if episodes_collected >= num_episodes:
                break

    return dataset

@config_wrapper()
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create the environment
    env_class = get_environment(cfg.environment)
    env = env_class().make_env(cfg, n_envs=1, seed=cfg.seed, collect_mode=True)
    
    dataset = collect_episodes(cfg_dict, env, cfg.dataset.num_episodes)
    
    print(f"Dataset collection complete. Total episodes: {len(dataset)}")

if __name__ == "__main__":
    main()