import torch
import numpy as np
import setup as setup
from experiment_setup import load_config, setup_experiment
from utils.dataset_utils import EnvironmentDataset
import time
from pathlib import Path

def run_experiment():
    config = load_config()
    _, environment = setup_experiment(config)

    dataset = EnvironmentDataset()
    num_episodes = 0

    while num_episodes < config["dataset_options"]["num_episodes"]:
        obs_sequence, actions, ego_states, next_obs_sequence, next_actions, done_sequence = [], [], [], [], [], []
        episode_done = False

        obs = environment.reset()[0]
        
        for t in range(config["dataset_options"]["t_obs"]):
            action = environment.action_space.sample()
            new_obs, reward, done, info = environment.step(actions=[action])
            if done:
                episode_done = True
                break
            new_obs = new_obs[0]
            reward = reward[0]
            done = done[0]
            info = info[0]

            ego_state = environment.get_attr('ego_vehicle_simulation')[0].ego_vehicle.state
            obs_sequence.append(obs)
            actions.append(action)
            ego_states.append(np.array([ego_state.velocity, ego_state.acceleration, ego_state.steering_angle, ego_state.yaw_rate]))
            obs = new_obs
        
        if episode_done:
            # time.sleep(2.0)
            continue

        # time.sleep(0.5)

        for t in range(config["dataset_options"]["t_pred"]):
            action = np.array([0.0, 0.0])
            new_obs, reward, done, info = environment.step(actions=[action])
            new_obs = new_obs[0]
            reward = reward[0]
            done = done[0]
            info = info[0]

            next_obs_sequence.append(new_obs)
            next_actions.append(action)
            done_sequence.append(done)

            if done:
                episode_done = True
                break

        dataset.add_episode(obs_sequence, actions, ego_states, next_obs_sequence, next_actions, done_sequence)

        print("Successfully collected episode")
        num_episodes += 1
        # time.sleep(2.0)

    dataset_dir = Path(config["project_dir"]) / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    torch.save(dataset, dataset_dir / "dataset.pt")

if __name__ == "__main__":
    run_experiment()
