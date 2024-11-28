from omegaconf import DictConfig
from pathlib import Path
from stable_baselines3 import PPO
from utils.config_utils import config_wrapper
from environments import get_environment
from enjoy_rl_agent import load_env_and_model_from_cfg
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch

def analyze_array(array: np.ndarray) -> dict:
    """Compute comprehensive statistics for an array."""
    try:
        if array is None or len(array) == 0:
            return {
                "max": None,
                "min": None,
                "std": None,
                "mean": None,
                "absmean": None
            }
        return {
            "max": float(np.max(array)),
            "min": float(np.min(array)),
            "std": float(np.std(array)),
            "mean": float(np.mean(array)),
            "absmean": float(np.mean(np.abs(array)))
        }
    except Exception as e:
        print(f"Error analyzing array: {str(e)}")
        print(array)
        return {
            "max": None,
            "min": None,
            "std": None,
            "mean": None,
            "absmean": None
        }

@config_wrapper()
def main(cfg: DictConfig) -> None:
    # Load environment and model
    env, model, model_path = load_env_and_model_from_cfg(cfg)

    # Initialize metrics
    metrics = defaultdict(list)
    reward_components = set()
    termination_reasons = set()
    min_episode_length = cfg.evaluation.get('min_episode_length', 10)
    valid_episodes = 0

    num_episodes = cfg.evaluation.num_episodes
    pbar = tqdm(total=num_episodes, desc="Evaluating")

    while valid_episodes < num_episodes:
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_reward_components = defaultdict(list)
        episode_vehicle_states = defaultdict(list)
        episode_metrics = defaultdict(list)
        
        # Store episode trajectories
        episode_actions = []
        episode_values = []
        episode_observations = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            episode_actions.append(action)
            
            # Get value estimate
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs).to(model.device)
                episode_values.append(model.policy.predict_values(obs_tensor)[0].cpu().numpy())
            
            episode_observations.append(obs)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward.item()
            episode_length += 1
            
            # Track reward components
            for comp, value in info[0]['reward_components'].items():
                reward_components.add(comp)
                episode_reward_components[comp].append(value)
            
            # Track vehicle states
            for state, value in info[0]['vehicle_current_state'].items():
                episode_vehicle_states[state].append(value)

        # Check minimum length requirement
        if episode_length < min_episode_length:
            continue

        # Convert episode trajectories to numpy arrays
        episode_actions = np.array(episode_actions)
        episode_values = np.array(episode_values)
        episode_observations = np.array(episode_observations)

        # Record basic episode metrics
        final_info = info[0]
        termination_reason = final_info.get('termination_reason')
        termination_reasons.add(termination_reason)

        metrics['episode_reward'].append(episode_reward)
        metrics['episode_length'].append(episode_length)
        metrics['termination_reason'].append(termination_reason)
        metrics['scenario_id'].append(final_info['scenario_id'])
        metrics['current_num_obstacles'].append(final_info['current_num_obstacles'])
        metrics['total_num_obstacles'].append(final_info['total_num_obstacles'])
        metrics['cumulative_reward'].append(final_info['cumulative_reward'])

        # Analyze actions
        action_stats = analyze_array(episode_actions)
        for stat_name, value in action_stats.items():
            metrics[f'action_{stat_name}'].append(value)

        # Analyze values
        value_stats = analyze_array(episode_values)
        for stat_name, value in value_stats.items():
            metrics[f'value_{stat_name}'].append(value)

        # Process vehicle states
        for state_name, values in episode_vehicle_states.items():
            try:
                # Convert values to a NumPy array of floats
                values_array = np.array(values, dtype=float)
            except ValueError:
                # Skip if conversion fails due to non-numeric data
                continue
            if np.any(np.isnan(values_array)):
                # Skip if any NaN values are present
                continue

            state_stats = analyze_array(np.array(values))
            for stat_name, value in state_stats.items():
                metrics[f'vehicle_{state_name}_{stat_name}'].append(value)

        # Process aggregate vehicle stats from info
        for stat, values in final_info['vehicle_aggregate_stats'].items():
            for metric_name, value in values.items():
                metrics[f'vehicle_agg_{stat}_{metric_name}'].append(value)

        print(f"\nEpisode {valid_episodes + 1}: Reward={episode_reward:.2f}, Length={episode_length}, Reason={termination_reason}")
        
        valid_episodes += 1
        pbar.update(1)

    pbar.close()

    # Calculate final metrics
    print(f"\nCompleted {valid_episodes} valid episodes (min length: {min_episode_length})")
    total_episodes = valid_episodes + (num_episodes - valid_episodes)
    print(f"Skipped {total_episodes - valid_episodes} episodes")

    # Calculate aggregate statistics
    aggregate_metrics = {}
    
    # Process all numerical metrics
    for metric_name, values in metrics.items():
        if isinstance(values[0], (int, float)):
            aggregate_metrics[f'{metric_name}_mean'] = np.mean(values)
            aggregate_metrics[f'{metric_name}_std'] = np.std(values)
            aggregate_metrics[f'{metric_name}_min'] = np.min(values)
            aggregate_metrics[f'{metric_name}_max'] = np.max(values)

    # Calculate termination reason rates
    for reason in termination_reasons:
        if reason is not None:
            rate = sum(1 for r in metrics['termination_reason'] if r == reason) / valid_episodes
            aggregate_metrics[f'{reason}_rate'] = rate

    # Add evaluation metadata
    aggregate_metrics.update({
        'min_episode_length_threshold': min_episode_length,
        'episodes_skipped': total_episodes - valid_episodes,
        'skipped_episode_rate': (total_episodes - valid_episodes) / total_episodes,
        'model_path': str(model_path),
        'total_episodes_evaluated': total_episodes
    })

    # Print aggregate metrics
    print("\nAggregate Metrics:")
    for metric, value in aggregate_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # Save detailed metrics
    output_dir = Path(cfg.project_dir) / cfg.evaluation_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed episode-level metrics
    df = pd.DataFrame(metrics)
    df.to_csv(output_dir / f"detailed_metrics_{timestamp}.csv", index=False)
    print(f"\nDetailed metrics saved to {output_dir / f'detailed_metrics_{timestamp}.csv'}")

    # Save aggregate metrics
    pd.DataFrame([aggregate_metrics]).to_csv(output_dir / f"aggregate_metrics_{timestamp}.csv", index=False)
    print(f"Aggregate metrics saved to {output_dir / f'aggregate_metrics_{timestamp}.csv'}")

if __name__ == "__main__":
    main()