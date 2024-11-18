from omegaconf import DictConfig
from pathlib import Path
from stable_baselines3 import PPO
from utils.config_utils import config_wrapper
from environments import get_environment
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

@config_wrapper()
def main(cfg: DictConfig) -> None:
    # Create the environment
    env_class = get_environment(cfg.environment)
    env = env_class().make_env(cfg, n_envs=1, seed=cfg.seed, rl_mode=True)

    # Load the model
    model_path = sorted(Path(cfg.project_dir).rglob('*.zip'), key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print("Loading model from:", model_path)
    model = PPO.load(model_path, env=env)

    # Initialize metrics
    metrics = defaultdict(list)
    reward_components = set()
    termination_reasons = set()
    min_episode_length = cfg.evaluation.get('min_episode_length', 10)  # Default to 10 if not specified
    valid_episodes = 0

    num_episodes = cfg.evaluation.num_episodes
    pbar = tqdm(total=num_episodes, desc="Evaluating")

    while valid_episodes < num_episodes:
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_reward_components = defaultdict(list)
        episode_metrics = defaultdict(list)
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward.item()
            episode_length += 1
            
            # Dynamically track reward components
            for comp, value in info[0]['reward_components'].items():
                reward_components.add(comp)
                episode_reward_components[comp].append(value)

        # Check if episode meets minimum length requirement
        if episode_length < min_episode_length:
            print(f"\nSkipping episode {valid_episodes + 1} (length {episode_length} < minimum {min_episode_length})")
            continue

        # Extract final episode info
        final_info = info[0]

        # Record termination reason
        termination_reason = final_info.get('termination_reason')
        termination_reasons.add(termination_reason)

        # Calculate metrics for this episode
        metrics['episode_reward'].append(episode_reward)
        metrics['episode_length'].append(episode_length)
        metrics['termination_reason'].append(termination_reason)
        metrics['scenario_id'].append(final_info['scenario_id'])
        metrics['current_num_obstacles'].append(final_info['current_num_obstacles'])
        metrics['total_num_obstacles'].append(final_info['total_num_obstacles'])
        metrics['cumulative_reward'].append(final_info['cumulative_reward'])

        # Calculate averages from vehicle_aggregate_stats
        for stat, values in final_info['vehicle_aggregate_stats'].items():
            metrics[f'avg_{stat}'].append(values['mean'])

        # Calculate average reward components
        for comp in reward_components:
            metrics[f'avg_{comp}'].append(np.mean(episode_reward_components[comp]))

        valid_episodes += 1
        pbar.update(1)

    pbar.close()

    print(f"\nCompleted evaluation with {valid_episodes} valid episodes (minimum length: {min_episode_length})")
    total_episodes = valid_episodes + (num_episodes - valid_episodes)
    print(f"Skipped {total_episodes - valid_episodes} episodes due to length requirement")

    # Calculate aggregate metrics
    aggregate_metrics = {
        'mean_episode_reward': np.mean(metrics['episode_reward']),
        'std_episode_reward': np.std(metrics['episode_reward']),
        'mean_episode_length': np.mean(metrics['episode_length']),
        'std_episode_length': np.std(metrics['episode_length']),
        'min_episode_length_threshold': min_episode_length,
        'episodes_skipped': total_episodes - valid_episodes,
        'skipped_episode_rate': (total_episodes - valid_episodes) / total_episodes
    }

    # Calculate rates for each termination reason
    for reason in termination_reasons:
        if reason is not None:
            rate = sum(1 for r in metrics['termination_reason'] if r == reason) / valid_episodes
            aggregate_metrics[f'{reason}_rate'] = rate

    # Calculate mean for all averaged metrics
    for key in metrics:
        if key.startswith('avg_'):
            aggregate_metrics[f'mean_{key[4:]}'] = np.mean(metrics[key])

    # Print aggregate metrics
    print("\nAggregate Metrics:")
    for metric, value in aggregate_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # Save detailed metrics to CSV
    df = pd.DataFrame(metrics)
    output_dir = Path(cfg.project_dir) / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "detailed_metrics.csv", index=False)
    print(f"\nDetailed metrics saved to {output_dir / 'detailed_metrics.csv'}")

    # Save aggregate metrics to CSV
    pd.DataFrame([aggregate_metrics]).to_csv(output_dir / "aggregate_metrics.csv", index=False)
    print(f"Aggregate metrics saved to {output_dir / 'aggregate_metrics.csv'}")

if __name__ == "__main__":
    main()