from omegaconf import DictConfig
from pathlib import Path
from stable_baselines3 import PPO
from utils.config_utils import config_wrapper
from environments import get_environment

@config_wrapper()
def main(cfg: DictConfig) -> None:
    # Create the environment
    env_class = get_environment(cfg.environment)
    env = env_class().make_env(cfg, n_envs=1, seed=cfg.seed, rl_mode=True)

    # Find and load the model
    model_path = sorted(Path(cfg.project_dir).rglob('*.zip'), 
                       key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print(f"Loading model from: {model_path}")
    
    try:
        model = PPO.load(model_path, env=env)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

    # Reset environment and model states
    obs = env.reset()
    if hasattr(model.policy, 'reset_noise'):
        try:
            model.policy.reset_noise()
        except Exception as e:
            print(f"Error resetting noise: {str(e)}")
            pass
    
    total_reward = 0.0
    episode_count = 0

    print("\nStarting interaction with environment...")
    try:
        while True:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, rewards, dones, info = env.step(action)
            total_reward += rewards.item()
            
            # Print step information
            print(f"\rAction: {action}, Reward: {rewards.item():.3f}, "
                  f"Total Reward: {total_reward:.3f}", end="")

            # Handle episode end
            if dones:
                episode_count += 1
                print(f"\nEpisode {episode_count} completed. "
                      f"Total Reward: {total_reward:.3f}")
                
                if info[0].get('termination_reason'):
                    print(f"Termination reason: {info[0]['termination_reason']}")
                
                obs = env.reset()
                try:
                    model.policy.reset_noise()
                except Exception as e:
                    print(f"Error resetting noise: {str(e)}")
                    pass
                total_reward = 0.0

            # Render
            env.render('rgb_array')

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()

if __name__ == "__main__":
    main()