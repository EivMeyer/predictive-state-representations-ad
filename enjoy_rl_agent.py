from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from stable_baselines3 import PPO
from utils.config_utils import config_wrapper
from environments import get_environment
from utils.policy_utils import PPOWithSRL, DetachedSRLCallback
from utils.rl_utils import create_representation_model
from utils.training_utils import compute_rl_checksums
import argparse
from environments.commonroad_env.experiment_setup import setup_rl_experiment

def enjoy_agent(env, model):

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

def load_env_and_model_from_cfg(cfg: DictConfig):
    env_class = get_environment(cfg.environment)
    env = env_class().make_env(cfg, n_envs=1, seed=cfg.seed, rl_mode=True)

    # Determine model path
    if cfg.rl_training.warmstart_model is not None:
        model_path = Path(cfg.rl_training.warmstart_model)
        if not model_path.exists():
            raise FileNotFoundError(f"Specified model path does not exist: {model_path}")
    else:
        # Default behavior: load the most recent "best_model.zip" in the project directory
        model_path = sorted(
            Path(cfg.project_dir).rglob('best_model.zip'),  # Only match the file named "best_model.zip"
            key=lambda x: x.stat().st_mtime,  # Sort by last modified time
            reverse=True  # Most recent first
        )[0]
    print("Loading model from:", model_path)

    # Create representation model and callback if needed
    srl_callback = None
    if cfg.rl_training.detached_srl:
        representation_model = create_representation_model(cfg, cfg.device)
        srl_callback = DetachedSRLCallback(cfg, representation_model)
    else:
        representation_model = None

    # Load model with SRL support
    model = PPOWithSRL.load(
        model_path,
        env=env,
        srl_callback=srl_callback,
        device=cfg.device
    )

    checksums = compute_rl_checksums(
        rl_model=model,
        srl_model=representation_model
    )
    print("Model checksums:", checksums)

    return env, model, model_path

@config_wrapper()
def main(cfg: DictConfig) -> None:
    # Load environment and model
    env, model, model_path, = load_env_and_model_from_cfg(cfg)

    print(f"Loaded model from {model_path}")

    # Enjoy the agent
    enjoy_agent(env, model)


if __name__ == "__main__":
    main()