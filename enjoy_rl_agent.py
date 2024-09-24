from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from stable_baselines3 import PPO
from utils.config_utils import config_wrapper
from environments import get_environment

@config_wrapper()
def main(cfg: DictConfig) -> None:
    # Create the environment
    env_class = get_environment(cfg.environment)
    env = env_class().make_env(cfg, n_envs=1, seed=cfg.seed, rl_mode=True)

    # Load the model. We load the most recent model in the project directory
    model_path = sorted(Path(cfg.project_dir).rglob('*.zip'), key=lambda x: x.stat().st_mtime, reverse=True)[0]
    print("Loading model from:", model_path)
    model = PPO.load(model_path, env=env)

    obs = env.reset()
    total_reward = 0.0

    print("Starting interaction with environment...")
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        total_reward += rewards
        print(f"Action: {action}, Reward: {rewards.item()}, Total Reward: {total_reward}")

        if dones:
            obs = env.reset()
            print(f"Episode completed. Total Reward: {total_reward}")
            total_reward = 0

        env.render('rgb_array')

if __name__ == "__main__":
    main()
