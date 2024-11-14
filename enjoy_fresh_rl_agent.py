from omegaconf import DictConfig
from utils.config_utils import config_wrapper
from environments import get_environment
from train_rl_agent import create_new_ppo_model

DETERMINISTIC = False

@config_wrapper()
def main(cfg: DictConfig) -> None:
    # Create the environment
    env_class = get_environment(cfg.environment)
    env = env_class().make_env(cfg, n_envs=1, seed=cfg.seed, rl_mode=True)

    model = create_new_ppo_model(cfg, env, cfg.device)
    try:
        model.policy.reset_noise()
    except:
        pass

    obs = env.reset()
    total_reward = 0.0

    print("Starting interaction with environment...")
    while True:
        action, _states = model.predict(obs, deterministic=DETERMINISTIC)
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
