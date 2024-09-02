from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
import wandb

from utils.rl_utils import setup_rl_experiment, VideoRecorderEvalCallback, DebugCallback, WandbCallback
from commonroad_geometric.learning.reinforcement.training.custom_callbacks import LogEpisodeMetricsCallback, LogPolicyMetricsCallback

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Initialize wandb if enabled
    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project + '-RL', config=OmegaConf.to_container(cfg, resolve=True))

    experiment = setup_rl_experiment(cfg)

    env = experiment.make_env(
        scenario_dir=cfg.scenario_dir,
        n_envs=cfg.rl_training.num_envs,
        seed=cfg.seed
    )
    cfg.dataset.num_workers

    # Extract the net_arch configuration
    net_arch = {
        "pi": cfg.rl_training.ppo_model.net_arch.pi,
        "vf": cfg.rl_training.ppo_model.net_arch.vf
    }

    # Create the RL agent
    model = PPO("MlpPolicy", env, verbose=1, device=cfg.device,
                learning_rate=cfg.rl_training.learning_rate,
                policy_kwargs={"net_arch": net_arch},
                n_steps=cfg.rl_training.n_steps,
                batch_size=cfg.rl_training.batch_size,
                n_epochs=cfg.rl_training.n_epochs,
                gamma=cfg.rl_training.gamma,
                gae_lambda=cfg.rl_training.gae_lambda,
                clip_range=cfg.rl_training.clip_range,
                ent_coef=cfg.rl_training.ent_coef,
                vf_coef=cfg.rl_training.vf_coef,
                max_grad_norm=cfg.rl_training.max_grad_norm,
                tensorboard_log=Path(cfg.project_dir) / "tensorboard_logs" if cfg.wandb.enabled else None
    )

    # Setup evaluation environment
    eval_env = experiment.make_env(
        scenario_dir=cfg.scenario_dir,
        n_envs=1,
        seed=cfg.seed + 1000,  # Different seed for eval env
    )
    
    # Setup video recording
    video_folder = Path(cfg.project_dir) / cfg.rl_training.video_folder
    video_folder.mkdir(parents=True, exist_ok=True)

    # Setup evaluation callback with video recording
    eval_callback = VideoRecorderEvalCallback(
        eval_env=eval_env,
        video_folder=str(video_folder),
        video_freq=cfg.rl_training.video_freq,
        video_length=cfg.rl_training.video_length,
        n_eval_episodes=cfg.rl_training.n_eval_episodes,
        best_model_save_path=Path(cfg.project_dir) / cfg.rl_training.save_path,
        log_path=Path(cfg.project_dir) / cfg.rl_training.log_path,
        eval_freq=cfg.rl_training.eval_freq,
        deterministic=True,
        render=False
    )

    # Setup callbacks
    callbacks = [eval_callback]
    if cfg.wandb.enabled:
        callbacks.append(WandbCallback())
    if cfg.debug_mode:
        debug_callback = DebugCallback()
        callbacks.append(debug_callback)

    # Train the agent
    model.learn(total_timesteps=cfg.rl_training.total_timesteps, callback=callbacks,
                tb_log_name="ppo_run" if cfg.wandb.enabled else None)

    # Save the final model
    final_model_path = Path(cfg.project_dir) / cfg.rl_training.save_path / "final_model"
    model.save(final_model_path)

    if cfg.wandb.enabled:
        wandb.save(str(final_model_path))
        wandb.finish()

if __name__ == "__main__":
    main()