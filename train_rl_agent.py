from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.utils import configure_logger
import wandb
import torch
from utils.file_utils import find_model_path
from utils.config_utils import config_wrapper
from environments import get_environment
from utils.rl_utils import VideoRecorderEvalCallback, DebugCallback, BaseWandbCallback, create_representation_model

def initialize_ppo_model(cfg, env, device):
    model = None
    tensorboard_log = None

    # Set up tensorboard logging path
    if cfg.wandb.enabled:
        tensorboard_log = Path(cfg.project_dir) / "tensorboard_logs"
        tensorboard_log.mkdir(parents=True, exist_ok=True)
        tensorboard_log = str(tensorboard_log)

    # Apply warmstart if specified
    if cfg.rl_training.warmstart_model:
        model = load_warmstart_ppo_model(cfg.project_dir, cfg.rl_training.warmstart_model, env, device)
        if model:
            print(f"Loaded warmstart model from {cfg.rl_training.warmstart_model}")
            # Update the model's tensorboard log path
            if tensorboard_log:
                model.tensorboard_log = tensorboard_log
                # Recreate the logger with the new path
                model.set_logger(configure_logger(
                    verbose=model.verbose,
                    tensorboard_log=tensorboard_log,
                    tb_log_name="PPO_warmstart"
                ))
        else:
            print(f"Warning: Warmstart model not found at {cfg.rl_training.warmstart_model}")

    if model is None:
        print("No warmstart model loaded. Training from scratch")


        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=cfg.rl_training.learning_rate,
            policy_kwargs={
                "net_arch": OmegaConf.to_container(cfg.rl_training.net_arch, resolve=True)
            },
            n_steps=cfg.rl_training.n_steps,
            batch_size=cfg.rl_training.batch_size,
            n_epochs=cfg.rl_training.n_epochs,
            gamma=cfg.rl_training.gamma,
            gae_lambda=cfg.rl_training.gae_lambda,
            clip_range=cfg.rl_training.clip_range,
            ent_coef=cfg.rl_training.ent_coef,
            vf_coef=cfg.rl_training.vf_coef,
            max_grad_norm=cfg.rl_training.max_grad_norm,
            tensorboard_log=tensorboard_log
        )

    return model

def load_warmstart_ppo_model(project_dir, model_path, env, device):
    """Load a pre-trained model for warmstarting."""
    full_model_path = find_model_path(project_dir, model_path)
    if full_model_path is None:
        return None
    try:
        # Load the model without specifying tensorboard_log
        model = PPO.load(full_model_path, env=env, device=device, tensorboard_log=None)
        return model
    except Exception as e:
        print(f"Error loading warmstart model: {e}")
        return None


@config_wrapper()
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Initialize wandb if enabled
    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project + "-" + cfg.environment + '-RL', config=cfg_dict)

    # Create the environment
    env_class = get_environment(cfg.environment)
    env_instance = env_class()
    env = env_instance.make_env(cfg, n_envs=cfg.rl_training.num_envs, seed=cfg.seed, rl_mode=True)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else cfg.device)
    if not torch.cuda.is_available():
        print("CUDA is not available. Training on CPU.")
        device = torch.device("cpu")
    cfg.device = str(device)

    model = initialize_ppo_model(cfg, env, device)

    # Setup evaluation environment
    eval_env = env_instance.make_env(
        cfg,
        n_envs=1,
        seed=cfg.seed + 1000,
        rl_mode=True
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
        callbacks.append(BaseWandbCallback())
    if cfg.debug_mode:
        debug_callback = DebugCallback()
        callbacks.append(debug_callback)
    custom_callbacks = env_instance.custom_callbacks(cfg_dict)
    callbacks.extend(custom_callbacks)

    if cfg.rl_training.online_srl:
        from utils.baselines_utils import DetachedSRLCallback
        representation_model = create_representation_model(cfg, device)
        callbacks.append(DetachedSRLCallback(
            cfg,
            representation_model
        ))

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