import numpy as np
from utils.sb3_custom.ppo import PPO
from typing import Optional, Tuple, Dict, Any, Union
import torch as th
from gymnasium import spaces
import gymnasium as gym
from typing import Type
from typing import Optional, Tuple, Dict, Any, Union

import torch
import numpy as np
import torch as th
from typing import Optional, Tuple, Dict, Any, Union, List
from utils.sb3_custom.on_policy_algorithm import ActorCriticPolicy
import os

import torch
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from pathlib import Path
import wandb
import os
from utils.rl_utils import create_representation_model


from utils.training_utils import compute_rl_checksums
from stable_baselines3.common.utils import configure_logger
import torch
import os
from typing import Any, Dict, Optional, Type, Union


import io
import pathlib
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first
from stable_baselines3.common.save_util import load_from_zip_file, recursive_getattr, recursive_setattr, save_to_zip_file
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TensorDict
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_device,
    get_schedule_fn,
    get_system_info,
    set_random_seed,
    update_learning_rate,
)
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
    unwrap_vec_normalize,
)
from stable_baselines3.common.vec_env.patch_gym import _convert_space, _patch_env

SelfBaseAlgorithm = TypeVar("SelfBaseAlgorithm", bound="BaseAlgorithm")


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process noise generator."""
    def __init__(self, action_space, mu=0., sigma=0.2, theta=0.15, dt=0.01, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.action_dim = action_space.shape[0]
        
        self.x0 = x0
        if self.x0 is None:
            self.x0 = np.zeros(self.action_dim)
        
        self.reset()
    
    def __call__(self):
        """Update and return the noise."""
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.x_prev = x
        return x
    
    def reset(self):
        """Reset the noise process."""
        self.x_prev = self.x0


class NoisyActorCriticPolicy(ActorCriticPolicy):
    """Actor critic policy that adds OU noise to actions during prediction"""
    def __init__(self, *args, noise_params: Optional[Dict[str, float]] = None,
                 noise_decay: float = 0.999, min_noise_scale: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise = None
        self.noise_params = noise_params or {}
        self.noise_scale = 1.0
        self.noise_decay = noise_decay
        self.min_noise_scale = min_noise_scale
        self._setup_noise()
        
    def _setup_noise(self):
        """Set up noise generator"""
        if self.noise is None and hasattr(self, 'action_space'):
            self.noise = OrnsteinUhlenbeckNoise(self.action_space, **self.noise_params)

    def add_noise_to_action(self, actions: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """Helper method to add noise to actions"""
        if not deterministic and self.noise is not None:
            # Get noise and convert to tensor
            noise = th.as_tensor(
                self.noise() * self.noise_scale, 
                dtype=actions.dtype,
                device=actions.device
            )
            
            # Add noise to actions
            actions = actions + noise
            
            # Clip actions to valid range
            actions = th.clamp(actions, 
                             th.as_tensor(self.action_space.low, device=actions.device),
                             th.as_tensor(self.action_space.high, device=actions.device))
            
            # Decay noise
            self.noise_scale = max(self.min_noise_scale,
                                 self.noise_scale * self.noise_decay)
            
            print(f"Noise scale: {self.noise_scale}, Noise: {noise}, Actions: {actions}, Original Actions: {actions - noise}")
        
        return actions

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """Get the action according to the policy for a given observation."""
        # Get original actions using parent class behavior
        actions = super()._predict(observation, deterministic=deterministic)
        return self.add_noise_to_action(actions, deterministic)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass in all the networks (actor and critic)"""
        # Get features
        features = self.extract_features(obs)
        if isinstance(features, tuple):
            features = features[0]

        # Get latent features for policy and value function
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Get values and distribution
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Get actions and add noise
        actions = distribution.get_actions(deterministic=deterministic)
        actions = self.add_noise_to_action(actions, deterministic)
        
        # Get log prob of the noisy actions
        # Note: This means the policy will learn with noise-adjusted actions
        log_prob = distribution.log_prob(actions)
        
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def reset_noise(self):
        """Reset noise process and scale"""
        if self.noise is not None:
            self.noise.reset()
            self.noise_scale = 1.0


class PPOWithNoise(PPO):
    """PPO that uses the noisy policy"""
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env,
        noise_params: Optional[Dict[str, float]] = None,
        noise_decay: float = 0.999,
        min_noise_scale: float = 0.1,
        *args, **kwargs
    ):
        # If using a string policy name, ensure we use our noisy policy class
        if isinstance(policy, str):
            policy = NoisyActorCriticPolicy

        # Initialize PPO with our custom policy
        super().__init__(
            policy=policy,
            env=env,
            *args,
            policy_kwargs=dict(
                noise_params=noise_params,
                noise_decay=noise_decay,
                min_noise_scale=min_noise_scale,
                **(kwargs.get('policy_kwargs', {}))
            ),
            **{k: v for k, v in kwargs.items() if k != 'policy_kwargs'}
        )

    def collect_rollouts(self, *args, **kwargs):
        """Ensure noise is reset at the start of each rollout"""
        if hasattr(self.policy, 'reset_noise'):
            self.policy.reset_noise()
        return super().collect_rollouts(*args, **kwargs)
    

class RepresentationActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self, 
        observation_space, 
        action_space, 
        lr_schedule,
        cfg,
        device = None, 
        train_representations=True,
        **kwargs
    ):
        
        representation_model = create_representation_model(cfg, device, load=cfg.rl_training.load_pretrained_representation)
        
        encoded_dim = representation_model.hidden_dim
        
        # Create encoded observation space
        encoded_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(encoded_dim,),
            dtype=np.float32
        )
        
        # Call parent init first
        super().__init__(
            observation_space=encoded_observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )
        
        # Store attributes after parent init
        self.representation_model = representation_model
        self._original_observation_space = observation_space
        self._train_representations = train_representations
        if train_representations:
            self.representation_model.train()


    def extract_features(self, obs):
        # Reshape observation if needed (B, T, H, W, C) or (B, H, W, C)
        if obs.ndim == 4:
            obs = obs.unsqueeze(1)

        obs = obs.float() / 255.0 # Normalize

        if obs.shape[-1] == 3:
            obs = obs.permute(0, 1, 4, 2, 3) # (B, T, C, H, W)
            
        # Prepare batch for representation model
        batch = {
            'observations': obs,
            'ego_states': torch.zeros(obs.shape[0], obs.shape[1], 4, device=obs.device) # TODO: Get actual ego state
        }
        
        # Get encoded state
        with torch.set_grad_enabled(self._train_representations):
            model_outputs = self.representation_model.forward(batch)
            encoded_state = model_outputs['encoded_state']
            loss = 0.0
            #  loss, loss_components = self.representation_model.compute_loss(batch, model_outputs)
            
        assert torch.isfinite(encoded_state).all()

        return encoded_state, loss

    def _get_features(self, obs):
        """Override to work with original observation space"""
        features = super()._get_features(obs)
        return features
    
    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features, loss = self.extract_features(obs)  
        if isinstance(features, tuple):
            features = features[0]
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf), loss
    
    def obs_to_tensor(self, observation):
        """Convert observation to tensor and handle preprocessing."""
        # Check if observation is already a tensor
        if isinstance(observation, torch.Tensor):
            obs_tensor = observation
        else:
            obs_tensor = torch.as_tensor(observation)
        
        # Move to correct device and convert to float
        obs_tensor = obs_tensor.to(device=self.device).float()

        # Ensure observation is in correct format
        if obs_tensor.ndim == 4:  # Single observation
            obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension

        obs_tensor = obs_tensor.float() / 255.0  # Normalize

        if obs_tensor.shape[-1] == 3:
            obs_tensor = obs_tensor.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)

        # Prepare batch for representation model
        batch = {
            'observations': obs_tensor,
            'ego_states': torch.zeros(obs_tensor.shape[0], obs_tensor.shape[1], 4, 
                                    device=self.device)
        }

        # Get encoded state
        with torch.set_grad_enabled(self._train_representations):
            encoded_state = self.representation_model.encode(batch)
            if isinstance(encoded_state, tuple):
                encoded_state = encoded_state[0]

        return encoded_state, True

    
class DetachedSRLCallback(BaseCallback):
    """
    A callback that trains a representation model and saves it alongside the RL agent.
    Ensures SRL model states are tied to specific agent checkpoints.
    """
    def __init__(self, cfg, representation_model, save_freq=None):
        super().__init__()
        self.cfg = cfg
        self.representation_model = representation_model
        
        # Initialize optimizer with proper parameters
        self.optimizer = torch.optim.Adam(
            self.representation_model.parameters(),
            lr=cfg.models.PredictiveModelV9M3.decoder_learning_rate,  # Use decoder learning rate from config
            weight_decay=cfg.training.optimizer.weight_decay
        )
        
        # Use the same save frequency as the main RL training
        self.save_freq = save_freq or cfg.rl_training.eval_freq
        
        # Initialize buffers
        self._reset_buffers()
        self.obs_shape = None
        
        # Metrics tracking
        self.running_loss = None
        self.loss_alpha = 0.99  # For exponential moving average
        self.train_count = 0

    def _on_step(self) -> bool:
        try:
            # Get current observation safely
            obs = self.training_env.unwrapped.get_attr('last_obs')[0]
            if obs is None:
                return True

            ego_state = np.zeros((4,))  # TODO: Get actual ego state
            
            if self.obs_shape is None:
                self.obs_shape = obs.shape
                
            self.terminated = self.locals['dones'][-1]
            
            if self.collecting_obs:
                if self.terminated:
                    self._reset_buffers()
                    return True
                
                if len(self.obs_buffer) < self.cfg.dataset.t_obs:
                    if self.t % (self.cfg.dataset.obs_skip_frames + 1) == 0:
                        # Validate observation before adding
                        if np.isfinite(obs).all():
                            self.obs_buffer.append(obs.copy())
                            self.ego_buffer.append(ego_state.copy())
                    self.t += 1
                    
                    if len(self.obs_buffer) == self.cfg.dataset.t_obs:
                        self.collecting_obs = False
                        self.t = 0
            else:
                if len(self.next_obs_buffer) < self.cfg.dataset.t_pred:
                    if self.t % (self.cfg.dataset.pred_skip_frames + 1) == 0:
                        if self.terminated:
                            zero_obs = np.zeros(self.obs_shape, dtype=np.float32)
                            self.next_obs_buffer.append(zero_obs)
                            self.done_buffer.append(True)
                            while len(self.next_obs_buffer) < self.cfg.dataset.t_pred:
                                self.next_obs_buffer.append(zero_obs.copy())
                                self.done_buffer.append(True)
                        else:
                            # Validate observation before adding
                            if np.isfinite(obs).all():
                                self.next_obs_buffer.append(obs.copy())
                                self.done_buffer.append(False)
                    self.t += 1
                    
                    if len(self.next_obs_buffer) == self.cfg.dataset.t_pred:
                        self._train_step()
                        self._reset_buffers()
            
            return True
            
        except Exception as e:
            print(f"Error in DetachedSRLCallback._on_step: {str(e)}")
            self._reset_buffers()
            return True
    
    def _train_step(self):
        try:
            # Validate sequence lengths
            if (len(self.obs_buffer) != self.cfg.dataset.t_obs or 
                len(self.next_obs_buffer) != self.cfg.dataset.t_pred):
                print("Invalid sequence lengths, skipping training step")
                return

            # Stack and validate sequences
            obs_sequence = np.stack(self.obs_buffer)
            ego_sequence = np.stack(self.ego_buffer)
            next_obs_sequence = np.stack(self.next_obs_buffer)
            done_sequence = np.array(self.done_buffer)

            # Validate arrays
            if not (np.isfinite(obs_sequence).all() and 
                   np.isfinite(ego_sequence).all() and 
                   np.isfinite(next_obs_sequence).all()):
                print("Non-finite values detected in sequences, skipping training step")
                return
            
            # Prepare batch
            batch = {
                'observations': (torch.from_numpy(obs_sequence).float()/255.0)
                              .unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.cfg.device),
                'ego_states': torch.from_numpy(ego_sequence).float()
                              .unsqueeze(0).to(self.cfg.device),
                'next_observations': (torch.from_numpy(next_obs_sequence).float()/255.0)
                                   .unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.cfg.device),
                'dones': torch.from_numpy(done_sequence).bool()
                         .unsqueeze(0).to(self.cfg.device)
            }
            
            # Train step with gradient clipping
            self.representation_model.train()
            self.optimizer.zero_grad()
            output = self.representation_model(batch)
            loss, loss_components = self.representation_model.compute_loss(batch, output)
            
            # Check for invalid loss
            if not torch.isfinite(loss):
                print(f"Invalid loss value: {loss.item()}, skipping update")
                return
                
            if isinstance(loss, tuple):
                loss = loss[0]
            
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.representation_model.parameters(), 1.0)
            
            self.optimizer.step()

            # Update running loss
            loss_value = loss.item()
            if self.running_loss is None:
                self.running_loss = loss_value
            else:
                self.running_loss = self.loss_alpha * self.running_loss + (1 - self.loss_alpha) * loss_value

            self.train_count += 1

            # Log metrics
            if self.cfg.wandb.enabled and self.train_count % 10 == 0:
                wandb.log({
                    "srl/loss": loss_value,
                    "srl/running_loss": self.running_loss,
                    "srl/train_count": self.train_count
                }, step=self.num_timesteps)
                
                for name, value in loss_components.items():
                    wandb.log({f"srl/loss_{name}": value}, step=self.num_timesteps)

        except Exception as e:
            print(f"Error in training step: {str(e)}")
            self._reset_buffers()

    def _reset_buffers(self):
        """Reset all buffers to initial state."""
        self.obs_buffer = []
        self.ego_buffer = []
        self.next_obs_buffer = []
        self.done_buffer = []
        self.collecting_obs = True
        self.terminated = False
        self.t = 0
    
    def _on_rollout_end(self):
        """Clean up at the end of a rollout."""
        self._reset_buffers()

    def on_training_end(self):
        """Clean up at the end of training."""
        self._reset_buffers()


class PPOWithSRL(PPO):
    """
    PPO that can optionally handle SRL integration, maintaining compatibility with regular PPO usage.
    """
    def __init__(
        self,
        policy,
        env,
        cfg,
        **kwargs
    ):
        if cfg.rl_training.end_to_end_srl:
            self.srl_mode = "end_to_end"
            train_representations = True
        elif cfg.rl_training.detached_srl:
            self.srl_mode = "detached"
            train_representations = False
        else:
            self.srl_mode = "none"
            train_representations = False
        
        # Handle end-to-end SRL case
        if self.srl_mode in ["end_to_end", "detached"]:
            if isinstance(policy, str) and policy == "MlpPolicy":
                policy = RepresentationActorCriticPolicy
            kwargs["policy_kwargs"] = {
                **(kwargs.get("policy_kwargs", {})),
                "cfg": cfg,
                "device": kwargs.get("device", "auto"),
                "train_representations": train_representations
            }
        
        super().__init__(policy=policy, env=env, **kwargs)

    def save(
        self,
        path: str,
        exclude: Optional[List[str]] = None,
        include: Optional[List[str]] = None,
    ) -> None:
        """
        Save model to path.

        Args:
            path: Path to save to
            exclude: List of parameters to exclude
            include: List of parameters to include
        """
        # Save PPO state
        super().save(path, exclude=exclude, include=include)
        
        # Save SRL state if using end_to_end or detached SRL
        if self.srl_mode in ["end_to_end", "detached"]:
            srl_path = str(path).replace('.zip', '_srl.pth')
            srl_state = {
                'srl_mode': self.srl_mode,
                'model_state': self.get_srl_state()
            }
            torch.save(srl_state, srl_path)

    @classmethod
    def load(
        cls,
        path: str,
        env=None,
        device='auto',
        custom_objects=None,
        cfg=None,
        **kwargs
    ):
        """Load both PPO and SRL states if applicable."""
        # Load PPO using the parent class's load method

        model = PPOWithSRL.load_internal(path, env, device=device, cfg=cfg, custom_objects=custom_objects, **kwargs)
        
        # Check for and load SRL state
        srl_path = str(path).replace('.zip', '_srl.pth')
        if os.path.exists(srl_path):
            srl_state = torch.load(srl_path, map_location=device)
            model.srl_mode = srl_state['srl_mode']
            model.load_srl_state(srl_state['model_state'])

        checksums = compute_rl_checksums(rl_model=model)
        print(f"Loaded PPOWithSRL model from {path}. Checksums:", checksums)
        
        return model

    @classmethod
    def load_internal(  # noqa: C901
        cls: Type[SelfBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        cfg = None,
        **kwargs,
    ) -> SelfBaseAlgorithm:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if "net_arch" in data["policy_kwargs"] and len(data["policy_kwargs"]["net_arch"]) > 0:
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(saved_net_arch[0], dict):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(data[key])

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            cfg=cfg,
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model

    def get_srl_state(self):
        """Get current SRL model and optimizer state."""
        return {
            'srl_model_state': self.policy.representation_model.state_dict(),
            'step': self.num_timesteps
        }
    
    def load_srl_state(self, srl_state):
        """Load SRL model and optimizer state."""
        self.policy.representation_model.load_state_dict(srl_state['srl_model_state'])
