import numpy as np
from stable_baselines3 import PPO
from typing import Optional, Tuple, Dict, Any, Union
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Type
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
from typing import Optional, Tuple, Dict, Any, Union

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