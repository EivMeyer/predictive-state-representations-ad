# base_predictive_model.py
import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class BasePredictiveModel(nn.Module, ABC):
    def __init__(self, obs_shape, action_dim, ego_state_dim, cfg):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.ego_state_dim = ego_state_dim
        self.cfg = cfg
        
        self.num_frames_to_predict=self.cfg['dataset']['t_pred']
        self.hidden_dim=self.cfg['training']['hidden_dim']

    @abstractmethod
    def encode(self, batch):
        pass

    @abstractmethod
    def decode(self, batch, encoded_state):
        pass

    def decode_image(self, batch, encoded_state):
        # Overwrite this method if your model does not predict images
        return self.decode(batch, encoded_state)

    @abstractmethod
    def compute_loss(self, batch, encoded_state, predictions) -> Tuple[Tensor, Dict[str, float]]:
        pass

    def forward(self, batch):
        encoded_state = self.encode(batch)
        predictions = self.decode(batch, encoded_state)

        return {
            'predictions': predictions,
            'encoded_state': encoded_state
        }

    def get_save_state(self) -> Dict[str, Any]:
        """Get state dict for saving, including any nested models."""
        state_dict = self.state_dict()
        metadata = {
            'model_type': self.__class__.__name__,
            'obs_shape': self.obs_shape,
            'action_dim': self.action_dim,
            'ego_state_dim': self.ego_state_dim,
        }
        
        # Add nested model states (e.g., autoencoder)
        nested_states = {}
        if hasattr(self, 'autoencoder'):
            nested_states['autoencoder'] = {
                'state_dict': self.autoencoder.state_dict(),
                'model_type': self.autoencoder.__class__.__name__
            }

        return {
            'state_dict': state_dict,
            'metadata': metadata,
            'nested_states': nested_states
        }
    
    def load_save_state(self, save_state: Dict[str, Any], strict: bool = False) -> None:
        """Load a saved state, including nested models."""
        metadata = save_state.get('metadata', {})
        
        # Verify model compatibility
        if metadata.get('model_type') != self.__class__.__name__:
            print(f"Warning: Loading state from {metadata.get('model_type')} into {self.__class__.__name__}")

        # Load main state dict
        self.load_state_dict(save_state['state_dict'], strict=strict)

        # Load nested states if they exist
        nested_states = save_state.get('nested_states', {})
        if 'autoencoder' in nested_states and hasattr(self, 'autoencoder'):
            self.autoencoder.load_state_dict(
                nested_states['autoencoder']['state_dict'], 
                strict=strict
            )