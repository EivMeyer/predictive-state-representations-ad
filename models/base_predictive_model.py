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
