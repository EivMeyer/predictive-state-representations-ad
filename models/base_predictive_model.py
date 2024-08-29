import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BasePredictiveModel(nn.Module, ABC):
    def __init__(self, obs_shape, action_dim, ego_state_dim, num_frames_to_predict, hidden_dim):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.ego_state_dim = ego_state_dim
        self.num_frames_to_predict = num_frames_to_predict
        self.hidden_dim = hidden_dim

    @abstractmethod
    def encode(self, observations, ego_states):
        """
        Encode the input observations and ego states.
        
        :param observations: Tensor of shape [batch_size, seq_len, channels, height, width]
        :param ego_states: Tensor of shape [batch_size, seq_len, ego_state_dim]
        :return: Encoded representation
        """
        pass

    @abstractmethod
    def decode(self, encoded_state):
        """
        Decode the encoded state to predict future frames.
        
        :param encoded_state: The encoded representation from the encode method
        :return: Tensor of shape [batch_size, num_frames_to_predict, channels, height, width]
        """
        pass

    def forward(self, batch):
        """
        Forward pass of the model.
        
        :param batch: Dictionary containing 'observations' and 'ego_states'
        :return: Predicted future frames
        """

        observations = batch['observations']
        ego_states = batch['ego_states']

        last_memory = self.encode(observations, ego_states)
        predictions = self.decode(last_memory)

        return predictions

