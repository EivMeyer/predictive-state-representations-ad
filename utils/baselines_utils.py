from stable_baselines3.common.callbacks import BaseCallback
import torch
import numpy as np

class DetachedSRLCallback(BaseCallback):
    def __init__(self, cfg, representation_model):
        super().__init__()
        self.cfg = cfg
        self.representation_model = representation_model
        self.optimizer = torch.optim.Adam(self.representation_model.parameters())
        
        # Initialize buffers
        self._reset_buffers()
        
        # Get initial observation to determine shape
        self.obs_shape = None
        
    def _on_step(self) -> bool:
        obs = self.training_env.unwrapped.get_attr('last_obs')[0]
        ego_state = np.zeros((4,))  # TODO: Get actual ego state
        
        # Initialize observation shape if not set
        if self.obs_shape is None:
            self.obs_shape = obs.shape
            
        self.terminated = self.locals['dones'][-1]
        
        if self.collecting_obs:
            if self.terminated:
                self._reset_buffers()
                return True
            
            # Collecting observation sequence
            if len(self.obs_buffer) < self.cfg.dataset.t_obs:
                if self.t % (self.cfg.dataset.obs_skip_frames + 1) == 0:
                    self.obs_buffer.append(obs.copy())  # Make sure to copy the observation
                    self.ego_buffer.append(ego_state.copy())
                self.t += 1
                
                if len(self.obs_buffer) == self.cfg.dataset.t_obs:
                    self.collecting_obs = False
                    self.t = 0
        else:
            # Collecting prediction sequence
            if len(self.next_obs_buffer) < self.cfg.dataset.t_pred:
                if self.t % (self.cfg.dataset.pred_skip_frames + 1) == 0:
                    if self.terminated:
                        # Create zero-filled observation with correct shape
                        zero_obs = np.zeros(self.obs_shape, dtype=np.float32)
                        self.next_obs_buffer.append(zero_obs)
                        self.done_buffer.append(True)
                        # Fill remaining steps with zero observations
                        while len(self.next_obs_buffer) < self.cfg.dataset.t_pred:
                            self.next_obs_buffer.append(zero_obs.copy())
                            self.done_buffer.append(True)
                    else:
                        self.next_obs_buffer.append(obs.copy())
                        self.done_buffer.append(False)
                self.t += 1
                
                # If we have collected all sequences, train
                if len(self.next_obs_buffer) == self.cfg.dataset.t_pred:
                    self._train_step()
                    self._reset_buffers()
                    
        return True
    
    def _train_step(self):
        try:
            # Convert sequences to numpy arrays with explicit shapes
            obs_sequence = np.stack(self.obs_buffer)  # (t_obs, *obs_shape)
            ego_sequence = np.stack(self.ego_buffer)  # (t_obs, ego_dim)
            next_obs_sequence = np.stack(self.next_obs_buffer)  # (t_pred, *obs_shape)
            done_sequence = np.array(self.done_buffer)  # (t_pred,)
            
            # Create batch dict with proper reshaping for the model
            batch = {
                'observations': torch.from_numpy(obs_sequence).float().unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.cfg.device),
                'ego_states': torch.from_numpy(ego_sequence).float().unsqueeze(0).to(self.cfg.device),
                'next_observations': torch.from_numpy(next_obs_sequence).float().unsqueeze(0).permute(0, 1, 4, 2, 3).to(self.cfg.device),
                'dones': torch.from_numpy(done_sequence).bool().unsqueeze(0).to(self.cfg.device)
            }
            
            # Verify shapes before training
            expected_obs_shape = (1, self.cfg.dataset.t_obs, *self.obs_shape[-3:])
            expected_next_obs_shape = (1, self.cfg.dataset.t_pred, *self.obs_shape[-3:])
            assert batch['observations'].shape[:-3] == expected_obs_shape[:-3], f"Observation shape mismatch: {batch['observations'].shape} vs {expected_obs_shape}"
            assert batch['next_observations'].shape[:-3] == expected_next_obs_shape[:-3], f"Next observation shape mismatch: {batch['next_observations'].shape} vs {expected_next_obs_shape}"
            
            # Train step
            self.representation_model.train()
            self.optimizer.zero_grad()
            output = self.representation_model(batch)
            loss = self.representation_model.compute_loss(batch, output)
            if isinstance(loss, tuple):
                loss = loss[0]
            loss.backward()
            self.optimizer.step()

            if self.cfg.verbose:
                print(f"Training step loss: {loss.item():.4f}")

        except Exception as e:
            print(f"Error in training step: {str(e)}")
            self._reset_buffers()

    def _reset_buffers(self):
        """Reset all internal buffers and state."""
        self.obs_buffer = []
        self.ego_buffer = []
        self.next_obs_buffer = []
        self.done_buffer = []
        self.collecting_obs = True
        self.terminated = False
        self.t = 0
    
    def _on_rollout_end(self):
        self._reset_buffers()

    def on_training_end(self):
        self._reset_buffers()