from omegaconf import DictConfig
import numpy as np
from pathlib import Path
import heapq
import cv2
from typing import List, Dict, Any, Tuple
import time
from utils.config_utils import config_wrapper
from environments import get_environment
from stable_baselines3 import PPO
from utils.policy_utils import PPOWithSRL
from dataclasses import dataclass
from datetime import datetime
import torch
from collections import deque
import signal
import sys
from enjoy_rl_agent import load_env_and_model_from_cfg

@dataclass
class Episode:
    frames: List[np.ndarray]
    cumulative_reward: float
    success: bool
    total_steering: float
    total_acceleration: float
    timestamp: datetime
    
    def get_score(self) -> float:
        """Calculate episode score based on total steering and acceleration."""
        if not self.success:
            return float('-inf')
        return self.total_steering + self.total_acceleration
    
    def save_video(self, filepath: Path, fps: int = 30) -> None:
        """Save the episode frames as an MP4 video."""
        if not self.frames:
            return
            
        height, width = self.frames[0].shape[:2]
        
        # Add text overlay to first frame with episode stats
        first_frame = self.frames[0].copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_lines = [
            f"Success: {self.success}",
            f"Reward: {self.cumulative_reward:.2f}",
            f"Total Steering: {self.total_steering:.2f}",
            f"Total Acceleration: {self.total_acceleration:.2f}",
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        y = 30
        for line in text_lines:
            cv2.putText(first_frame, line, (10, y), font, 0.7, (255, 255, 255), 2)
            y += 30
            
        # Replace first frame with overlay version
        self.frames[0] = first_frame
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
        
        for frame in self.frames:
            out.write(frame)
            
        out.release()

class EpisodeManager:
    def __init__(self, output_dir: Path, num_episodes: int):
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.episodes: List[Episode] = []
        self.total_episodes = 0
        self.successful_episodes = 0
        
    def add_episode(self, episode: Episode) -> bool:
        """
        Add an episode if it's better than the current worst episode.
        Returns True if episode was added.
        """
        if len(self.episodes) < self.num_episodes:
            heapq.heappush(self.episodes, (episode.get_score(), self.total_episodes, episode))
            self._save_episode(episode)
            self.total_episodes += 1
            if episode.success:
                self.successful_episodes += 1
            return True
            
        if episode.get_score() > self.episodes[0][0]:  # Better than worst episode
            _, _, old_episode = heapq.heapreplace(self.episodes, (episode.get_score(), self.total_episodes, episode))
            self._save_episode(episode)
            self.total_episodes += 1
            if episode.success:
                self.successful_episodes += 1
            return True
            
        self.total_episodes += 1
        if episode.success:
            self.successful_episodes += 1
        return False
        
    def _save_episode(self, episode: Episode):
        """Save episode video with timestamp and score in filename."""
        timestamp = episode.timestamp.strftime("%Y%m%d_%H%M%S")
        score = episode.get_score()
        filename = f"episode_{timestamp}_score_{score:.2f}.mp4"
        episode.save_video(self.output_dir / filename)
        
    def print_stats(self):
        """Print current statistics."""
        print("\nEpisode Statistics:")
        print(f"Total Episodes: {self.total_episodes}")
        print(f"Successful Episodes: {self.successful_episodes}")
        success_rate = (self.successful_episodes / self.total_episodes * 100) if self.total_episodes > 0 else 0
        print(f"Success Rate: {success_rate:.2f}%")
        print("\nTop Episodes:")
        sorted_episodes = sorted(self.episodes, reverse=True)
        for i, (score, _, episode) in enumerate(sorted_episodes, 1):
            print(f"{i}. Score: {score:.2f}, Reward: {episode.cumulative_reward:.2f}, "
                  f"Steering: {episode.total_steering:.2f}, Acceleration: {episode.total_acceleration:.2f}")

def record_episode(env, model) -> Episode:
    """Record a single episode and return Episode object."""
    frames = []
    total_reward = 0.0
    obs = env.reset()
    total_steering = 0.0
    total_acceleration = 0.0
    done = False
    
    while not done:
        # Render and capture frame
        frame = env.render('rgb_array')
        frames.append(frame)
        
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        
        # Update totals
        total_steering += abs(action[0][0])  # Assuming steering is first
        total_acceleration += abs(action[0][1])  # Assuming acceleration is second
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward.item()
        
    # Determine if episode was successful
    success = info[0].get('termination_reason') == 'ReachedEnd'
        
    return Episode(
        frames=frames,
        cumulative_reward=total_reward,
        success=success,
        total_steering=total_steering,
        total_acceleration=total_acceleration,
        timestamp=datetime.now()
    )

def signal_handler(sig, frame):
    """Handle Ctrl+C by printing final stats before exiting."""
    print("\nReceived interrupt signal. Printing final statistics before exit...")
    if hasattr(signal_handler, 'episode_manager'):
        signal_handler.episode_manager.print_stats()
    sys.exit(0)

@config_wrapper()
def main(cfg: DictConfig) -> None:
    # Create output directory
    output_dir = Path(cfg.project_dir) / "videos" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env, model, model_path = load_env_and_model_from_cfg(cfg)
    
    # Initialize episode manager
    episode_manager = EpisodeManager(output_dir, cfg.recording.num_episodes_to_keep)
    signal_handler.episode_manager = episode_manager
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Recording episodes... Press Ctrl+C to exit")
    print(f"Videos will be saved to: {output_dir}")
    
    try:
        while True:  # Run indefinitely until interrupted
            episode = record_episode(env, model)
            if episode_manager.add_episode(episode):
                print(f"\nNew episode recorded! Score: {episode.get_score():.2f}")
                episode_manager.print_stats()
            
            # Optional: Add a small delay between episodes
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        episode_manager.print_stats()
    finally:
        env.close()

if __name__ == "__main__":
    main()