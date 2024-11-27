from omegaconf import DictConfig
import numpy as np
from pathlib import Path
import heapq
import cv2
from typing import List, Dict, Any, Tuple, Optional
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
import matplotlib.pyplot as plt
from io import BytesIO
from environments.commonroad_env.observers import RepresentationObserver

@dataclass
class Episode:
    frames: List[np.ndarray]
    debug_frames: Optional[List[np.ndarray]]  # New field for debug visualization frames
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
    
    def save_video(self, filepath: Path, fps: int = 20) -> None:
        """Save the episode frames as an MP4 video, optionally including debug visualization."""
        if not self.frames:
            return
            
        base_height, base_width = self.frames[0].shape[:2]
        
        # If we have debug frames, stack them vertically with main frames
        if self.debug_frames and len(self.debug_frames) == len(self.frames):
            debug_height, debug_width = self.debug_frames[0].shape[:2]
            combined_height = base_height + debug_height
            combined_width = max(base_width, debug_width)
            
            # Create combined frames
            combined_frames = []
            for debug_frame, main_frame in zip(self.debug_frames, self.frames):
                # Resize debug frame if width doesn't match
                if debug_width != combined_width:
                    debug_frame = cv2.resize(debug_frame, (combined_width, debug_height))
                
                # Create a white background for the main frame
                main_background = np.full((base_height, combined_width, 3), 255, dtype=np.uint8)
                
                # Calculate padding for centering
                pad_left = (combined_width - base_width) // 2
                
                # Place the main frame in the center
                main_background[:, pad_left:pad_left + base_width] = main_frame
                    
                combined_frame = np.vstack([debug_frame, main_background])
                combined_frames.append(combined_frame)
        else:
            combined_frames = self.frames
            combined_height, combined_width = base_height, base_width
        
        # Add text overlay to first frame with episode stats
        first_frame = combined_frames[0].copy()
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
            
        combined_frames[0] = first_frame
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(filepath), fourcc, fps, (combined_width, combined_height))
        
        for frame in combined_frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        print(f"Saved episode video: {filename}")
        
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

def capture_debug_plot(observer: RepresentationObserver) -> Optional[np.ndarray]:
    """Capture the current matplotlib figure as a numpy array."""
    try:
        fig = observer.get_debug_figure()
        if fig is None:
            return None
            
        # Make sure the figure has the right background color
        fig.patch.set_facecolor('white')
        
        # Get dimensions of the figure in pixels
        width, height = fig.canvas.get_width_height()
        
        # Get the RGB buffer from the figure
        buf = fig.canvas.buffer_rgba()
        # Convert it to a numpy array
        image = np.asarray(buf)
        
        # Convert RGBA to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        return image_rgb
    except Exception as e:
        print(f"Failed to capture debug plot: {e}")
        return None

class LivePlotter:
    def __init__(self, window_size=100):
        """Initialize the live plotter with a fixed window size."""
        self.window_size = window_size
        self.steering_history = deque(maxlen=window_size)
        self.accel_history = deque(maxlen=window_size)
        self.reward_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
        self.current_time = 0
        self.cumulative_reward = 0
        
        # Make figure wider - increase the width from 4 to 8
        self.fig, (self.ax_steering, self.ax_accel, self.ax_reward) = plt.subplots(
            3, 1, figsize=(8, 8),  # Changed width from 4 to 8
            gridspec_kw={'height_ratios': [1, 1, 1.5]}
        )
        
        # Initialize line plots
        self.steering_line, = self.ax_steering.plot([], [], 'b-', label='Steering')
        self.accel_line, = self.ax_accel.plot([], [], 'g-', label='Acceleration')
        self.reward_line, = self.ax_reward.plot([], [], 'orange', label='Cumulative Reward')
        
        # Add limit lines for steering and acceleration
        self.ax_steering.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
        self.ax_steering.axhline(y=-1.0, color='r', linestyle='--', alpha=0.3)
        self.ax_accel.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
        self.ax_accel.axhline(y=-1.0, color='r', linestyle='--', alpha=0.3)
        
        # Configure axes
        self.ax_steering.set_ylabel('Steering\nAngle')
        self.ax_accel.set_ylabel('Acceleration')
        self.ax_reward.set_ylabel('Cumulative\nReward')
        self.ax_reward.set_xlabel('Time Steps')
        
        # Add grid
        for ax in [self.ax_steering, self.ax_accel, self.ax_reward]:
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
        
        plt.tight_layout()
        
    def update(self, steering: float, acceleration: float, reward: float) -> np.ndarray:
        """Update the plots with new data and return the figure as an image."""
        self.current_time += 1
        self.cumulative_reward += reward
        
        # Update histories
        self.time_history.append(self.current_time)
        self.steering_history.append(steering)
        self.accel_history.append(acceleration)
        self.reward_history.append(self.cumulative_reward)
        
        # Update line data
        self.steering_line.set_data(self.time_history, self.steering_history)
        self.accel_line.set_data(self.time_history, self.accel_history)
        self.reward_line.set_data(self.time_history, self.reward_history)
        
        # Adjust axes limits
        for ax in [self.ax_steering, self.ax_accel, self.ax_reward]:
            ax.relim()
            ax.autoscale_view()
            
        # Set x-axis limits to show fixed window
        if len(self.time_history) > 0:
            x_min = max(0, self.current_time - self.window_size)
            x_max = self.current_time
            for ax in [self.ax_steering, self.ax_accel, self.ax_reward]:
                ax.set_xlim(x_min, x_max)
        
        # Update steering and acceleration y-limits
        self.ax_steering.set_ylim(-1.2, 1.2)
        self.ax_accel.set_ylim(-1.2, 1.2)
        
        # Draw and convert to image
        self.fig.canvas.draw()
        
        # Convert plot to numpy array
        plot_img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        
        return plot_img
    
    def reset(self):
        """Reset the plotter for a new episode."""
        self.current_time = 0
        self.cumulative_reward = 0
        self.steering_history.clear()
        self.accel_history.clear()
        self.reward_history.clear()
        self.time_history.clear()

def record_episode(env, model, record_debug: bool = False) -> Episode:
    """Record a single episode with debug visualization and live plots."""
    frames = []
    debug_frames = [] if record_debug else None
    total_reward = 0.0
    obs = env.reset()
    total_steering = 0.0
    total_acceleration = 0.0
    done = False
    
    # Get the RepresentationObserver if it exists
    observer = None
    if record_debug:
        try:
            unwrapped_env = env.env if hasattr(env, 'env') else env
            observer = unwrapped_env.get_attr('observer')[0]
            if not isinstance(observer, RepresentationObserver):
                observer = None
        except:
            print("Warning: Could not access RepresentationObserver")
            observer = None
    
    # Initialize live plotter
    plotter = LivePlotter(window_size=100)
    
    while not done:
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        steering = action[0][0]
        acceleration = action[0][1]
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward.item()
        
        # Update totals
        total_steering += abs(steering)
        total_acceleration += abs(acceleration)
        
        # Render environment
        env_frame = env.render('rgb_array')
        
        # Update and get plot image
        plot_img = plotter.update(steering, acceleration, reward.item())
        
        # Get debug visualization if enabled
        if record_debug and observer is not None:
            debug_frame = capture_debug_plot(observer)
            if debug_frame is not None:
                debug_frames.append(debug_frame)
        
        # Combine plot and environment frame side by side with right alignment
        plot_height = plot_img.shape[0]
        env_height = env_frame.shape[0]
        main_height = max(plot_height, env_height)
        
        # Calculate widths and padding
        plot_width = plot_img.shape[1]
        env_width = env_frame.shape[1]
        combined_width = plot_width + env_width

        # Create the combined frame
        combined_frame = np.zeros((main_height, combined_width, 3), dtype=np.uint8)
        
        # Add plot to the left side
        y_offset = (main_height - plot_height) // 2
        combined_frame[y_offset:y_offset + plot_height, :plot_width] = plot_img
        
        # Add environment frame to the right side - aligned to the right edge
        y_offset = (main_height - env_height) // 2
        combined_frame[y_offset:y_offset + env_height, -env_width:] = env_frame
        
        frames.append(combined_frame)
    
    # Clean up
    plt.close(plotter.fig)
    
    # Determine if episode was successful
    success = info[0].get('termination_reason') == 'ReachedEnd'
    
    return Episode(
        frames=frames,
        debug_frames=debug_frames,
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
    
    # Get recording configuration
    record_debug = cfg.debug_mode
    
    # Initialize episode manager
    episode_manager = EpisodeManager(output_dir, cfg.recording.num_episodes_to_keep)
    signal_handler.episode_manager = episode_manager
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Recording episodes... Press Ctrl+C to exit")
    print(f"Videos will be saved to: {output_dir}")
    print(f"Debug visualization recording: {'enabled' if record_debug else 'disabled'}")
    
    try:
        while True:  # Run indefinitely until interrupted
            episode = record_episode(env, model, record_debug)
            if episode_manager.add_episode(episode):
                print(f"\nNew episode recorded! Score: {episode.get_score():.2f}")
                episode_manager.print_stats()
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
        episode_manager.print_stats()
    finally:
        env.close()

if __name__ == "__main__":
    main()