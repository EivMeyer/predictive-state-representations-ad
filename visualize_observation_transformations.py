import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import map_coordinates
from pathlib import Path
from omegaconf import DictConfig
from utils.config_utils import config_wrapper
from environments import get_environment

from plotting_setup import setup_plotting
setup_plotting()

prev_observation, prev_polar_observation = None, None

def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar_transform(image):
    h, w = image.shape[:2]
    center = (h // 2, w // 2)
    max_radius = min(center)

    theta, r = np.meshgrid(np.linspace(0, 2*np.pi, w), np.linspace(0, max_radius, h))
    x = r * np.cos(theta) + center[1]
    y = r * np.sin(theta) + center[0]

    polar_img = np.zeros_like(image)
    for channel in range(image.shape[2]):
        polar_img[:, :, channel] = map_coordinates(image[:, :, channel], [y, x], order=1, mode='nearest')

    return polar_img

def create_polar_plot(ax, polar_image, is_circular=True):
    h, w = polar_image.shape[:2]
    if is_circular:
        theta = np.linspace(0, 2*np.pi, w)
        r = np.linspace(0, 1, h)
        T, R = np.meshgrid(theta, r)
        ax.pcolormesh(T, R, polar_image, shading='auto')
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        # ax.set_theta_offset(np.deg2rad(-80))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
    else:
        ax.imshow(polar_image)

def calculate_flow(current_frame, next_frame):
    return next_frame.astype(float) - current_frame.astype(float)

def create_figure():
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, wspace=0.3, hspace=0.3)

    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    axes[1] = fig.add_subplot(gs[0, 1], projection='polar')
    axes[4] = fig.add_subplot(gs[1, 1], projection='polar')

    titles = ['Cartesian Image', 'Polar Image (Circular)', 'Polar Image (Rectangular)',
              'Cartesian Flow', 'Polar Flow (Circular)', 'Polar Flow (Rectangular)']

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        if ax.name != 'polar':
            ax.axis('off')

    return fig, axes

def update_figure(frame_num, env, axes, save=False):
    global prev_observation, prev_polar_observation
    
    if frame_num == 0:
        obs = env.reset()
    else:
        action = env.action_space.sample()
        obs, _, done, _ = env.step([action])
        if done:
            obs = env.reset()

    observation = obs[0]
    polar_observation = polar_transform(observation)

    if not save or prev_observation is None:
        prev_observation = observation
        prev_polar_observation = polar_observation
        return axes

    if frame_num > 0:
        flow = calculate_flow(prev_observation, observation)
        polar_flow = calculate_flow(prev_polar_observation, polar_observation)
    else:
        flow = np.zeros_like(observation)
        polar_flow = np.zeros_like(polar_observation)

    images = [
        observation,
        polar_observation,
        polar_observation,
        np.sum(np.abs(flow), axis=2),
        np.sum(np.abs(polar_flow), axis=2),
        np.sum(np.abs(polar_flow), axis=2)
    ]

    for ax, img in zip(axes, images):
        ax.clear()
        if ax.name == 'polar':
            create_polar_plot(ax, img, is_circular=True)
        else:
            ax.imshow(img)
            ax.axis('off')

    prev_observation = observation
    prev_polar_observation = polar_observation

    return axes

@config_wrapper()
def main(cfg: DictConfig):
    env_class = get_environment(cfg.environment)
    env = env_class().make_env(cfg, n_envs=1, seed=cfg.seed)

    fig, axes = create_figure()

    if cfg.run_mode == 'single':
        output_dir = Path('./output')
        output_dir.mkdir(exist_ok=True)

        for frame in range(0, 301, 1):  # 0, 30, 60, ..., 300
            save_fig = frame % 30 == 0
            update_figure(frame, env, axes, save=save_fig)
            
            if save_fig:    
                # Save full figure
                plt.savefig(output_dir / f'polar_transform_frame_{frame:03d}.pdf', format='pdf', dpi=100, bbox_inches='tight')
                
                # Save individual subfigures
                for i, ax in enumerate(axes):
                    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    fig.savefig(output_dir / f'subfig_{i+1}_frame_{frame:03d}.pdf', bbox_inches=extent.expanded(1.1, 1.1))
        
                print(f"Figures saved in '{output_dir}'")
    elif cfg.run_mode == 'animate':
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, update_figure, fargs=(env, axes), 
                             frames=200, interval=50, blit=False)
        plt.show()
    else:
        print("Invalid run_mode. Choose 'single' or 'animate'.")

    plt.close()

if __name__ == "__main__":
    main()