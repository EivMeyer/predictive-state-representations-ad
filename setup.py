# setup.py

import logging
import torch
import yaml
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from commonroad_geometric.learning.reinforcement.constants import COMMONROAD_GYM_ENV_ID

# Load configuration from YAML
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Register the CommonRoad environment
register(
    id=COMMONROAD_GYM_ENV_ID,
    entry_point='commonroad_geometric.learning.reinforcement.commonroad_gym_env:CommonRoadGymEnv',
)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the random seed for reproducibility
torch.manual_seed(config["seed"])

# Determine the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() and config["device"] == "auto" else "cpu")
logger.info(f"Using device: {device}")
