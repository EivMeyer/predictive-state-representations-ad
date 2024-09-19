from environments.commonroad_env.commonroad_env import CommonRoadEnv
from environments.image_cartpole_env import VectorizedImageCartPoleEnv

def get_environment(env_type: str):
    if env_type == "commonroad":
        return CommonRoadEnv
    elif env_type == "image_cartpole":
        return VectorizedImageCartPoleEnv
    else:
        raise ValueError(f"Unknown environment type: {env_type}")