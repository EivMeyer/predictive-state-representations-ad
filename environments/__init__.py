from environments.commonroad_env.commonroad_env import CommonRoadEnv
from environments.image_cartpole_env import VectorizedImageCartPoleEnv
from environments.cartpole_env import CartPoleEnv
from environments.mountain_car_env import MountainCarEnv
 # todo: automatically import all environments

def get_environment(env_type: str):
    if env_type == "commonroad":
        return CommonRoadEnv
    elif env_type == "image_cartpole":
        return VectorizedImageCartPoleEnv
    elif env_type == "cartpole":
        return CartPoleEnv
    elif env_type == "mountain_car":
        return MountainCarEnv
    else:
        raise ValueError(f"Unknown environment type: {env_type}")