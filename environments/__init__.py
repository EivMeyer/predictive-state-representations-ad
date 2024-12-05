from environments.commonroad_env.commonroad_env import CommonRoadEnv
from environments.image_cartpole_env import VectorizedImageCartPoleEnv
from environments.cartpole_env import CartPoleEnv
 # todo: automatically import all environments
 
def get_environment(env_type: str):
    if env_type == "commonroad":
        return CommonRoadEnv
    elif env_type == "image_cartpole":
        return VectorizedImageCartPoleEnv
    elif env_type == "cartpole":
        return CartPoleEnv
    else:
        raise ValueError(f"Unknown environment type: {env_type}")