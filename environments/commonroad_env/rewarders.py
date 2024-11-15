from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.implementations import *
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
import numpy as np
import torch
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.learning.reinforcement.termination_criteria.implementations import *
from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.implementations import *
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.base_reward_computer import BaseRewardComputer
from commonroad_geometric.learning.reinforcement.rewarder.reward_computer.types import RewardLossMetric
from commonroad_geometric.learning.reinforcement.observer.base_observer import BaseObserver, T_Observation
from environments.commonroad_env.termination_criteria import has_reached_end
from typing import Optional

class CustomReachedEndRewardComputer(BaseRewardComputer):
    def __init__(self, arclength_threshold: float, reward: float):
        self.arclength_threshold = arclength_threshold
        self.reward = reward
        super().__init__()

    def __call__(
        self,
        action: np.ndarray,
        simulation: EgoVehicleSimulation,
        data: CommonRoadData,
        observation: T_Observation
    ) -> float:
        reached_end = has_reached_end(simulation, self.arclength_threshold)
        if reached_end:
            return self.reward
        return 0.0


class GoalDistanceRewardComputer(BaseRewardComputer):
   def __init__(
       self,
       weight: float = 1.0,
   ) -> None:
       self._weight = weight
       super().__init__()

   def __call__(
       self,
       action: np.ndarray,
       simulation: EgoVehicleSimulation, 
       data: CommonRoadData,
       observation: T_Observation
   ) -> float:
       assert simulation.ego_vehicle.ego_route is not None
       route = simulation.ego_vehicle.ego_route

       if route.planning_problem_path_polyline is None:
           return 0.0

       ego_position = simulation.ego_vehicle.state.position
       goal_position = route.planning_problem_path_polyline.end
       distance = np.linalg.norm(goal_position - ego_position)
           
       reward = -self._weight * distance
       return reward

   def _reset(self) -> None:
       pass


def create_rewarders():
    rewarders = [
        # AccelerationPenaltyRewardComputer(
        #     weight=0.0,
        #     loss_type=RewardLossMetric.L2
        # ),
        CustomReachedEndRewardComputer(
            arclength_threshold=1.0,
            reward=100
        ),
        CollisionPenaltyRewardComputer(
            penalty=-100.0,
        ),
        # FrictionViolationPenaltyRewardComputer(penalty=-0.01),
        TrajectoryProgressionRewardComputer(
            weight=1.0,
            delta_threshold=3,
            relative_arclength=False,
            linear_path_projection=True
        ),
        # ConstantRewardComputer(reward=-0.2),
        GoalDistanceRewardComputer(weight=1/500),
        #
        # ReachedGoalRewardComputer(reward=3.5),
        # OvershotGoalRewardComputer(reward=0.0),
        # SteeringAnglePenaltyRewardComputer(weight=0.0005, loss_type=RewardLossMetric.L1),
        # StillStandingPenaltyRewardComputer(penalty=-0.05, velocity_threshold=2.0),
        # TimeToCollisionPenaltyRewardComputer(weight=0.1), 
        OffroadPenaltyRewardComputer(penalty=-100.0),
        # VelocityPenaltyRewardComputer(
        #     reference_velocity=34.0,
        #     weight=0.2,
        #     loss_type=RewardLossMetric.L2,
        #     only_upper=True
        # ),

        # LateralErrorPenaltyRewardComputer(weight=0.0001, loss_type=RewardLossMetric.L1),
        # YawratePenaltyRewardComputer(weight=0.01),
        # HeadingErrorPenaltyRewardComputer(
        #     weight=0.01,
        #     loss_type=RewardLossMetric.L2,
        #     wrong_direction_penalty=-0.01
        # )
    ]

    return rewarders
