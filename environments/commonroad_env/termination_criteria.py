from commonroad_geometric.learning.reinforcement.termination_criteria.implementations import *
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from typing import Tuple, Optional, Set


def has_reached_end(simulation: EgoVehicleSimulation, arclength_threshold: float) -> bool:
    ego_position = simulation.ego_vehicle.state.position
    ego_trajectory_polyline = simulation.ego_vehicle.ego_route.planning_problem_path_polyline
    arclength = ego_trajectory_polyline.get_projected_arclength(
        ego_position,
        relative=True,
        linear_projection=True
    )
    reached_end = arclength >= arclength_threshold
    return reached_end



class CustomReachedEndCriterion(BaseTerminationCriterion):
    def __init__(self, arclength_threshold: float):
        self.arclength_threshold = arclength_threshold
        super().__init__()

    def __call__(
        self,
        simulation: EgoVehicleSimulation
    ) -> Tuple[bool, Optional[str]]:
        reached_end = has_reached_end(simulation, self.arclength_threshold)
        return reached_end, 'ReachedEnd' if reached_end else None

    @property
    def reasons(self) -> Set[str]:
        return {'ReachedEnd'}
    

def create_termination_criteria():
    termination_criteria = [
        CollisionCriterion(),
        OffroadCriterion(),
        # ReachedGoalCriterion(),
        # OvershotGoalCriterion(),
        TimeoutCriterion(max_timesteps=500),
        CustomReachedEndCriterion(arclength_threshold=1.0)
    ]
    return termination_criteria