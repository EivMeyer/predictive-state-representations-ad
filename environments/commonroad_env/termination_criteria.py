from commonroad_geometric.learning.reinforcement.termination_criteria.implementations import *
from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation
from commonroad_geometric.learning.reinforcement.termination_criteria.base_termination_criterion import BaseTerminationCriterion
from typing import Tuple, Optional, Set, List


def has_reached_end(simulation: EgoVehicleSimulation, remaining_dist_threshold: float) -> bool:
    ego_position = simulation.ego_vehicle.state.position
    ego_trajectory_polyline = simulation.ego_vehicle.ego_route.planning_problem_path_polyline
    arclength = ego_trajectory_polyline.get_projected_arclength(
        ego_position,
        relative=False,
        linear_projection=True
    )
    remaining_distance = ego_trajectory_polyline.length - arclength
    reached_end = remaining_distance < remaining_dist_threshold
    return reached_end



class CustomReachedEndCriterion(BaseTerminationCriterion):
    def __init__(self, remaining_dist_threshold: float):
        self.remaining_dist_threshold = remaining_dist_threshold
        super().__init__()

    def __call__(
        self,
        simulation: EgoVehicleSimulation
    ) -> Tuple[bool, Optional[str]]:
        reached_end = has_reached_end(simulation, self.remaining_dist_threshold)
        return reached_end, 'ReachedEnd' if reached_end else None

    @property
    def reasons(self) -> Set[str]:
        return {'ReachedEnd'}
    

def create_termination_criteria(terminate_on_collision: bool) -> List[BaseTerminationCriterion]:
    termination_criteria = [
        TimeoutCriterion(max_timesteps=500),
        # # ReachedGoalCriterion(),
        # # OvershotGoalCriterion(),
        CustomReachedEndCriterion(remaining_dist_threshold=40.0)
    ]
    if terminate_on_collision:
        termination_criteria.append(OffroadCriterion())
        termination_criteria.append(CollisionCriterion())

    print(f"Termination criteria: {termination_criteria}")
        
    return termination_criteria