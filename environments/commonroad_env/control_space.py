from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import random
import numpy as np
from gymnasium.spaces import Box, Space

from commonroad.scenario.obstacle import ObstacleType
from commonroad_geometric.simulation.ego_simulation.control_space.base_control_space import BaseControlSpace

if TYPE_CHECKING:
    from commonroad_geometric.simulation.ego_simulation.ego_vehicle_simulation import EgoVehicleSimulation


class TrackVehicleControlSpace(BaseControlSpace):
    """
    Low-level control space for longitudinal and lateral motion planning.
    """

    def __init__(self, options):
        self._selected_vehicle_id = None
        super().__init__(options)

    @property
    def gym_action_space(self) -> Space:
        return Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype="float64"
        )

    def _substep(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
        action: np.ndarray,
        substep_index: int
    ) -> bool:
        vehicles = ego_vehicle_simulation.current_non_ego_obstacles
        vehicles = [v for v in vehicles if v.obstacle_type == ObstacleType.CAR]
        if not vehicles:
            return False
        
        current_vehicle_ids = list(v.obstacle_id for v in vehicles)

        if self._selected_vehicle_id is None or self._selected_vehicle_id not in current_vehicle_ids:
            # Randomly select a vehicle to track
            self._selected_vehicle_id = random.choice(current_vehicle_ids)

        selected_vehicle_idx = current_vehicle_ids.index(self._selected_vehicle_id)

        tracked_vehicle = vehicles[selected_vehicle_idx]
        tracked_vehicle.obstacle_shape._vertices*=0 # This is a hack to prevent the vehicle from being drawn in the simulation
        
        # Select the current state of the tracked vehicle as the next state
        next_state = tracked_vehicle.state_at_time(ego_vehicle_simulation.current_time_step)

        ego_vehicle_simulation.ego_vehicle.set_next_state(next_state)

        return True
    
    def _reset(
        self,
        ego_vehicle_simulation: EgoVehicleSimulation,
    ) -> None:
        self._selected_vehicle_id = None
