import numpy as np
from py_trees.common import Status

from agent import MalmoAgent


class BaselinesNodeAgent(MalmoAgent):

    def __init__(self, agent_host, observation_manager, name="execution/node_skeleton"):
        super().__init__(agent_host)
        self.name = name
        self.observation_manager = observation_manager
        self.tree = None

    def move_towards_flat_direction(self, wanted_flat_direction_vector):
        direction_vector = self.observation_manager.observation.dict["direction"]
        flat_direction_vector = np.array([direction_vector[0], direction_vector[2]])
        flat_direction_vector /= np.linalg.norm(flat_direction_vector)
        side_direction_vector = np.array([flat_direction_vector[1], -flat_direction_vector[0]])

        angle = np.arccos(np.dot(wanted_flat_direction_vector, flat_direction_vector))
        sign = 1 if np.dot(wanted_flat_direction_vector, side_direction_vector) > 0 else -1

        self.continuous_move(np.cos(angle))
        self.continuous_strafe(-sign * np.sin(angle))

    def reset(self):
        self.observation_manager.reset()

    def is_mission_over(self):
        return not self.observation_manager.is_agent_alive() or self.tree.status == Status.SUCCESS or self.tree.status == Status.FAILURE

    def control_loop(self):
        self.tree.tick_once()
