import numpy as np

from agent import MalmoAgent


class BaselinesNodeAgent(MalmoAgent):

    def __init__(self, agent_host, observation_manager, name="execution/node_skeleton"):
        super().__init__(agent_host)
        self.name = name
        self.observation_manager = observation_manager
        self.tree = None

    def move_towards_flat_direction(self, wanted_flat_direction_vector):
        self.continuous_move(wanted_flat_direction_vector[0])
        self.continuous_strafe(wanted_flat_direction_vector[1])

    def reset(self):
        self.observation_manager.reset()

    def is_mission_over(self):
        return not self.observation_manager.is_agent_alive()

    def control_loop(self):
        self.tree.tick_once()
