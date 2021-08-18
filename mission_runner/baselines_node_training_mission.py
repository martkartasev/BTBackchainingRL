import json
from collections import namedtuple

from mission_runner.abstract_mission import AbstractMission
from observation import Observation


class BaselinesNodeTrainingMission(AbstractMission):

    def run_mission(self):
        world_state = self.agent.get_next_world_state()
        observation = Observation(world_state.observations)
        self.agent.set_observation(observation)

        for error in world_state.errors:
            print("Error:", error.text)

        self.agent.store_observations(observation, world_state.rewards)
        self.agent.control_loop()

    def run(self):
        pass
