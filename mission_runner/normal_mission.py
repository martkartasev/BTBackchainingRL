import json
import time
from collections import namedtuple

from mission_runner.abstract_mission import AbstractMission
from observation import Observation


class NormalMission(AbstractMission):

    def run_mission(self):
        world_state = self.agent.get_world_state()
        while world_state.is_mission_running:
            world_state = self.agent.get_next_world_state()
            observation = Observation(world_state.observations)

            self.agent.set_observation(observation)
            self.agent.set_rewards(world_state.rewards)

            for error in world_state.errors:
                print("Error:", error.text)
            if self.agent.is_mission_over():
                self.agent.quit()
                break

            self.agent.control_loop()

    def run(self):
        self.mission_initialization()

        start = time.time()
        self.run_mission()
        end = time.time()

        print("took " + str((end - start) * 1000) + ' milliseconds')
        print("Mission ended")
