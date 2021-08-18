import json
import time
from collections import namedtuple

from mission_runner.abstract_mission import AbstractMission


class NormalMission(AbstractMission):

    def run_mission(self):
        world_state = self.agent.getWorldState()
        while world_state.is_mission_running:
            observations = None
            while world_state.is_mission_running and observations is None:
                if len(world_state.observations) is not 0:
                    observations = json.loads(world_state.observations[0].text, object_hook=lambda d: namedtuple('Observations', d.keys())(*d.values()))
                world_state = self.agent.getWorldState()

            self.agent.store_observations(observations, world_state.rewards)
            self.agent.next_observations()

            for error in world_state.errors:
                print("Error:", error.text)
            if self.agent.is_mission_over():
                self.agent.sendCommand("quit")
                break

            self.agent.control_loop()

    def run(self):
        self.mission_initialization()

        start = time.time()
        self.run_mission()
        end = time.time()

        print("took " + str((end - start) * 1000) + ' milliseconds')
        print("Mission ended")
