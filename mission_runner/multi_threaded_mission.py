import json
import threading
import time
from collections import namedtuple

from mission_runner.abstract_mission import AbstractMission


class MultiThreadedMission(AbstractMission):

    def run_mission(self):
        world_state = self.agent.getWorldState()
        while world_state.is_mission_running:
            observations = None
            while world_state.is_mission_running and observations is None:
                world_state = self.agent.getWorldState()
                if len(world_state.observations) is not 0:
                    observations = json.loads(world_state.observations[0].text, object_hook=lambda d: namedtuple('Observations', d.keys())(*d.values()))

            for error in world_state.errors:
                print("Error:", error.text)

            if self.agentExited:
                self.agent.sendCommand("quit")
                break

            self.agent.store_observations(observations)

    def run(self):
        self.mission_initialization()

        x = threading.Thread(target=self.run_mission, args=())
        start = time.time()
        x.start()

        while not (self.agent.next_observations() and not self.agent.is_mission_over()):  # Needed because all the internals are otherwise evaluated to old values.
            pass

        while not self.agent.is_mission_over():
            self.agent.control_loop()
            self.agent.next_observations()

        self.agentExited = True
        x.join()
        end = time.time()
        print("took " + str((end - start) * 1000) + ' milliseconds')
        print("Mission ended")
