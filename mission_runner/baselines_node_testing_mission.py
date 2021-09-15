import time

from mission_runner.abstract_mission import AbstractMission
from observation import Observation


class BaselinesNodeTestingMission(AbstractMission):

    def __init__(self, agent, tree, filename=None):
        super().__init__(agent, filename)
        self.tree = tree

    def run_mission(self):
        world_state = self.agent.get_world_state()
        while world_state.is_mission_running:
            observations, rewards = self.agent.get_next_observations_and_reward()
            observation = Observation(observations)

            self.agent.set_observation(observation)
            self.agent.set_rewards(rewards)

            if self.agent.is_mission_over():
                self.agent.quit()
                break

            self.tree.tick_once()

    def run(self):
        self.mission_initialization()
        self.soft_reset()

        start = time.time()
        self.run_mission()
        end = time.time()

        print("took " + str((end - start) * 1000) + ' milliseconds')
        print("Mission ended")
