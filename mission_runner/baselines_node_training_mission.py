from mission_runner.abstract_mission import AbstractMission
from observation import Observation


class BaselinesNodeTrainingMission(AbstractMission):

    def run_mission(self):
        world_state = self.agent.get_next_world_state()
        observation = Observation(world_state.observations)
        self.agent.set_observation(observation)
        self.agent.set_rewards(world_state.rewards)

        for error in world_state.errors:
            print("Error:", error.text)

    def run(self):
        pass
