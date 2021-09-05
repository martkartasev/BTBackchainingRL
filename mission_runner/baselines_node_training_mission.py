from mission_runner.abstract_mission import AbstractMission
from observation.observation import Observation


class BaselinesNodeTrainingMission(AbstractMission):

    def run_mission(self):
        observations, reward = self.agent.get_next_observations_and_reward()
        observation = Observation(observations)
        self.agent.set_observation(observation)
        self.agent.set_rewards(reward)

    def run(self):
        pass
