from mission_runner.baselines_node_mission import BaselinesNodeMission
from observation import Observation


class BaselinesNodeTrainingMission(BaselinesNodeMission):

    def update_observations_and_reward(self):
        observations, reward = self.agent.get_next_observations_and_reward()
        observation = Observation(observations)
        self.agent.set_observation(observation)
        self.agent.set_rewards(reward)
