from mission_runner.abstract_mission import AbstractMission
from observation import Observation


class BaselinesNodeMission(AbstractMission):

    def __init__(self, agent, tree, filename=None, hard_reset=True):
        super().__init__(agent, filename)
        self.tree = tree
        self.hard_reset = hard_reset

    def soft_reset(self):
        self.agent.go_to_spawn()
        self.agent.destroy_all_entities()
        self.wait_for_entity(False)
        self.agent.get_next_observations_and_reward()
        self.agent.create_static_skeleton()
        self.agent.create_cow()
        self.wait_for_entity(True)
        self.agent.get_next_observations_and_reward()

    def wait_for_entity(self, expect_entity):
        while True:
            observations, _ = self.agent.get_next_observations_and_reward()
            observation = Observation(observations)
            if expect_entity == observation.dict["entity_visible"]:
                break
