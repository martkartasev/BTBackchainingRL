from mission_runner.baselines_node_mission import BaselinesNodeMission


class BaselinesNodeTrainingMission(BaselinesNodeMission):

    def __init__(self, agent, tree, filename=None, hard_reset=True):
        super().__init__(agent, tree, filename, hard_reset)

    def run_mission(self):
        self.agent.update_observations_and_reward()
