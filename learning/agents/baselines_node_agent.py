from agent import ObservationAgent


class BaselinesNodeAgent(ObservationAgent):

    def __init__(self, observation_filter=None, name="execution/node_skeleton"):
        super().__init__(observation_filter)
        self.name = name

    def reset(self):
        self.observation = None

    def is_mission_over(self):
        return not self.is_agent_alive()

    def control_loop(self):
        self.tree.tick_once()
