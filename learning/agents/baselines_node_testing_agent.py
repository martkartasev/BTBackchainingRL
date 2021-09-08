from agent import ObservationAgent


class BaselinesNodeTestingAgent(ObservationAgent):

    def __init__(self, name="execution/manual_skeleton"):
        super().__init__()
        self.name = name

    def reset_agent(self):
        self.observation = None

    def is_mission_over(self):
        return not self.is_agent_alive()

    def control_loop(self):
        self.tree.tick_once()
