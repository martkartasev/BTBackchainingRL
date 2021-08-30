from agent import ObservationAgent


class BaselinesNodeTrainingAgent(ObservationAgent):

    def reset_agent(self):
        return NotImplementedError()

    def is_mission_over(self):
        return NotImplementedError()

    def __init__(self, name="execution/manual_skeleton"):
        super().__init__()
        self.name = name

    def control_loop(self):
        self.next_observations()


class BasicFighterNodeTrainingAgent(BaselinesNodeTrainingAgent):

    def __init__(self, name="execution/node_skeleton"):
        super().__init__()
        self.name = name

    def reset_agent(self):
        return NotImplementedError()

    def is_mission_over(self):
        return not self.is_agent_alive()
