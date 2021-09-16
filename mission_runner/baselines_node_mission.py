from mission_runner.abstract_mission import AbstractMission


class BaselinesNodeMission(AbstractMission):

    def __init__(self, agent, filename=None, hard_reset=True):
        super().__init__(agent, filename)
        self.hard_reset = hard_reset

    def soft_reset(self):
        self.agent.go_to_spawn()

        self.agent.destroy_all_entities()
        self.agent.wait_for_entity(False)

        self.agent.create_static_skeleton()
        self.agent.create_cow()
        self.agent.wait_for_entity(True)
