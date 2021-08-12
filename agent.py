from MalmoPython import AgentHost


class BaseAgent:
    def __init__(self):
        self.agent_host = AgentHost()

    def start_mission(self, mission, pool, mission_record, experiment_id):
        self.agent_host.startMission(mission, pool, mission_record, 0, experiment_id)

    def get_world_state(self):
        return self.agent_host.getWorldState()

    def get_next_world_state(self):
        observations = None
        world_state = None
        while observations is None or len(observations) == 0:
            world_state = self.get_world_state()
            observations = world_state.observations
        return world_state

    def activate_night_vision(self):
        self.agent_host.sendCommand(f"chat /effect @p night_vision 99999 255")