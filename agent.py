import time

from MalmoPython import AgentHost

from observation import Observation


class BaseAgent:
    def __init__(self):
        self.agent_host = AgentHost()
        self.observation = None
        self.rewards = None

    def start_mission(self, mission, mission_record):
        self.agent_host.startMission(mission, mission_record)

    def start_mission_with_pool(self, mission, pool, mission_record, experiment_id):
        self.agent_host.startMission(mission, pool, mission_record, 0, experiment_id)

    def get_world_state(self):
        return self.agent_host.getWorldState()

    def set_observation(self, observation):
        self.observation = observation

    def set_rewards(self, rewards):
        self.rewards = rewards

    def get_next_world_state(self):
        observations = None
        world_state = None
        while observations is None or len(observations) == 0:
            world_state = self.get_world_state()
            observations = world_state.observations
        return world_state

    def activate_night_vision(self):
        self.agent_host.sendCommand(f"chat /effect @p night_vision 99999 255")



class MalmoAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def continuous_move(self, val):
        self.agent_host.sendCommand("move " + str(val))

    def continuous_turn(self, val):
        self.agent_host.sendCommand("turn " + str(val))

    def set_yaw(self, val):
        self.agent_host.sendCommand("setYaw " + str(val))

    def continuous_strafe(self, val):
        self.agent_host.sendCommand("strafe " + str(val))

    def continuous_pitch(self, val):
        self.agent_host.sendCommand("pitch " + str(val))

    def set_pitch(self, val):
        self.agent_host.sendCommand("setPitch " + str(val))

    def continuous_jump(self, toggle):
        self.agent_host.sendCommand("jump " + str(toggle))

    def attack(self, toggle):
        self.agent_host.sendCommand("attack " + str(toggle))

    def crouch(self, toggle):
        self.agent_host.sendCommand("crouch " + str(toggle))

    def use(self, toggle=None):
        if toggle is None:
            self.agent_host.sendCommand("use")
            time.sleep(0.5)  # Stupid but necessary
        elif toggle == 1:
            self.agent_host.sendCommand("use " + str(toggle))
            self.agent_host.sendCommand("use 0")
            time.sleep(0.001)
            self.agent_host.sendCommand("use 0")
            time.sleep(0.5)  # Stupid but necessary

    def quit(self):
        self.agent_host.sendCommand("quit")

    def reset_agent(self):
        raise NotImplementedError()

    def is_mission_over(self):
        raise NotImplementedError()

    def control_loop(self):
        raise NotImplementedError()

    def store_observations(self, observations):
        raise NotImplementedError()


class ObservationAgent(MalmoAgent):

    def __init__(self):
        super(ObservationAgent, self).__init__()
        self.rewards = None
        self.tree = None
        self.latestObservations = None
        self.previousObservations = None
        self.observations = None
        self.index = 0

    def store_observations(self, observation, rewards=None):
        if observation is not None:
            self.index = self.index + 1
            observation.index = self.index
            if rewards is not None:
                self.rewards = rewards
            self.latestObservations = observation

    def next_observations(self):
        if self.latestObservations is None:
            return False

        if self.latestObservations is not None and self.observations is None:
            self.previousObservations = self.observations
            self.observations = self.latestObservations
            return True

        while self.observations.index >= self.latestObservations.index:
            pass

        self.previousObservations = self.observations
        self.observations = self.latestObservations
        return True

    def is_agent_alive(self):
        return self.observations.vector[6] > 0
