import time

import numpy as np
from malmo.MalmoPython import AgentHost


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

    def get_next_observations_and_reward(self):
        observations = None
        reward = 0
        while observations is None or len(observations) == 0:
            world_state = self.get_world_state()
            observations = world_state.observations
            reward += sum(reward.getValue() for reward in world_state.rewards)
        return observations, reward

    def activate_night_vision(self):
        self.agent_host.sendCommand("chat /effect @p night_vision 99999 255")

    def set_fire_eternal(self):
        self.agent_host.sendCommand("chat /gamerule doFireTick false")

    def make_hungry(self):
        self.agent_host.sendCommand("chat /effect @p hunger 5 255")

    def create_static_skeleton(self):
        self.agent_host.sendCommand("chat /summon skeleton -14 4 0 {NoAI:1}")

    def create_cow(self):
        self.agent_host.sendCommand("chat /summon cow 14 4 0 {NoAI:1}")

    def destroy_all_entities(self):
        self.agent_host.sendCommand("chat /kill @e[type=skeleton]")
        self.agent_host.sendCommand("chat /kill @e[type=cow]")
        self.agent_host.sendCommand("chat /kill @e[type=item]")  # Do this last to remove drops from the mobs

    def go_to_spawn(self):
        self.agent_host.sendCommand("chat /tp 0 4 0")


class MalmoAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def move_towards_flat_direction(self, wanted_flat_direction_vector):
        direction_vector = self.observation.dict["direction"]
        flat_direction_vector = np.array([direction_vector[0], direction_vector[2]])
        flat_direction_vector /= np.linalg.norm(flat_direction_vector)
        side_direction_vector = np.array([flat_direction_vector[1], -flat_direction_vector[0]])

        angle = np.arccos(np.dot(wanted_flat_direction_vector, flat_direction_vector))
        sign = 1 if np.dot(wanted_flat_direction_vector, side_direction_vector) > 0 else -1

        self.continuous_move(np.cos(angle))
        self.continuous_strafe(-sign * np.sin(angle))

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

    def swap_items(self, position1, position2):
        self.agent_host.sendCommand("swapInventoryItems {0} {1}".format(position1, position2))

    def select_on_hotbar(self, position):
        self.agent_host.sendCommand(f"hotbar.{position + 1} 1")  # press
        self.agent_host.sendCommand(f"hotbar.{position + 1} 0")  # release
        time.sleep(0.1)  # Stupid but necessary

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

    def continuous_use(self, toggle):
        self.agent_host.sendCommand("use " + str(toggle))

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
        self.tree = None
        self.latestObservations = None
        self.previousObservations = None
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

        if self.observation is None:
            self.previousObservations = self.observation
            self.observation = self.latestObservations
            return True

        while self.observation.index >= self.latestObservations.index:
            pass

        self.previousObservations = self.observation
        self.observation = self.latestObservations
        return True

    def is_agent_alive(self):
        return self.observation.dict["health"] > 0
