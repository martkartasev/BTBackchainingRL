import time

import gym


class BaselinesNodeTrainingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, learning_node, mission):
        self.node = learning_node
        self.agent = self.node.agent
        self.mission = mission

        self.action_space = gym.spaces.Discrete(len(self.node.children))
        self.observation_space = self.node.get_observation_space()

    def step(self, action):
        self.node.set_tick_child(action)
        self.node.tick_once()

        self.mission.run_mission()

        reward = self.node.calculate_rewards()
        ob = self.node.get_observation_array()
        done = self.agent.is_mission_over()
        return ob, reward, done, {}

    def close(self):
        self.agent.quit()

    def reset(self):
        if self.agent.get_world_state().is_mission_running:
            self.agent.quit()

        self.mission.mission_initialization()
        self.mission.run_mission()

        return self.node.get_observation_array()

    def render(self, mode='human', close=False):
        """ Minecraft is started separately. """
        raise NotImplementedError()
