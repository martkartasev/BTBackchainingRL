import gym

EP_MAX_TIME_STEPS = 15000


class BaselinesNodeTrainingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, learning_node, mission, hard_reset=True):
        self.node = learning_node
        self.agent = self.node.agent
        self.mission = mission
        self.hard_reset = hard_reset

        self.is_acc_violated = False

        self.action_space = gym.spaces.Discrete(len(self.node.children))
        self.observation_space = self.node.get_observation_space()
        self.steps = 0

    def step(self, action):
        self.steps += 1
        self.node.set_tick_child(action)
        self.node.tick_once()

        self.mission.tick_mission()

        reward = self.node.calculate_rewards()
        ob = self.node.get_observation_array()

        is_mission_over = self.agent.is_mission_over()
        self.is_acc_violated = self.node.is_acc_violated()
        is_post_conditions_fulfilled = self.node.is_post_conditions_fulfilled()
        is_timed_out = self.steps > EP_MAX_TIME_STEPS

        done = is_mission_over or self.is_acc_violated or is_timed_out or is_post_conditions_fulfilled

        return ob, reward, done, {}

    def close(self):
        self.agent.quit()

    def reset(self):
        self.steps = 0
        if self.is_acc_violated:
            while self.node.is_acc_violated() and not self.agent.is_mission_over():
                self.agent.control_loop()
                self.mission.tick_mission()
            self.is_acc_violated = False
            if self.agent.is_mission_over():
                self.restart_mission()
        else:
            self.restart_mission()
        self.node.reset_node()
        return self.node.get_observation_array()

    def restart_mission(self):
        self.steps = 0
        self.mission.reset()

        return self.node.get_observation_array()

    def render(self, mode='human', close=False):
        """ Minecraft is started separately. """
        raise NotImplementedError()
