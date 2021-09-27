import gym

EP_MAX_TIME_STEPS = 15000


class BaselinesNodeTrainingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, learning_node, mission, hard_reset=True):
        self.node = learning_node
        self.agent = self.node.agent
        self.mission = mission
        self.hard_reset = True
        self.constraint_violated = False

        self.action_space = gym.spaces.Discrete(len(self.node.children))
        self.observation_space = self.node.get_observation_space()
        self.steps = 0

    def step(self, action):
        self.steps += 1
        self.node.set_tick_child(action)
        self.node.tick_once()

        self.mission.tick_mission()

        self.constraint_violated = self.node.constraints is not None and False in self.check_constraints()
        is_timed_out = self.steps > EP_MAX_TIME_STEPS

        reward = self.node.calculate_rewards()
        ob = self.node.get_observation_array()
        done = self.constraint_violated or self.agent.is_mission_over() or is_timed_out
        return ob, reward, done, {}

    def check_constraints(self):
        return [constraint.evaluate(self.agent) for constraint in self.node.constraints]

    def close(self):
        self.agent.quit()
    #TODO: More melding of code here
    def reset(self):
        self.steps = 0
        if self.constraint_violated:
            while False in self.check_constraints() and not self.agent.is_mission_over():
                print("moving out")
                self.agent.control_loop()
                self.mission.run_mission()
            self.constraint_violated = False
            if self.agent.is_mission_over():
                self.restart_mission()
        else:
            self.restart_mission()

        return self.node.get_observation_array()

    def restart_mission(self):
        if self.agent.getWorldState().is_mission_running:
            self.agent.sendCommand("quit")
        self.mission.mission_initialization()
        self.mission.run_mission()

    def render(self, mode='human', close=False):
        """ Minecraft is started separately. """
        raise NotImplementedError()
