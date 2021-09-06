from stable_baselines3 import A2C

from agent import ObservationAgent
from bt import conditions
from bt.back_chain_tree import BackChainTree
from utils.file import get_project_root


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
