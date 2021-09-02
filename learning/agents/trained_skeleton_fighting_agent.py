from stable_baselines3 import A2C

from agent import ObservationAgent
from bt import conditions
from bt.back_chain_tree import BackChainTree
from utils.file import get_project_root


class TrainedSkeletonFightingAgent(ObservationAgent):

    def __init__(self, name="execution/manual_skeleton"):
        super().__init__()
        self.name = name
        self.tree = self.define_behavior()

    def reset_agent(self):
        self.observation = None

    def is_mission_over(self):
        return not self.is_agent_alive()

    def control_loop(self):
        self.tree.tick_once()

    def define_behavior(self):
        tree = BackChainTree(self, conditions.IsSkeletonDefeated(self))
        node = tree.baseline_nodes[0]

        fighter_model = A2C.load(get_project_root() / "results/basicfighter3_only_little_fire/best_model_18")
        node.set_model(fighter_model)

        return tree.root
