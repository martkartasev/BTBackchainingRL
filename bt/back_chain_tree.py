from bt.ppa import PPA, condition_to_ppa_tree, back_chain_recursive
from bt.sequence import Sequence


class BackChainTree:
    def __init__(self, agent, goal):
        self.agent = agent
        self.root = self.get_base_back_chain_tree(goal)

    def get_base_back_chain_tree(self, goal):
        children = []

        goal_ppa_tree = back_chain_recursive(self.agent, goal)
        if goal_ppa_tree is not None:
            goal_ppa_tree.setup_with_descendants()
            return Sequence("BaseTree", children=[goal_ppa_tree])