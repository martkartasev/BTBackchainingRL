from bt.accs import find_accs
from bt.ppa import PPA
from bt.sequence import Sequence
from learning.baseline_node import PPABaselinesNode


class BackChainTree:
    def __init__(self, agent, goals, evaluation_manager=None):
        self.agent = agent
        self.baseline_nodes = []
        self.root = self.get_base_back_chain_tree(goals, evaluation_manager)

    def get_base_back_chain_tree(self, goals, evaluation_manager=None):
        if len(goals) == 1:
            back_chain_tree = self.back_chain_recursive(self.agent, goals[0])
        else:
            goal_ppa_trees = []
            for goal in goals:
                goal_ppa_trees.append(self.back_chain_recursive(self.agent, goal, evaluation_manager))
            back_chain_tree = Sequence("Back-chain Tree", children=goal_ppa_trees)

        if back_chain_tree is None:
            return None
        back_chain_tree.setup_with_descendants()

        for baseline_node in self.baseline_nodes:
            accs = find_accs(baseline_node)
            if evaluation_manager is not None:
                [acc.set_evaluation_manager(evaluation_manager) for acc in accs]
            baseline_node.accs = accs
        return back_chain_tree

    def back_chain_recursive(self, agent, condition, evaluation_manager=None):
        ppa = self.condition_to_ppa_tree(agent, condition)
        if evaluation_manager is not None:
            ppa.post_condition.set_evaluation_manager(evaluation_manager)
            [precond.set_evaluation_manager(evaluation_manager) for precond in ppa.pre_conditions]

        if ppa is not None and isinstance(ppa.action, PPABaselinesNode):
            self.baseline_nodes.append(ppa.action)
            ppa.action.post_conditions.append(ppa.post_condition)

        if ppa is not None:
            for i, pre_condition in enumerate(ppa.pre_conditions):
                ppa_condition_tree = self.back_chain_recursive(agent, ppa.pre_conditions[i])
                if ppa_condition_tree is not None:
                    ppa.pre_conditions[i] = ppa_condition_tree
            return ppa.as_tree()
        return condition

    def condition_to_ppa_tree(self, agent, condition):
        for ppa in [sub(agent) for sub in PPA.__subclasses__()]:
            if isinstance(ppa.post_condition, type(condition)):
                return ppa

        return None
