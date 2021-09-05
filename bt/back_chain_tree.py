from bt import conditions
from bt.accs import find_accs
from bt.ppa import AvoidFirePPA, DefeatSkeletonPPA, EatPPA, PickupBeefPPA, DefeatCowPPA
from learning.baseline_node import DynamicBaselinesNode


class BackChainTree:
    def __init__(self, agent, goal):
        self.agent = agent
        self.baseline_nodes = []
        self.root = self.get_base_back_chain_tree(goal)

    def get_base_back_chain_tree(self, goal):
        goal_ppa_tree = self.back_chain_recursive(self.agent, goal)
        if goal_ppa_tree is None:
            return None
        goal_ppa_tree.setup_with_descendants()
        for baseline_node in self.baseline_nodes:
            baseline_node.accs = find_accs(baseline_node)
        return goal_ppa_tree

    def back_chain_recursive(self, agent, condition):
        ppa = self.condition_to_ppa_tree(agent, condition)
        if ppa is not None:
            for i, pre_condition in enumerate(ppa.pre_conditions):
                ppa_condition_tree = self.back_chain_recursive(agent, ppa.pre_conditions[i])
                if ppa_condition_tree is not None:
                    ppa.pre_conditions[i] = ppa_condition_tree
            return ppa.as_tree()
        return None

    def condition_to_ppa_tree(self, agent, condition):
        ppa = None
        if isinstance(condition, conditions.IsNotInFire):
            ppa = AvoidFirePPA(agent)
        elif isinstance(condition, conditions.IsSkeletonDefeated):
            ppa = DefeatSkeletonPPA(agent)
        elif isinstance(condition, conditions.IsNotHungry):
            ppa = EatPPA(agent)
        elif isinstance(condition, conditions.HasBeef):
            ppa = PickupBeefPPA(agent)
        elif isinstance(condition, conditions.IsBeefOnGround):
            ppa = DefeatCowPPA(agent)

        if ppa is not None and isinstance(ppa.action, DynamicBaselinesNode):
            self.baseline_nodes.append(ppa.action)
            ppa.action.post_conditions.append(ppa.post_condition)
        return ppa
