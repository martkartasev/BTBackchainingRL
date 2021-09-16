from bt import conditions
from bt.accs import find_accs
from bt.ppa import AvoidFirePPA, DefeatSkeletonPPA, EatPPA, PickupBeefPPA, DefeatCowPPA, IsNotAttackedByEnemyPPA, \
    ChaseEntityPPA
from bt.sequence import Sequence
from learning.baseline_node import DynamicBaselinesNode


class BackChainTree:
    def __init__(self, agent, goals):
        self.agent = agent
        self.baseline_nodes = []
        self.root = self.get_base_back_chain_tree(goals)

    def get_base_back_chain_tree(self, goals):
        if len(goals) == 1:
            back_chain_tree = self.back_chain_recursive(self.agent, goals[0])
        else:
            goal_ppa_trees = []
            for goal in goals:
                goal_ppa_trees.append(self.back_chain_recursive(self.agent, goal))
            back_chain_tree = Sequence("Back-chain Tree", children=goal_ppa_trees)

        if back_chain_tree is None:
            return None
        back_chain_tree.setup_with_descendants()

        for baseline_node in self.baseline_nodes:
            baseline_node.accs = find_accs(baseline_node)
        return back_chain_tree

    def back_chain_recursive(self, agent, condition):
        ppa = self.condition_to_ppa_tree(agent, condition)
        if ppa is not None:
            for i, pre_condition in enumerate(ppa.pre_conditions):
                ppa_condition_tree = self.back_chain_recursive(agent, ppa.pre_conditions[i])
                if ppa_condition_tree is not None:
                    ppa.pre_conditions[i] = ppa_condition_tree
            return ppa.as_tree()
        return condition

    def condition_to_ppa_tree(self, agent, condition):
        ppa = None
        if isinstance(condition, conditions.IsNotInFire):
            ppa = AvoidFirePPA(agent)
        elif isinstance(condition, conditions.IsEnemyDefeated):
            ppa = DefeatSkeletonPPA(agent)
        elif isinstance(condition, conditions.IsNotHungry):
            ppa = EatPPA(agent)
        elif isinstance(condition, conditions.HasFood):
            ppa = PickupBeefPPA(agent)
        elif isinstance(condition, conditions.IsEntityPickable):
            ppa = DefeatCowPPA(agent)
        elif isinstance(condition, conditions.IsNotAttackedByEnemy):
            ppa = IsNotAttackedByEnemyPPA(agent)
        elif isinstance(condition, conditions.IsCloseToEntity):
            ppa = ChaseEntityPPA(agent)

        if ppa is not None and isinstance(ppa.action, DynamicBaselinesNode):
            self.baseline_nodes.append(ppa.action)
            ppa.action.post_conditions.append(ppa.post_condition)
        return ppa
