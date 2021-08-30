from py_trees.composites import Selector

from bt import conditions, actions
from bt.sequence import Sequence
from learning.baseline_node import DefeatSkeleton


class PPA:

    def __init__(self):
        self.name = ""
        self.post_condition = None
        self.pre_conditions = []
        self.action = None

    def as_tree(self):
        if self.action is None:
            return None

        if len(self.pre_conditions) > 0:
            sub_tree = Sequence(
                name=f"Precondition Handler {self.name}",
                children=self.pre_conditions + [self.action]
            )
        else:
            sub_tree = self.action
        if self.post_condition is not None:
            tree = Selector(
                name=f"Postcondition Handler {self.name}",
                children=[self.post_condition, sub_tree]
            )
        else:
            tree = sub_tree

        tree.name = f"PPA {self.name}"

        return tree


class AvoidFirePPA(PPA):
    def __init__(self, agent):
        super(AvoidFirePPA, self).__init__()
        self.name = f"Avoid Fire"
        self.post_condition = conditions.IsNotInFire(agent)
        self.pre_conditions = []
        self.action = actions.AvoidFire(agent)


class DefeatSkeletonPPA(PPA):
    def __init__(self, agent):
        super(DefeatSkeletonPPA, self).__init__()
        self.name = f"Avoid Fire"
        self.post_condition = conditions.IsSkeletonDefeated(agent)
        self.pre_conditions = [conditions.IsNotInFire(agent)]
        self.action = DefeatSkeleton(agent)
