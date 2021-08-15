from py_trees.composites import Selector

from bt import conditions, actions
from bt.sequence import Sequence


def back_chain_recursive(agent, condition):
    ppa = condition_to_ppa_tree(agent, condition)
    if ppa is not None:
        for i, pre_condition in enumerate(ppa.pre_conditions):
            ppa_condition_tree = back_chain_recursive(agent, ppa.pre_conditions[i])
            if ppa_condition_tree is not None:
                ppa.pre_conditions[i] = ppa_condition_tree
        return ppa.as_tree()
    return None


def condition_to_ppa_tree(agent, condition):
    if isinstance(condition, conditions.IsNotInFire):
        return AvoidFirePPA(agent)


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
