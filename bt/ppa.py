from py_trees.composites import Selector

from bt import conditions, actions
from bt.sequence import Sequence
from learning.baseline_node import DefeatSkeleton, DefeatCow, ChaseEntity


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
        self.name = "Avoid Fire"
        self.post_condition = conditions.IsNotInFire(agent)
        self.pre_conditions = []
        self.action = actions.AvoidFire(agent)


class DefeatSkeletonPPA(PPA):
    def __init__(self, agent):
        super(DefeatSkeletonPPA, self).__init__()
        self.name = "Defeat Skeleton"
        self.post_condition = conditions.IsEnemyDefeated(agent)
        self.pre_conditions = [conditions.IsNotInFire(agent)]
        self.action = actions.ActionPlaceholder(agent)


"""
TODO: We don't really support choosing which baseline node to train so it will always take the first.
"""
class IsNotAttackedByEnemyPPA(PPA):
    def __init__(self, agent):
        super(IsNotAttackedByEnemyPPA, self).__init__()
        self.name = "Is not Attacked by Enemy PPA"
        self.post_condition = conditions.IsNotAttackedByEnemy(agent)
        self.pre_conditions = []
        self.action = actions.ActionPlaceholder(agent)


class ChaseEntityPPA(PPA):
    def __init__(self, agent):
        super(ChaseEntityPPA, self).__init__()
        self.name = "Is not Attacked by Enemy PPA"
        self.post_condition = conditions.IsCloseToEntity(agent)
        self.pre_conditions = [conditions.IsNotAttackedByEnemy(agent), conditions.IsNotInFire(agent)]
        self.action = ChaseEntity(agent)


class EatPPA(PPA):
    def __init__(self, agent):
        super(EatPPA, self).__init__()
        self.name = "Eat"
        self.post_condition = conditions.IsNotHungry(agent)
        self.pre_conditions = [conditions.HasFood(agent)]
        self.action = actions.Eat(agent)


class PickupBeefPPA(PPA):
    def __init__(self, agent):
        super(PickupBeefPPA, self).__init__()
        self.name = "Pick up"
        self.post_condition = conditions.HasFood(agent)
        self.pre_conditions = [conditions.IsEntityPickable(agent)]
        self.action = actions.PickUpEntity(agent)


class DefeatCowPPA(PPA):
    def __init__(self, agent):
        super(DefeatCowPPA, self).__init__()
        self.name = "Defeat Cow"
        self.post_condition = conditions.IsEntityPickable(agent)
        self.pre_conditions = [ conditions.IsNotInFire(agent)]
        self.action = DefeatCow(agent)
