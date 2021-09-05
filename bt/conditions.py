from py_trees.behaviour import Behaviour
from py_trees.common import Status

from observation import game_objects


class Condition(Behaviour):
    def __init__(self, name, agent=None):
        super(Condition, self).__init__(name)
        self.agent = agent


class IsNotInFire(Condition):
    def __init__(self, agent):
        super(IsNotInFire, self).__init__(f"Is not in fire", agent)

    def update(self):
        grid_list = self.agent.observation.vector[self.agent.observation.surroundings_list_index:]
        return Status.SUCCESS if grid_list[0] != (game_objects.index("fire") + 1) else Status.FAILURE


class IsSkeletonDefeated(Condition):
    def __init__(self, agent):
        super(IsSkeletonDefeated, self).__init__(f"Is skeleton dead", agent)

    def update(self):
        skeleton_life = self.agent.observation.vector[self.agent.observation.skeleton_life_index]
        return Status.SUCCESS if skeleton_life == 0 else Status.FAILURE


# TODO: Should we use hunger here? Discuss with Mart
class IsNotHungry(Condition):
    def __init__(self, agent):
        super(IsNotHungry, self).__init__(f"Is not hungry", agent)

    def update(self):
        return Status.FAILURE


# TODO: Here we need to integrate inventory to our observation
class HasBeef(Condition):
    def __init__(self, agent):
        super(HasBeef, self).__init__(f"Has beef", agent)

    def update(self):
        return Status.FAILURE


# TODO: Implement
class IsBeefOnGround(Condition):
    def __init__(self, agent):
        super(IsBeefOnGround, self).__init__(f"Is beef on ground", agent)

    def update(self):
        return Status.FAILURE
