import numpy as np
from py_trees.behaviour import Behaviour
from py_trees.common import Status

from observation.observation import game_objects, GRID_SIZE_AXIS


class Condition(Behaviour):
    def __init__(self, name, agent=None):
        super(Condition, self).__init__(name)
        self.agent = agent


class IsNotInFire(Condition):
    def __init__(self, agent):
        super(IsNotInFire, self).__init__(f"Is not in fire", agent)

    def update(self):
        grid_list = self.agent.observation.vector[self.agent.observation.surroundings_list_index:]
        me_position_index = GRID_SIZE_AXIS[1] * GRID_SIZE_AXIS[2] + 1
        return Status.SUCCESS if grid_list[me_position_index] != (game_objects.index("fire") + 1) else Status.FAILURE


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


class HasBeef(Condition):
    def __init__(self, agent):
        super(HasBeef, self).__init__(f"Has beef", agent)

    def update(self):
        return self.agent.observation.vector[self.agent.observation.beef_inventory_index_index] > 0


class IsBeefOnGround(Condition):
    def __init__(self, agent):
        super(IsBeefOnGround, self).__init__(f"Is beef on ground", agent)

    def update(self):
        is_beef_on_ground = self.agent.observation.vector[self.agent.observation.is_beef_on_ground_index]
        is_beef_on_ground = is_beef_on_ground == 1
        return Status.SUCCESS if is_beef_on_ground else Status.FAILURE
