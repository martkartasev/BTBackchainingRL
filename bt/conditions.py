import numpy as np
from py_trees.behaviour import Behaviour
from py_trees.common import Status

from observation import game_objects


class Condition(Behaviour):
    def __init__(self, name, agent = None):
        super(Condition, self).__init__(name)
        self.agent = agent


class IsNotInFire(Condition):
    def __init__(self, agent):
        super(IsNotInFire, self).__init__(f"Is not in fire", agent)

    def update(self):
        grid_list = self.agent.observation.vector[7:]
        return Status.SUCCESS if grid_list[0] != game_objects.index("fire") else Status.FAILURE


class IsSkeletonDefeated(Condition):
    def __init__(self, agent):
        super(IsSkeletonDefeated, self).__init__(f"Is skeleton dead", agent)

    def update(self):
        relative_distance = self.agent.observation.vector[0:3]
        return Status.SUCCESS if np.all(relative_distance == np.zeros(3)) else Status.FAILURE

