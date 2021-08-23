from py_trees.behaviour import Behaviour
from py_trees.common import Status

from observation import game_objects


class Condition(Behaviour):
    def __init__(self, name):
        super(Condition, self).__init__(name)


class IsNotInFire(Condition):
    def __init__(self, agent):
        super(IsNotInFire, self).__init__(f"Is not in fire")
        self.agent = agent

    def update(self):
        grid_list = self.agent.observation.vector[7:]

        return Status.FAILURE if grid_list[0] == game_objects.index("fire") else Status.SUCCESS
