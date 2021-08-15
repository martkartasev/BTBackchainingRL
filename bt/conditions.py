from py_trees.behaviour import Behaviour
from py_trees.common import Status


class Condition(Behaviour):
    def __init__(self, name):
        super(Condition, self).__init__(name)


class IsNotInFire(Condition):
    def __init__(self, agent):
        super(IsNotInFire, self).__init__(f"Is not in fire")
        self.agent = agent

    def update(self):
        # TODO: Implement this
        return Status.SUCCESS
