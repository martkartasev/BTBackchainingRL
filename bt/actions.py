from py_trees.behaviour import Behaviour
from py_trees.common import Status


class Action(Behaviour):
    def __init__(self, name):
        super(Action, self).__init__(name)


class AvoidFire(Action):
    def __init__(self, agent):
        super(AvoidFire, self).__init__(f"Avoid Fire")
        self.agent = agent

    def update(self):
        # TODO: Implement this
        return Status.SUCCESS
