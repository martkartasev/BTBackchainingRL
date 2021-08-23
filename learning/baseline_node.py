import numpy as np
from gym import spaces
from py_trees.common import Status

from bt.accs import find_accs
from bt.actions import TurnLeft, TurnRight, MoveForward, MoveBackward, Attack
from bt.sequence import Sequence
from observation import Observation


class BaselinesNode(Sequence):
    def __init__(self, agent, name="A2CLearner", children=None, model=None, ):
        self.agent = agent
        self.model = model
        self.tick_aux = self.execution_tick if model is not None else self.training_tick
        self.child_index = -1
        super(BaselinesNode, self).__init__(name=name, children=children)

    def initialise(self):
        pass

    def tick(self):
        return self.tick_aux()

    def training_tick(self):
        self.logger.debug("%s.tick()" % self.__class__.__name__)
        # Required behaviour for *all* behaviours and composites is
        # for tick() to check if it isn't running and initialise
        if self.status != Status.RUNNING:
            # selectors dont do anything specific on initialisation
            #   - the current child is managed by the update, never needs to be 'initialised'
            # run subclass (user) handles
            self.initialise()

        self.update()
        previous = self.current_child
        child = self.children[self.child_index]

        for other in self.children:
            if child != other and other.status == Status.RUNNING:
                other.stop(Status.INVALID)

        for node in child.tick():
            yield node
            if node is child:
                if node.status == Status.RUNNING or node.status == Status.FAILURE:
                    self.current_child = child
                    self.status = node.status
                    if previous is None or previous != self.current_child:
                        # we interrupted, invalidate everything at a lower priority
                        passed = False
                        for child in self.children:
                            if passed:
                                if child.status != Status.INVALID:
                                    child.stop(Status.INVALID)
                            passed = True if child == self.current_child else passed
                    yield self
                    return
        # all children succeded, set succed ourselves and current child to the last bugger who failed us
        self.status = Status.SUCCESS
        try:
            self.current_child = self.children[-1]
        except IndexError:
            self.current_child = None
        yield self

    def execution_tick(self):
        child_index, _ = self.model.predict(self.get_observation_array())
        self.child_index = child_index
        return self.training_tick()

    def set_tick_child(self, child_index):
        self.child_index = child_index

    def get_observation_space(self):
        raise NotImplementedError()

    def get_observation_array(self):
        raise NotImplementedError()

    def calculate_rewards(self):
        raise NotImplementedError()


class DynamicBaselinesNode(BaselinesNode):

    def __init__(self, agent, name="A2CLearner", children=None, model=None, ):
        super(DynamicBaselinesNode, self).__init__(agent, name=name, children=children, model=model)
        self.accs = []
        self.post_conditions = []

    def get_observation_space(self):
        return Observation.get_observation_space()

    def get_observation_array(self):
        return self.agent.observation.vector

    def calculate_rewards(self):
        rewards = 0
        for acc in self.accs:
            res = acc.tick_once()
            if res == Status.FAILURE:
                rewards -= 1000
        for post_condition in self.post_conditions:
            res = post_condition.tick_once()
            if res == Status.SUCCESS:
                rewards += 1000
        return rewards

    # Note this should only be called after the node is inside a tree with setup children
    def calculate_accs(self):
        self.accs = find_accs(self)


class DefeatSkeleton(DynamicBaselinesNode):

    def __init__(self, agent, model=None):
        children = [
            TurnLeft(agent),
            TurnRight(agent),
            MoveForward(agent),
            MoveBackward(agent),
            Attack(agent)
        ]
        super().__init__(agent, "KillSkeleton", children, model)