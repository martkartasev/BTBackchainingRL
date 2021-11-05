from py_trees.behaviour import Behaviour
from py_trees.common import Status


class EvaluationManager:

    def __init__(self):
        self.nodes = dict()

    def record_mission(self, state, steps, start, end):
        pass

    def record_node(self, node, status):
        record = self.nodes[node]
        record.steps += 1
        if status == Status.SUCCESS:
            record.successes += 1
        else:
            record.failures += 1

    def register_node(self, node):
        record = NodeRecord(node)
        self.nodes[node] = record
        return record


class NodeRecord:
    def __init__(self, node):
        self.node = node
        self.failures = 0
        self.successes = 0
        self.steps = 0


class EvaluationBehaviour(Behaviour):

    def __init__(self, embedded: Behaviour, manager: EvaluationManager):
        super(EvaluationBehaviour, self).__init__(name=embedded.name)
        self.embedded = embedded
        self.manager = manager
        self.record = self.manager.register_node(self)

    def update(self):
        update = self.embedded.update()
        self.manager.record_node(self.record, update)
        return update

    def terminate(self, new_status):
        return self.embedded.terminate(new_status)
