import copy

from py_trees.behaviour import Behaviour
from py_trees.common import Status

import numpy as np


class EvaluationManager:

    def __init__(self, runs=50):
        self.runs = runs

        self.nodes = dict()
        self.mission_results = list()
        self.current_record = MissionRecord(None)

    def register_node(self, node):
        record = NodeRecord(node)
        self.nodes[node] = record
        return record

    def record_mission_start(self, start):
        self.current_record = MissionRecord(start)

    def record_mission_end(self, state, steps, end):
        self.current_record.finalize(state, steps, end, self.nodes)
        [node.reset() for node in self.nodes.values()]
        self.mission_results.append(self.current_record)

    def record_node(self, node, status):
        record = self.nodes[node]
        record.steps += 1
        if status == Status.SUCCESS:
            record.successes += 1
        else:
            record.failures += 1


class MissionRecord:
    def __init__(self, start):
        self.steps = 0
        self.end = 0
        self.start = start
        self.nodes = list()

    def finalize(self, state, steps, end, nodes: dict):
        self.steps = steps
        self.end = end
        self.nodes = [copy.deepcopy(node) for node in nodes.values()]


class NodeRecord:
    def __init__(self, node):
        self.name = node.name
        self.failures = 0
        self.successes = 0
        self.steps = 0

    def reset(self):
        self.failures = 0
        self.successes = 0
        self.steps = 0


class EvaluationBehaviour(Behaviour):

    def __init__(self, embedded: Behaviour, manager: EvaluationManager):
        super().__init__(name=embedded.name)
        self.embedded = embedded
        self.manager = manager
        self.record = self.manager.register_node(self)

    def update(self):
        update = self.embedded.update()
        self.manager.record_node(self, update)
        return update

    def terminate(self, new_status):
        return self.embedded.terminate(new_status)


def print_skeleton_fire_results(evaluation_manager):
    values = np.array([[mission.steps, select(mission.nodes, "Is safe from fire").failures] for mission in evaluation_manager.mission_results])
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Evaluation results: ")
    print("Tex string: {0} & {1} & {2} & {3} ".format("Column", np.average(values[:, 0]), np.sum(values[:, 1]), np.average(values[:, 1])))  # steps
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


def select(node_list, name):
    return list(filter(lambda el: el.name == name and el.steps != 0, node_list))[0]
