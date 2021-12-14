import copy
import json
import os

from dataclasses import dataclass

import jsonpickle
from py_trees.behaviour import Behaviour
from py_trees.common import Status

from conditions import Condition
from utils.file import get_absolute_path

import numpy as np


class EvaluationManager:

    def __init__(self, runs=50, eval_log_file=None, name=""):
        self.runs = runs
        self.log_file = get_absolute_path(eval_log_file) if eval_log_file is not None else None
        self.nodes = dict()
        self.positions = list()
        self.mission_records = list()
        self.current_record = MissionRecord(None)
        self.name = name

    def register_node(self, node):
        record = NodeRecord(node)
        self.nodes[node] = record
        return record

    def record_mission_start(self, start):
        self.current_record = MissionRecord(start)

    def record_mission_end(self, state, steps, end):
        self.current_record.finalize(state, steps, end, self.nodes, self.positions)
        [node.reset() for node in self.nodes.values()]
        self.mission_records.append(self.current_record)

    def record_node(self, node, status):
        record = self.nodes[node]
        record.steps += 1
        if status == Status.SUCCESS:
            record.successes += 1
        else:
            record.failures += 1

    def record_position(self, x, z):
        self.positions.append(PositionRecord(x, z))

    def store_evaluation(self):
        if self.log_file is not None:
            jsonpickle.set_encoder_options('simplejson', sort_keys=True, indent=4)
            eval_json = jsonpickle.encode(self.mission_records, unpicklable=False)
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w+') as file:
                file.write(eval_json)


class MissionRecord:
    def __init__(self, start):
        self.steps = 0
        self.end = 0
        self.start = start
        self.positions = list()
        self.nodes = list()
        self.rewards = list()

    def finalize(self, state, steps, end, nodes: dict, positions: list):
        self.steps = steps
        self.end = end
        self.nodes = [copy.deepcopy(node) for node in nodes.values()]
        self.positions = copy.deepcopy(positions)


@dataclass
class PositionRecord:
    def __init__(self, x, z):
        self.x = float(x)
        self.z = float(z)


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

