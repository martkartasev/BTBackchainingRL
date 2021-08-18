import numpy as np
from stable_baselines3 import A2C

from agent import ObservationAgent
from bt.actions import TurnLeft, TurnRight, MoveForward, MoveBackward, Attack
from bt.sequence import Sequence
from learning.baseline_node import BasicFighterTrainingNode


class TrainedSkeletonFightingAgent(ObservationAgent):

    def __init__(self, name="execution/manual_skeleton"):
        super().__init__()
        self.name = name
        self.tree = self.define_behavior()

    def reset_agent(self):
        self.observations = None

    def is_mission_over(self):
        return not (self.is_enemy_close() and self.is_agent_alive())

    def control_loop(self):
        self.tree.tick_once()

    def define_behavior(self):
        fighter_model = A2C.load("../../results/basicfighter2/finalbasicfighter.mdl")
        root = Sequence("Root", children=[
            BasicFighterTrainingNode(self, model=fighter_model, children=[TurnLeft(self),
                                                                          TurnRight(self),
                                                                          MoveForward(self),
                                                                          MoveBackward(self),
                                                                          Attack(self)])
        ])

        return root

    def is_enemy_close(self):
        return self.observation.lookToPosition is not None
