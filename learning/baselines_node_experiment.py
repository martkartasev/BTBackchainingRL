import os
from enum import Enum

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from bt import conditions
from bt.back_chain_tree import BackChainTree
from learning.agents.baselines_node_agent import BaselinesNodeAgent
from learning.baselines_node_training_env import BaselinesNodeTrainingEnv
from learning.save_best_model_callback import SaveOnBestTrainingRewardCallback
from mission_runner.baselines_node_testing_mission import BaselinesNodeTestingMission
from mission_runner.baselines_node_training_mission import BaselinesNodeTrainingMission
from utils.file import get_absolute_path, get_project_root
from utils.visualisation import save_tree_to_log


class RunType(Enum):
    TRAIN_NODE = 0
    TEST_NODE = 1
    TEST_ENV = 2


TOTAL_TIME_STEPS = 3000000

MISSION_PATH = "resources/arena_cow_skeleton.xml"
MODEL_LOG_DIR = "results/basicfighter3_good"
TREE_LOG_DIR = "cow_tree.txt"
MODEL_PATH = MODEL_LOG_DIR + "/best_model_53"
FINAL_MODEL_PATH = MODEL_LOG_DIR + "/finalbasicfarmer.mdl"


class BaselinesNodeExperiment:

    def __init__(self, run_type):
        self.run_type = run_type

        self.agent = BaselinesNodeAgent()
        self.goals = [conditions.IsCloseToEntity(self.agent)]
        self.tree = BackChainTree(self.agent, self.goals)
        if TREE_LOG_DIR != "":
            save_tree_to_log(self.tree.root, TREE_LOG_DIR)

        self.baseline_node = self.tree.baseline_nodes[0]

    def run(self):
        if self.run_type == RunType.TEST_NODE:
            fighter_model = DQN.load(get_project_root() / MODEL_PATH)
            self.baseline_node.set_model(fighter_model)
            mission = BaselinesNodeTestingMission(self.agent, self.tree.root, get_absolute_path(MISSION_PATH))

            while True:
                mission.run()
        else:
            mission = BaselinesNodeTrainingMission(self.agent, get_absolute_path(MISSION_PATH))

            os.makedirs(get_absolute_path(MODEL_LOG_DIR), exist_ok=True)
            env = BaselinesNodeTrainingEnv(self.baseline_node, mission)
            env = Monitor(env, get_absolute_path(MODEL_LOG_DIR))

            if self.run_type == RunType.TEST_ENV:
                check_env(env)
            else:
                model = DQN(
                    'MultiInputPolicy', env, verbose=1, tensorboard_log=get_absolute_path("tensorboard"),
                    exploration_fraction=0.05
                )
                model.learn(total_timesteps=TOTAL_TIME_STEPS,
                            callback=SaveOnBestTrainingRewardCallback(5000, log_dir=get_absolute_path(MODEL_LOG_DIR)))
                model.save(FINAL_MODEL_PATH)


if __name__ == '__main__':
    experiment = BaselinesNodeExperiment(RunType.TRAIN_NODE)
    experiment.run()
