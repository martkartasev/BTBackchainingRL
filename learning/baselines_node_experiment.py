import os

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from bt.back_chain_tree import BackChainTree
from learning.agents.baselines_node_agent import BaselinesNodeAgent
from learning.baselines_node_training_env import BaselinesNodeTrainingEnv
from learning.save_best_model_callback import SaveOnBestTrainingRewardCallback
from mission_runner.baselines_node_testing_mission import BaselinesNodeTestingMission
from mission_runner.baselines_node_training_mission import BaselinesNodeTrainingMission
from utils.file import get_absolute_path, get_project_root
from utils.visualisation import save_tree_to_log

TOTAL_TIME_STEPS = 3000000

MODEL_LOG_DIR = "results/basicfighter3_good"
FINAL_MODEL_PATH = MODEL_LOG_DIR + "/finalbasicfarmer.mdl"


class BaselinesNodeExperiment:

    def __init__(self, goals, mission, tree_log="", hard_reset=True):
        self.mission_path = mission
        self.hard_reset = hard_reset

        self.agent = BaselinesNodeAgent()
        self.goals = [goal(self.agent) for goal in goals]
        self.tree = BackChainTree(self.agent, self.goals)

        if tree_log != "":
            save_tree_to_log(self.tree.root, tree_log)

        self.baseline_node = self.tree.baseline_nodes[0]

    def test_node(self, model):
        fighter_model = DQN.load(get_project_root() / MODEL_LOG_DIR / model)
        self.baseline_node.set_model(fighter_model)
        mission = BaselinesNodeTestingMission(self.agent, self.tree.root, get_absolute_path(self.mission_path))

        while True:
            mission.run()

    def setup_training_environment(self):
        mission = BaselinesNodeTrainingMission(self.agent, get_absolute_path(self.mission_path))

        os.makedirs(get_absolute_path(MODEL_LOG_DIR), exist_ok=True)
        env = BaselinesNodeTrainingEnv(self.baseline_node, mission, self.hard_reset)
        env = Monitor(env, get_absolute_path(MODEL_LOG_DIR))
        return env

    def train_node(self):
        env = self.setup_training_environment()

        model = DQN(
            'MultiInputPolicy', env, verbose=1, tensorboard_log=get_absolute_path("tensorboard"),
            exploration_fraction=0.05
        )
        model.learn(total_timesteps=TOTAL_TIME_STEPS,
                    callback=SaveOnBestTrainingRewardCallback(5000, log_dir=get_absolute_path(MODEL_LOG_DIR)))
        model.save(FINAL_MODEL_PATH)

    def check_env(self):
        env = self.setup_training_environment()
        check_env(env)
