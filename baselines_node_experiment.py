import os

from MalmoPython import AgentHost
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from bt.back_chain_tree import BackChainTree
from learning.agents.baselines_node_agent import BaselinesNodeAgent
from learning.baselines_node_training_env import BaselinesNodeTrainingEnv
from learning.save_best_model_callback import SaveOnBestTrainingRewardCallback
from mission.baselines_node_mission import BaselinesNodeMission
from observation import ObservationManager
from utils.file import get_absolute_path, get_project_root
from utils.visualisation import save_tree_to_log

TOTAL_TIME_STEPS = 3000000

MODEL_LOG_DIR = "results/basicfighter3_good"
FINAL_MODEL_PATH = MODEL_LOG_DIR + "/finalbasicfarmer.mdl"


class BaselinesNodeExperiment:

    def __init__(self, goals, mission, tree_log="", hard_reset=True, baseline_node_type=None, observation_filter=None):
        self.mission_path = mission
        self.hard_reset = hard_reset

        agent_host = AgentHost()
        self.agent = BaselinesNodeAgent(agent_host, ObservationManager(observation_filter))
        self.goals = [goal(self.agent) for goal in goals]
        self.tree = BackChainTree(self.agent, self.goals)
        self.agent.tree = self.tree.root

        if tree_log != "":
            save_tree_to_log(self.tree.root, tree_log)

        if len(self.tree.baseline_nodes) == 0:
            raise ValueError("The tree doesn't have a baseline node")
        elif len(self.tree.baseline_nodes) > 1:
            if baseline_node_type is None:
                raise ValueError("The tree has two or more baseline nodes. Specify a baseline node type.")
            for baseline_node in self.tree.baseline_nodes:
                if isinstance(baseline_node, baseline_node_type):
                    self.baseline_node = baseline_node
                    break
            else:
                raise ValueError("The tree does not contain the baseline node type.")

        self.baseline_node = self.tree.baseline_nodes[0]
        if baseline_node_type is not None and not isinstance(self.baseline_node, baseline_node_type):
            raise ValueError("The tree does not contain the baseline node type.")

    def test_node(self, model):
        fighter_model = DQN.load(get_project_root() / MODEL_LOG_DIR / model)
        self.baseline_node.set_model(fighter_model)

        mission = BaselinesNodeMission(
            self.agent, get_absolute_path(self.mission_path), self.hard_reset
        )

        mission.run()

    def setup_training_environment(self):
        mission = BaselinesNodeMission(self.agent, get_absolute_path(self.mission_path))

        os.makedirs(get_absolute_path(MODEL_LOG_DIR), exist_ok=True)
        env = BaselinesNodeTrainingEnv(self.baseline_node, mission, self.hard_reset)
        env = Monitor(env, get_absolute_path(MODEL_LOG_DIR))
        return env

    def train_node(self):
        env = self.setup_training_environment()

        model = DQN(
            'MultiInputPolicy',
            env,
            verbose=1,
            tensorboard_log=get_absolute_path("tensorboard"),
            exploration_fraction=0.05
        )
        model.learn(total_timesteps=TOTAL_TIME_STEPS,
                    callback=SaveOnBestTrainingRewardCallback(5000, log_dir=get_absolute_path(MODEL_LOG_DIR)))
        model.save(FINAL_MODEL_PATH)

    def check_env(self):
        env = self.setup_training_environment()
        check_env(env)
