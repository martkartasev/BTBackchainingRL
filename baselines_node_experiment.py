import os

try:
    from MalmoPython import AgentHost
except ImportError:
    from malmo.MalmoPython import AgentHost

from agents.behavior_tree_agent import BehaviorTreeAgent
from bt.back_chain_tree import BackChainTree
from learning.baselines_node_training_env import BaselinesNodeTrainingEnv
from learning.disable_malmo_ai_for_training_callback import DisableMalmoAIForTrainingCallback
from learning.save_best_model_callback import SaveOnBestTrainingRewardCallback
from mission.mission_runner import MissionRunner
from pathlib import Path
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from utils.file import get_absolute_path, get_project_root
from utils.visualisation import save_tree_to_log


class BaselinesNodeExperiment:

    def __init__(self, goals, mission, model_log_dir, total_timesteps=3000000, max_steps_per_episode=15000,
                 active_entities=True,
                 baseline_node_type=None, observation_manager=None, evaluation_manager=None, acc_ends_episode=True,
                 random_position_range=None, **kwargs):
        self.mission_path = mission
        self.active_entities = active_entities
        self.model_log_dir = model_log_dir
        self.total_timesteps = total_timesteps
        self.evaluation_manager = evaluation_manager
        self.acc_ends_episode = acc_ends_episode
        self.max_steps_per_episode = max_steps_per_episode
        self.random_position_range = random_position_range

        agent_host = AgentHost()
        self.agent = BehaviorTreeAgent(agent_host, observation_manager)
        self.goals = [goal(self.agent) for goal in goals]
        self.tree = BackChainTree(self.agent, self.goals, evaluation_manager)
        self.agent.tree = self.tree.root

        save_tree_to_log(self.tree.root, str(Path(model_log_dir) / "tree.csv"))

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

    def test_node(self, model_class, model_name):
        loaded_model = model_class.load(get_project_root() / self.model_log_dir / model_name)
        self.baseline_node.set_model(loaded_model)

        mission = MissionRunner(
            agent=self.agent,
            active_entities=self.active_entities,
            filename=get_absolute_path(self.mission_path),
            random_position_range=self.random_position_range
        )

        mission.run()

    def evaluate_node(self, model_class, model_name, mission_max_time=None):
        loaded_model = model_class.load(get_project_root() / self.model_log_dir / model_name)
        self.baseline_node.set_model(loaded_model)

        mission = MissionRunner(
            agent=self.agent,
            active_entities=self.active_entities,
            filename=get_absolute_path(self.mission_path),
            evaluation_manager=self.evaluation_manager,
            mission_max_time=mission_max_time,
            random_position_range=self.random_position_range
        )

        mission.run()

    def train_node(self, model_class, model_args):
        env = self.setup_training_environment()

        model_class = model_class(env=env, **model_args)
        model_class.learn(
            total_timesteps=self.total_timesteps,
            callback=[
                SaveOnBestTrainingRewardCallback(
                    check_freq=5000,
                    log_dir=get_absolute_path(self.model_log_dir)
                ),
                DisableMalmoAIForTrainingCallback(
                    mission_manager=env.mission.mission_manager,
                    agent=self.agent,
                    active_entities=self.active_entities
                )
            ]
        )

        model_class.save(self.model_log_dir + "/final.mdl")

    def check_env(self):
        env = self.setup_training_environment()
        check_env(env)

    def setup_training_environment(self):
        mission = MissionRunner(
            agent=self.agent,
            active_entities=self.active_entities,
            filename=get_absolute_path(self.mission_path),
            random_position_range=self.random_position_range
        )

        os.makedirs(get_absolute_path(self.model_log_dir), exist_ok=True)
        env = BaselinesNodeTrainingEnv(self.baseline_node, mission, self.acc_ends_episode,
                                       max_steps_per_episode=self.max_steps_per_episode)
        env = Monitor(env, get_absolute_path(self.model_log_dir))
        return env
