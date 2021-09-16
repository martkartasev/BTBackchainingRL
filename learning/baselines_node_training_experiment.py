import os

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

IS_TRAINING = True
TOTAL_TIMESTEPS = 3000000

MISSION_PATH = "resources/arena_cow_skeleton.xml"


def train_node():
    mission_xml_path = get_absolute_path(MISSION_PATH)
    log_dir = get_absolute_path("results/basicfighter3_good")

    agent = BaselinesNodeAgent()
    goals = [conditions.IsCloseToEntity(agent)]
    tree = BackChainTree(agent, goals)

    mission = BaselinesNodeTrainingMission(agent, mission_xml_path)

    save_tree_to_log(tree.root, "cow_tree.txt")

    node = tree.baseline_nodes[0]

    os.makedirs(log_dir, exist_ok=True)
    env = BaselinesNodeTrainingEnv(node, mission)
    env = Monitor(env, log_dir)

    model = DQN(
        'MultiInputPolicy', env, verbose=1, tensorboard_log=get_absolute_path("tensorboard"), exploration_fraction=0.05
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=SaveOnBestTrainingRewardCallback(5000, log_dir=log_dir))
    model.save(log_dir + "/finalbasicfarmer.mdl")


def test_node():
    model_path = "results/basicfighter3_good/best_model_53"

    agent = BaselinesNodeAgent()
    goals = [conditions.IsCloseToEntity(agent)]
    tree = BackChainTree(agent, goals)

    node = tree.baseline_nodes[0]
    fighter_model = DQN.load(get_project_root() / model_path)
    node.set_model(fighter_model)

    mission_xml_path = get_absolute_path(MISSION_PATH)
    mission = BaselinesNodeTestingMission(agent, tree.root, mission_xml_path)

    while True:
        mission.run()


def test_env():
    mission_xml_path = get_absolute_path(MISSION_PATH)
    log_dir = get_absolute_path("results/basicfighter3_good")

    agent = BaselinesNodeAgent()
    goals = [conditions.IsCloseToEntity(agent)]
    tree = BackChainTree(agent, goals)

    mission = BaselinesNodeTrainingMission(agent, mission_xml_path)

    save_tree_to_log(tree.root, "cow_tree.txt")

    node = tree.baseline_nodes[0]

    os.makedirs(log_dir, exist_ok=True)
    env = BaselinesNodeTrainingEnv(node, mission)
    env = Monitor(env, log_dir)

    check_env(env)

if __name__ == '__main__':
    test_node()
