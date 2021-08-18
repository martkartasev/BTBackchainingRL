import os

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from bt.actions import TurnLeft, TurnRight, MoveForward, MoveBackward, Attack
from learning.agents.baselines_node_training_agent import BasicFighterNodeTrainingAgent
from learning.agents.trained_skeleton_fighting_agent import TrainedSkeletonFightingAgent
from learning.baselines_node_traning_env import BaselinesNodeTrainingEnv
from mission_runner.baselines_node_training_mission import BaselinesNodeTrainingMission
from learning.baseline_node import BasicFighterTrainingNode
from learning.save_best_model_callback import SaveOnBestTrainingRewardCallback
from mission_runner.normal_mission import NormalMission
from utils.file import get_absolute_path


def train_node():
    total_timesteps = 200000
    mission_xml_path = get_absolute_path("resources/arena_skeleton.xml")
    log_dir = get_absolute_path("results/basicfighter3")

    agent = BasicFighterNodeTrainingAgent()
    mission = BaselinesNodeTrainingMission(agent, mission_xml_path)

    node = BasicFighterTrainingNode(agent, name="root", children=[TurnLeft(agent),
                                                                  TurnRight(agent),
                                                                  MoveForward(agent),
                                                                  MoveBackward(agent),
                                                                  Attack(agent)])
    node.setup_with_descendants()

    os.makedirs(log_dir, exist_ok=True)
    env = BaselinesNodeTrainingEnv(node, mission)
    env = Monitor(env, log_dir)

    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="./../../tensorboard/")
    model.learn(total_timesteps=total_timesteps, callback=SaveOnBestTrainingRewardCallback(5000, log_dir=log_dir))
    model.save(log_dir + "/finalbasicfighter.mdl")


def test_node():
    agent = TrainedSkeletonFightingAgent()
    mission_xml_path = "../../resources/arena_skeleton.xml"
    mission = NormalMission(agent, mission_xml_path)
    while True:
        mission.run()


if __name__ == '__main__':
    train_node()
