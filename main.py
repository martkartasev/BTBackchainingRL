import os
import time

from stable_baselines3 import PPO

from baselines_node_experiment import BaselinesNodeExperiment
from bt import conditions
from evaluation.evaluation_manager import EvaluationManager
from learning.baseline_node import ChaseEntity
from mission.observation_manager import ObservationManager, RewardDefinition, ObservationDefinition
from utils.file import store_spec, load_spec, get_absolute_path, get_model_file_names_from_folder
from utils.plotting import get_reward_series, plot_reward_series, plot_positions, plot_multi_series
import xml.etree.ElementTree as ET

cow_skeleton_experiment = {
    "goals": [conditions.IsCloseToEntity],
    "mission": "resources/arena_cow_skeleton_v2.xml",
    "model_log_dir": "results/cow_skeleton_experiment_1_2",
    "model_class": PPO,
    "acc_ends_episode": True,
    "model_arguments": {
        "policy": 'MultiInputPolicy',
        "verbose": 1,
        "tensorboard_log": get_absolute_path("tensorboard"),
    },
    "active_entities": False,
    "baseline_node_type": ChaseEntity,
    "observation_manager": ObservationManager(
        observation_filter=[
            "entity_relative_distance",
            "entity_relative_direction",
            "enemy_relative_distance",
            "enemy_relative_direction",
            "health",
            "entity_visible",
            "surroundings"
        ],
        reward_definition=RewardDefinition(
            POST_CONDITION_FULFILLED_REWARD=1000,
            AGENT_DEAD_REWARD=-1000,
            ACC_VIOLATED_REWARD=-1000,
        )),
    "max_steps_per_episode": 1500,
    "total_timesteps": 3000000,
}

skeleton_fire_experiment_v2 = {
    "goals": [conditions.IsSafeFromFire, conditions.IsEnemyDefeated],
    "mission": "resources/arena_skeleton_v2.xml",
    "model_log_dir": "results/basicfighter_ppo10",
    "active_entities": True,
    "acc_ends_episode": False,
    "observation_manager": ObservationManager(
        observation_filter=[
            "enemy_relative_distance",
            "enemy_relative_direction",
            "health",
            "enemy_health",
            "enemy_targeted",
            "surroundings"
        ],
        reward_definition=RewardDefinition(
            POST_CONDITION_FULFILLED_REWARD=1000,
            AGENT_DEAD_REWARD=-1000,
            ACC_VIOLATED_REWARD=-10
        ),
        observation_definition=ObservationDefinition(
            GRID_SIZE_AXIS=[1, 11, 11],
            FIRE_AVOID_DISTANCE=1.5
        )
    ),
    "model_class": PPO,
    "model_arguments": {
        "policy": 'MultiInputPolicy',
        "verbose": 1,
        "tensorboard_log": get_absolute_path("tensorboard"),
    },
    "total_timesteps": 2000000,
}

cow_fire_experiment = {
    "goals": [conditions.IsSafeFromFire, conditions.IsNotHungry],
    "mission": "resources/arena_cow_v2.xml",
    "model_log_dir": "",
    "active_entities": True,
    "observation_manager": ObservationManager(observation_filter=[
        "entity_relative_position",
        "enemy_relative_position",
        "direction",
        "health",
        "entity_visible",
        "surroundings",
        "is_entity_pickable",
        "has_food",
        "satiation"
    ]),
}


def experiment_evaluate(log_dir, model_name, runs, eval_log_file=None):
    spec = load_spec(log_dir)
    spec["evaluation_manager"] = EvaluationManager(runs, eval_log_file)
    experiment = BaselinesNodeExperiment(**spec)

    experiment.evaluate_node(spec['model_class'], model_name, 3)


def experiment_test(log_dir, model_name):
    spec = load_spec(log_dir)
    experiment = BaselinesNodeExperiment(**spec)

    experiment.test_node(spec['model_class'], model_name)


def experiment_train(spec):
    experiment = BaselinesNodeExperiment(**spec)
    store_spec(spec)

    experiment.train_node(spec['model_class'], spec['model_arguments'])


# Can be used to verify gym env and generate mission spec when spec definition has changed.
def experiment_check_env(spec):
    experiment = BaselinesNodeExperiment(**spec)
    store_spec(spec)
    experiment.check_env()


def plot_rewards():
    data = {
        "ACC": get_reward_series(get_absolute_path(r"results\cow_skeleton_experiment_2\run-PPO_11-tag-rollout_ep_rew_mean.csv")),
        "ACC + Reward": get_reward_series(get_absolute_path(r"results\cow_skeleton_experiment_1_2\run-PPO_11-tag-rollout_ep_rew_mean (2).csv")),
        "Normal": get_reward_series(get_absolute_path(r"results\cow_skeleton_experiment_3_2\run-PPO_12-tag-rollout_ep_rew_mean.csv")),
        "Normal + Reward": get_reward_series(get_absolute_path(r"results\cow_skeleton_experiment\run-PPO_COW-tag-rollout_ep_rew_mean.csv")),
    }
    plot_multi_series(data, (5, 3.5))


def evaluate_all_models_once(log_dir, eval_dir, eval_name):
    model_files = get_model_file_names_from_folder(log_dir)
    for i, model in enumerate(model_files):
        experiment_evaluate(log_dir, model, 1, f"{eval_dir}/{eval_name}_{i}.json")


def evaluate_different_positions(log_dir, eval_dir, eval_name, model_name):
    spec = load_spec(log_dir)

    mission_path = spec['mission']
    xml_namespaces = {"Malmo": "http://ProjectMalmo.microsoft.com"}
    xml_element = ET.parse(get_absolute_path(mission_path))
    ET.register_namespace("", "http://ProjectMalmo.microsoft.com")

    split_path = mission_path.split(".")
    temp_mission_path = f"{split_path[0]}_temp.{split_path[1]}"
    for x in range(-15, 16):
        for z in range(-15, 16):
            print(x)
            print(z)
            placement = xml_element.find(".//Malmo:Placement", xml_namespaces)
            placement.set('x', str(x))
            placement.set('z', str(z))

            with open(get_absolute_path(temp_mission_path), 'w+') as f:
                xml_element.write(f, encoding='unicode')

            spec['mission'] = temp_mission_path
            spec["evaluation_manager"] = EvaluationManager(1, f"{eval_dir}/{eval_name}_pos_{x}_{z}.json")

            experiment = BaselinesNodeExperiment(**spec)
            experiment.evaluate_node(spec['model_class'], model_name, 3)
            os.remove(get_absolute_path(temp_mission_path))


if __name__ == '__main__':
    # experiment_evaluate("results/basicfighter_ppo6", "best_model_63", EvaluationManager(runs=50))
    # experiment_train(cow_skeleton_experiment)
    # evaluate_all_models_once("results/cow_skeleton_experiment", "log/eval", "cow_skeleton_experiment")
    # evaluate_different_positions("results/cow_skeleton_experiment", "log/eval", "cow_skeleton_experiment", "best_model_41.zip")
    # plot_positions("log/eval", "cow_skeleton_experiment")
    # store_spec(cow_skeleton_experiment)
    # experiment_check_env(skeleton_fire_experiment_v2)
    plot_rewards()
