from msilib.schema import Patch

import jsonpickle
import numpy as np
from matplotlib.lines import Line2D
from stable_baselines3 import PPO, DQN

from baselines_node_experiment import BaselinesNodeExperiment
from bt import conditions
from bt.conditions import IsNotAttackedByEnemy, IsCloseToEntity
from evaluation.evaluation_manager import EvaluationManager
from learning.baseline_node import ChaseEntity
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from mission.observation_manager import ObservationManager, RewardDefinition, ObservationDefinition
from utils.plotting import get_reward_series, plot_reward_series
from utils.file import store_spec, load_spec, get_absolute_path

cow_skeleton_experiment = {
    "goals": [conditions.IsCloseToEntity],
    "mission": "resources/arena_cow_skeleton_v2.xml",
    "model_log_dir": "results/cow_skeleton_experiment",
    "model_class": PPO,
    "model_arguments": {
        "policy": 'MultiInputPolicy',
        "verbose": 1,
        "tensorboard_log": get_absolute_path("tensorboard"),
    },
    "active_entities": False,
    "baseline_node_type": ChaseEntity,
    "observation_manager": ObservationManager(observation_filter=[
        "entity_relative_position",
        "enemy_relative_position",
        "direction",
        "health",
        "entity_visible",
        "surroundings"
    ]),
}

skeleton_fire_experiment = {
    "goals": [conditions.IsSafeFromFire, conditions.IsEnemyDefeated],
    "mission": "resources/arena_skeleton_v2.xml",
    "model_log_dir": "",
    "active_entities": True,
    "observation_manager": ObservationManager(observation_filter=[
        "entity_relative_position",
        "enemy_relative_position",
        "direction", "health",
        "enemy_health",
        "entity_visible",
        "surroundings"
    ],
        reward_definition=RewardDefinition(
            ACC_VIOLATED_REWARD=-200
        )
    ),
    "total_timesteps": 1000,
}

skeleton_fire_experiment_v2 = {
    "goals": [conditions.IsSafeFromFire, conditions.IsEnemyDefeated],
    "mission": "resources/arena_skeleton_v2.xml",
    "model_log_dir": "results/basicfighter_ppo8",
    "active_entities": True,
    "acc_ends_episode": False,
    "observation_manager": ObservationManager(observation_filter=[
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
            ACC_VIOLATED_REWARD=0
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


def experiment_evaluate(log_dir, model_name, eval_log_file, runs):
    spec = load_spec(log_dir)
    spec["evaluation_manager"] = EvaluationManager(runs, eval_log_file)
    experiment = BaselinesNodeExperiment(**spec)

    experiment.evaluate_node(spec['model_class'], model_name, 60)


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
        "PPO5": get_reward_series(
            r"C:\Users\Mart9\Workspace\BTBackchainingRL\results\basicfighter_ppo5\run-PPO_5-tag-rollout_ep_rew_mean.csv"),
        "PPO7": get_reward_series(
            r"C:\Users\Mart9\Workspace\BTBackchainingRL\results\basicfighter_ppo7\run-PPO_7-tag-rollout_ep_rew_mean.csv"),
        #  "Targeting": get_reward_series(r"C:\Users\Mart\workspace\RLBT\resources\logs\simultaneous_node\run_DQNSimultaneousAgentAltTargetMix_2_targeting_1-tag-episode_reward.csv"),
    }
    plot_reward_series(data, (1, 1), (5, 3.5), (-2000, 3000))


def plot_paths(n_runs):
    for i in range(n_runs):
        """
        experiment_evaluate(
            "results/cow_skeleton_experiment",
            f"best_model_{i}",
            f"log/eval/cow_skeleton_experiment_{i}.json",
            1
        )
        """
        plot_path(i, n_runs)

    entity_size = 0.5
    player_position = (-14, 0)
    skeleton_position = (0, 0)
    cow_position = (14, 0)

    plt.xlim(-20, 20)
    plt.ylim(-20, 20)

    plt.gca().add_patch(plt.Circle(player_position, entity_size, color='r',  zorder=100))
    plt.gca().add_patch(plt.Circle(skeleton_position, entity_size, color='gray'))
    plt.gca().add_patch(plt.Circle(skeleton_position, IsNotAttackedByEnemy.ENEMY_AGGRO_RANGE, color='gray', fill=False))
    plt.gca().add_patch(plt.Circle(cow_position, entity_size, color='blue'))
    plt.gca().add_patch(plt.Circle(cow_position, IsCloseToEntity.RANGE, color='blue', fill=False))
    plt.gca().add_patch(plt.Rectangle((-15, -15), 31, 31, linewidth=1, edgecolor='black', fill=False))

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Start Position', markerfacecolor='r', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Goal Position', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Enemy Position', markerfacecolor='gray', markersize=8)
    ]

    # Create the figure
    plt.legend(handles=legend_elements, loc='lower right')

    plt.colorbar(plt.cm.ScalarMappable(
        norm=colors.Normalize(0, n_runs-1),
        cmap=cm.get_cmap('viridis')
    ))

    #plt.show()
    plt.savefig("positions")


def plot_path(i, n_runs):
    colors = cm.get_cmap('viridis')
    with open(get_absolute_path(f"log/eval/cow_skeleton_experiment_{i}.json"), "r") as file:
        record = jsonpickle.decode(file.read())
        positions = np.array([(position["x"], position["z"]) for position in record[0]["positions"]])
        x = positions[:, 0]
        z = positions[:, 1]
        color = colors(i/(n_runs-1))
        plt.plot(x, z, color=color, label=f"{i}")

if __name__ == '__main__':
    # experiment_evaluate("results/basicfighter_ppo6", "best_model_63", EvaluationManager(runs=50))
    # experiment_train(skeleton_fire_experiment_v2)
    plot_paths(42)
    # store_spec(cow_skeleton_experiment)
    # experiment_check_env(skeleton_fire_experiment_v2)
    # plot_rewards()
