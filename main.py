from py_trees.composites import Sequence
from stable_baselines3 import PPO

from baselines_node_experiment import BaselinesNodeExperiment
from bt import conditions
from bt.actions import Attack, Target, KeepDistance
from learning.baseline_node import ChaseEntity
from mission.observation_manager import ObservationManager, RewardDefinition, ObservationDefinition
from utils.file import store_spec, load_spec, get_absolute_path

cow_skeleton_experiment = {
    "goals": [conditions.IsCloseToEntity],
    "mission": "resources/arena_cow_skeleton_v2.xml",
    "model_log_dir": "results/cow_skeleton_experiment",
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
        reward_definition=RewardDefinition(
            POST_CONDITION_FULFILLED_REWARD=1000,
            AGENT_DEAD_REWARD=-1000,
            ACC_VIOLATED_REWARD=-10,
        ),
        observation_definition=ObservationDefinition(
            GRID_SIZE_AXIS=[1, 11, 11],
            FIRE_AVOID_DISTANCE=1.5
        )
    ),
    "observation_filter": [
        "entity_relative_distance",
        "entity_relative_direction",
        "enemy_relative_distance",
        "enemy_relative_direction",
        "health",
        "entity_visible",
        "surroundings"
    ],
    "max_steps_per_episode": 2500,
    "total_timesteps": 2000000,
    "random_position_range": {'x': [-14, -12], 'y': [6], 'z': [-12, 12]},
    "random_entities_position_range": {
        "cow": {'x': [14], 'y': [4], 'z': [-12, 12]},
        "skeleton": {'x': [0], 'y': [4], 'z': [-12, 12]}
    },
    'mission_max_time': 30
}

skeleton_fire_experiment_v2 = {
    "goals": [conditions.IsSafeFromFire, conditions.IsEnemyDefeated],
    "mission": "resources/arena_skeleton_v2.xml",
    "model_log_dir": "results/basicfighter_ppo11",
    "active_entities": True,
    "acc_ends_episode": False,
    "observation_manager": ObservationManager(
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
    "observation_filter": [
        "enemy_relative_distance",
        "enemy_relative_direction",
        "health",
        "enemy_health",
        "enemy_targeted",
        "surroundings"
    ],
    "model_class": PPO,
    "model_arguments": {
        "policy": 'MultiInputPolicy',
        "verbose": 1,
        "tensorboard_log": get_absolute_path("tensorboard"),
    },
    "total_timesteps": 3000000,
    "logging": 1
}

skeleton_fire_experiment_manual = {
    "mission": "resources/arena_skeleton_v2.xml",
    "observation_manager": ObservationManager(),
    "logging": 1
}

cow_fire_experiment = {
    "goals": [conditions.IsSafeFromFire, conditions.IsNotHungry],
    "mission": "resources/arena_cow_v2.xml",
    "model_log_dir": "",
    "active_entities": True,
    "observation_manager": ObservationManager(),
    "observation_filter": [
        "entity_relative_position",
        "enemy_relative_position",
        "direction",
        "health",
        "entity_visible",
        "surroundings",
        "is_entity_pickable",
        "has_food",
        "satiation"
    ]
}


def experiment_evaluate(log_dir, model_spec, evaluation_manager):
    spec = load_spec(log_dir)
    spec["evaluation_manager"] = evaluation_manager
    evaluation_manager.name = spec["model_log_dir"]
    experiment = BaselinesNodeExperiment(**spec)

    experiment.evaluate(model_spec)
    return evaluation_manager


def experiment_test(spec):
    experiment = BaselinesNodeExperiment(**spec)

    agent = experiment.agent
    agent.tree = Sequence("TestSequence", memory=False,
                          children=[Target(agent),
                                    KeepDistance(agent),
                                    Attack(agent)]
                          )

    experiment.mission.run()


def experiment_train(spec):
    experiment = BaselinesNodeExperiment(**spec)
    store_spec(spec)

    experiment.train_node(spec['model_class'], spec['model_arguments'])


# Can be used to verify gym env and generate mission spec when spec definition has changed.
def experiment_check_env(spec):
    experiment = BaselinesNodeExperiment(**spec)
    store_spec(spec)
    experiment.check_env()


if __name__ == '__main__':
    # experiment_test(skeleton_fire_experiment_manual)
    experiment_train(cow_skeleton_experiment)
    # evaluate_all_models_once("results/cow_skeleton_experiment", "log/eval", "cow_skeleton_experiment")
    # evaluate_different_positions("results/cow_skeleton_experiment", "log/eval", "cow_skeleton_experiment", "best_model_41.zip")
    # plot_positions("log/eval", "cow_skeleton_experiment")
    # store_spec(cow_skeleton_experiment)
    # experiment_check_env(skeleton_fire_experiment_v2)
