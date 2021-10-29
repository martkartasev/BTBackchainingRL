from stable_baselines3 import PPO, DQN

from baselines_node_experiment import BaselinesNodeExperiment
from bt import conditions
from learning.baseline_node import ChaseEntity
from mission.observation_manager import ObservationManager, RewardDefinition
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
    "total_timesteps": 3000000,
}

skeleton_fire_experiment_v2 = {
    "goals": [conditions.IsSafeFromFire, conditions.IsEnemyDefeated],
    "mission": "resources/arena_skeleton_v2.xml",
    "model_log_dir": "results/basicfighter_ppo4",
    "active_entities": True,
    "observation_manager": ObservationManager(observation_filter=[
        # TODO We should modify this so we provide the obs manager from here with parameters for filtering, but also global parameters like max distance, life, etc
        "enemy_relative_distance",
        "enemy_relative_direction",
        "health",
        "enemy_health",
        # "enemy_targeted",
        "surroundings"
    ],
        reward_definition=RewardDefinition(
            POST_CONDITION_FULFILLED_REWARD=1000,
            AGENT_DEAD_REWARD=-1000,
            ACC_VIOLATED_REWARD=-1000
        )
    ),
    "model_class": PPO,
    "model_arguments": {
        "policy": 'MultiInputPolicy',
        "verbose": 1,
        "tensorboard_log": get_absolute_path("tensorboard"),
    },
    "total_timesteps": 3000000,
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


def experiment_evaluate(log_dir, model):
    spec = load_spec(log_dir)
    experiment = BaselinesNodeExperiment(**spec)

    experiment.evaluate_node(model)


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


if __name__ == '__main__':
    experiment_train(cow_skeleton_experiment)
    # experiment_check_env(cow_skeleton_experiment)
