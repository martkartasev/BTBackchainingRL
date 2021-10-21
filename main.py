from baselines_node_experiment import BaselinesNodeExperiment
from bt import conditions
from learning.baseline_node import ChaseEntity
from observation import ObservationManager, RewardDefinition

cow_skeleton_experiment = {
    "goals": [conditions.IsCloseToEntity],
    "mission": "resources/arena_cow_skeleton.xml",
    "model_log_dir": "",
    "tree_log": "cow_tree.txt",
    "hard_reset": False,
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
    "tree_log": "skeleton_tree.txt",
    "hard_reset": True,
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
}

skeleton_fire_experiment_v2 = {
    "goals": [conditions.IsSafeFromFire, conditions.IsEnemyDefeated],
    "mission": "resources/arena_skeleton_v2.xml",
    "model_log_dir": "results/basicfighter8",
    "tree_log": "skeleton_tree_v2.txt",
    "hard_reset": True,
    "observation_manager": ObservationManager(observation_filter=[  # TODO We should modify this so we provide the obs manager from here with parameters for filtering, but also global parameters like max distance, life, etc
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
            ACC_VIOLATED_REWARD=-1000
        )
    ),
    "total_timesteps": 3100000,
}

cow_fire_experiment = {
    "goals": [conditions.IsSafeFromFire, conditions.IsNotHungry],
    "mission": "resources/arena_cow_v2.xml",
    "model_log_dir": "",
    "tree_log": "cow_tree.txt",
    "hard_reset": True,
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


def experiment_train(specs):
    experiment = BaselinesNodeExperiment(**specs)
    experiment.train_node()


def experiment_test(specs, model):
    experiment = BaselinesNodeExperiment(**specs)
    experiment.test_node(model)


def experiment_check_env(specs):
    experiment = BaselinesNodeExperiment(**specs)
    experiment.check_env()


if __name__ == '__main__':
    experiment_train(skeleton_fire_experiment_v2)
    # experiment_test(cow_fire_experiment, "best_model_0")
    # experiment_check_env(cow_skeleton_experiment)
