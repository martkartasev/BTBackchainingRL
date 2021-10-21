from baselines_node_experiment import BaselinesNodeExperiment
from bt import conditions
from learning.baseline_node import ChaseEntity

cow_skeleton_experiment = {
    "goals": [conditions.IsCloseToEntity],
    "mission": "resources/arena_cow_skeleton.xml",
    "tree_log": "cow_tree.txt",
    "hard_reset": False,
    "baseline_node_type": ChaseEntity,
    "observation_filter": [
        "entity_relative_position",
        "enemy_relative_position",
        "direction",
        "health",
        "entity_visible",
        "surroundings"
    ]
}

skeleton_fire_experiment = {
    "goals": [conditions.IsSafeFromFire, conditions.IsEnemyDefeated],
    "mission": "resources/arena_skeleton_v2.xml",
    "tree_log": "skeleton_tree.txt",
    "hard_reset": True,  # TODO: Add support for soft reset to experiments with fire
    "observation_filter": [
        "entity_relative_position",
        "enemy_relative_position",
        "direction", "health",
        "enemy_health",
        "entity_visible",
        "surroundings"
    ]
}

cow_fire_experiment = {
    "goals": [conditions.IsSafeFromFire, conditions.IsNotHungry],
    "mission": "resources/arena_cow_v2.xml",
    "tree_log": "cow_tree.txt",
    "hard_reset": True,  # TODO: Add support for soft reset to experiments with fire
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
    experiment_train(cow_skeleton_experiment)
    # experiment_test(cow_fire_experiment, "best_model_0") #TODO: Might want to consider building the path to "folder / model_x" differently somehow. Right now we have to change it in two different places
    # experiment_check_env(cow_skeleton_experiment)
