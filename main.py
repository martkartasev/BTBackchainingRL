from bt import conditions
from learning.baselines_node_experiment import BaselinesNodeExperiment

cow_skeleton_experiment = {
    "goals": [conditions.IsCloseToEntity],
    "mission": "resources/arena_cow_skeleton.xml",
    "tree_log": "cow_tree.txt",
    "hard_reset": False
}

skeleton_fire_experiment = {
    "goals": [conditions.IsEnemyDefeated],
    "mission": "resources/arena_skeleton_v2.xml",
    "tree_log": "skeleton_tree.txt",
    "hard_reset": True  # TODO: Add support for soft reset to experiments with fire
}

cow_fire_experiment = {
    "goals": [conditions.IsNotInFire, conditions.IsNotHungry],
    "mission": "resources/arena_cow_v2.xml",
    "tree_log": "cow_tree.txt",
    "hard_reset": True  # TODO: Add support for soft reset to experiments with fire
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
    experiment_train(skeleton_fire_experiment)
    # experiment_test(cow_skeleton_experiment, "best_model_53")
    # experiment_check_env(cow_skeleton_experiment)
