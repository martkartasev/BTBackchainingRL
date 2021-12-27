import numpy as np

from evaluation.evaluation_manager import EvaluationManager
from learning.baseline_node import DefeatSkeleton, ChaseEntity
from main import experiment_evaluate
from utils.file import get_absolute_path
from utils.plotting import get_reward_series, plot_multi_series


def plot_rewards_cow():
    data = {
        "Standard RL": get_reward_series(get_absolute_path(r"results\cow_skeleton_experiment_4_SA\run-PPO_COW_SA-tag-rollout_ep_rew_mean.csv")),
        "ACC Aware (Neg. Reward)": get_reward_series(get_absolute_path(r"results\cow_skeleton_experiment_4_SA_AR\run-PPO_COW_SA_AR-tag-rollout_ep_rew_mean.csv")),
        "ACC Aware (End Episode)": get_reward_series(get_absolute_path(r"results\cow_skeleton_experiment_4_FBT\run-PPO_COW_FBT-tag-rollout_ep_rew_mean.csv")),
        "ACC Aware (NR and EE)": get_reward_series(get_absolute_path(r"results\cow_skeleton_experiment_4_FBT_AR\run-PPO_COW_FBT_AR-tag-rollout_ep_rew_mean.csv")),
    }
    plot_multi_series(data, (5, 3.5))


def plot_rewards_skeleton():
    data = {
        "Standard RL": get_reward_series(get_absolute_path(r"results\basicfighter_ppo8\run-PPO_8-tag-rollout_ep_rew_mean.csv")),
        "ACC Aware (Neg. Reward)": get_reward_series(get_absolute_path(r"results\basicfighter_ppo10\run-PPO_10-tag-rollout_ep_rew_mean.csv")),
        "ACC Aware (End Episode)": get_reward_series(get_absolute_path(r"results\basicfighter_ppo5\run-PPO_5-tag-rollout_ep_rew_mean.csv")),
        "ACC Aware (NR and EE)": get_reward_series(get_absolute_path(r"results\basicfighter_ppo7\run-PPO_7-tag-rollout_ep_rew_mean.csv")),
    }
    plot_multi_series(data, (5, 3.5))


def evaluate_combined():
    print_node_results(experiment_evaluate(log_dir="results/cow_skeleton_experiment_4_SA", model_spec={
        DefeatSkeleton: ("results/basicfighter_ppo8", "final.mdl"),
        ChaseEntity: ("results/cow_skeleton_experiment_4_SA", "final.mdl")
    }, evaluation_manager=EvaluationManager(runs=1000)), "Is not attacked by enemy")
    print_node_results(experiment_evaluate(log_dir="results/cow_skeleton_experiment_4_SA_AR", model_spec={
        DefeatSkeleton: ("results/basicfighter_ppo10", "final.mdl"),
        ChaseEntity: ("results/cow_skeleton_experiment_4_SA_AR", "final.mdl")
    }, evaluation_manager=EvaluationManager(runs=1000)), "Is not attacked by enemy")
    print_node_results(experiment_evaluate(log_dir="results/cow_skeleton_experiment_4_FBT", model_spec={
        DefeatSkeleton: ("results/basicfighter_ppo5", "final.mdl"),
        ChaseEntity: ("results/cow_skeleton_experiment_4_FBT", "final.mdl")
    }, evaluation_manager=EvaluationManager(runs=1000)), "Is not attacked by enemy")
    print_node_results(experiment_evaluate(log_dir="results/cow_skeleton_experiment_4_FBT_AR", model_spec={
        DefeatSkeleton: ("results/basicfighter_ppo7", "final.mdl"),
        ChaseEntity: ("results/cow_skeleton_experiment_4_FBT_AR", "final.mdl")
    }, evaluation_manager=EvaluationManager(runs=1000)), "Is not attacked by enemy")


def evaluate_fighter():
    print_node_results(experiment_evaluate(log_dir="results/basicfighter_ppo8", model_spec={
        DefeatSkeleton: ("results/basicfighter_ppo8", "final.mdl"),
    }, evaluation_manager=EvaluationManager(runs=1000)), "Is safe from fire")
    print_node_results(experiment_evaluate(log_dir="results/basicfighter_ppo10", model_spec={
        DefeatSkeleton: ("results/basicfighter_ppo10", "final.mdl"),
    }, evaluation_manager=EvaluationManager(runs=1000)), "Is safe from fire")
    print_node_results(experiment_evaluate(log_dir="results/basicfighter_ppo5", model_spec={
        DefeatSkeleton: ("results/basicfighter_ppo5", "final.mdl"),
    }, evaluation_manager=EvaluationManager(runs=1000)), "Is safe from fire")
    print_node_results(experiment_evaluate(log_dir="results/basicfighter_ppo7", model_spec={
        DefeatSkeleton: ("results/basicfighter_ppo7", "final.mdl"),
    }, evaluation_manager=EvaluationManager(runs=1000)), "Is safe from fire")


def print_node_results(evaluation_manager, evaluated_node):
    values = np.array([[mission.steps, select(mission.nodes, evaluated_node).failures, mission.health] for mission in evaluation_manager.mission_records])
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Evaluation results: " + str(evaluation_manager.name))
    print(f"Tex string: & % Episodes ACC_f & Avg ACC_f & Std dev ACC_f & Avg health remaining & % of failures & Timesteps & Timesteps std dev ")
    print(f"Tex string: "
          f"& {np.sum(values[:, 1] > 0) / evaluation_manager.runs * 100} "
          f"& {np.average(values[:, 1])} "
          f"& {np.std(values[:, 1])} "
          f"& {np.average(values[:, 2])} "
          f"& {np.sum(values[:, 2] == 0) / evaluation_manager.runs * 100} "
          f"& {np.average(values[:, 0])} "
          f"& {np.std(values[:, 0])} "
          )
    # steps
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


def select(node_list, name):
    return list(filter(lambda el: el.name == name and el.steps != 0, node_list))[0]


if __name__ == '__main__':
  #  evaluate_fighter()
    evaluate_combined()