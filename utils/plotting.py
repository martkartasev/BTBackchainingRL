import csv
import xml.etree.ElementTree as ET

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from stable_baselines3 import PPO

from bt.conditions import IsNotAttackedByEnemy, IsCloseToEntity
from mission.minecraft_types import Enemy
from utils.file import get_model_file_names_from_folder, get_project_root, get_absolute_path


def get_reward_series(filename):
    timesteps = list()
    values = list()
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            if row[0] == "Wall time":
                continue
            timesteps.append(float(row[1]))
            values.append(float(row[2]))
    df = pd.DataFrame()
    df["Timestep"] = timesteps
    df["Mean Reward"] = values
    return df


def plot_reward_series(data, spacing, figsize, yrange):
    # Make a data frame

    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('Set1')

    # multiple line plot
    num = 0
    fig = plt.figure(figsize=figsize)
    for name, values in data.items():
        num += 1

        # Find the right spot on the plot
        plt.subplot(spacing[0], spacing[1], num)

        for _, shadow in data.items():
            sns.lineplot(x="Timestep", y="Mean Reward", data=shadow, color="grey", alpha=0.2)
        # Plot the lineplot
        sns.lineplot(x="Timestep", y="Mean Reward", data=values)
        plt.ylim((yrange))

        # if num < 7:
        #     plt.xticks(ticks=[])
        #     plt.xlabel(None)
        plt.title(name, loc='left', fontsize=8, fontweight=0)

    plt.tight_layout()
    plt.show()


def plot_multi_series(data, figsize):
    # Make a data frame

    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('Set1')

    fig = plt.figure(figsize=figsize)

    f, ax = plt.subplots(1, 1)

    for name, values in data.items():
        sns.lineplot(x="Timestep", y="Mean Reward", data=values)

    plt.legend(labels=data.keys())
    plt.title("Training mean reward over", loc='left', fontsize=8, fontweight=0)
    plt.tight_layout()
    plt.show()


def plot_paths(spec, eval_dir, eval_name):
    model_file_names = get_model_file_names_from_folder(spec['model_log_dir'])
    time_steps = [
        PPO.load(get_project_root() / spec['model_log_dir'] / model_file_name).num_timesteps
        for model_file_name
        in model_file_names
    ]
    max_time_steps = max(time_steps)
    for i, time_steps in enumerate(time_steps):
        plot_path(f"{eval_dir}/{eval_name}_{i}.json", time_steps, max_time_steps)

    entity_plot_size = 0.5
    enemy_color = 'gray'
    goal_color = 'blue'
    player_color = 'red'
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)

    xml_namespaces = {"Malmo": "http://ProjectMalmo.microsoft.com"}
    xml_element = ET.parse(get_absolute_path(spec['mission']))
    entities = xml_element.findall(".//Malmo:DrawEntity", xml_namespaces)

    for entity in entities:
        position = (float(entity.get('x')), float(entity.get('z')))
        if Enemy.is_enemy(entity.get('type')):
            color = enemy_color
            entity_range = IsNotAttackedByEnemy.ENEMY_AGGRO_RANGE
        else:
            color = goal_color
            entity_range = IsCloseToEntity.RANGE

        plt.gca().add_patch(plt.Circle(position, entity_plot_size, color=color))
        plt.gca().add_patch(plt.Circle(position, entity_range, color=color, fill=False))

    placement = xml_element.find(".//Malmo:Placement", xml_namespaces)
    player_position = (float(placement.get('x')), float(placement.get('z')))
    plt.gca().add_patch(plt.Circle(player_position, entity_plot_size, color='r', zorder=100))

    cuboids = xml_element.findall(".//Malmo:DrawCuboid", xml_namespaces)
    air_cube = next((cuboid for cuboid in cuboids if cuboid.get('type') == 'air'), None)
    air_cube_start = (float(air_cube.get('x1')), float(air_cube.get('z1')))
    air_cube_end = (float(air_cube.get('x2')), float(air_cube.get('z2')))
    width = air_cube_end[0] - air_cube_start[0]
    height = air_cube_end[1] - air_cube_start[1]
    plt.gca().add_patch(
        plt.Rectangle(air_cube_start, width + 1, height + 1, linewidth=1, edgecolor='black', fill=False))

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Start Position', markerfacecolor=player_color, markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Goal Position', markerfacecolor=goal_color, markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Enemy Position', markerfacecolor=enemy_color, markersize=8)
    ]

    # Create the figure
    plt.legend(handles=legend_elements, loc='lower right')

    plt.colorbar(
        plt.cm.ScalarMappable(
            norm=colors.Normalize(0, max_time_steps),
            cmap=cm.get_cmap('viridis'),
        ),
        label='Training timesteps'
    )

    plt.show()


def plot_path(eval_file_name, time_steps, max_time_steps):
    colors = cm.get_cmap('viridis')
    with open(get_absolute_path(eval_file_name), "r") as file:
        record = jsonpickle.decode(file.read())
        positions = np.array([(position["x"], position["z"]) for position in record[0]["positions"]])
        plt.plot(positions[:, 0], positions[:, 1], color=colors(time_steps / max_time_steps))
