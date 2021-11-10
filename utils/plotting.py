import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
