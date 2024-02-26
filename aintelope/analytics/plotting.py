from typing import Optional

import dateutil.parser as dparser
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt

"""
Create and return plots for various analytics.
"""


def plot_history(events):
    """
    Plot the events from a history.
    args:
        events: pandas DataFrame
    return:
        plot: matplotlib.axes.Axes
    """
    plot = "NYI"

    return plot


def plot_performance(all_events, score_dimensions, save_path: Optional[str]):
    """
    Plot performance between rewards and scores.
    Accepts a list of event records from which a boxplot is done.
    TODO: further consideration should be had on *what* to average over.
    """
    keys = ["Run_id", "Episode", "Agent_id", "Reward"] + score_dimensions
    data = pd.DataFrame(columns=keys)
    for events in all_events:
        if len(data) == 0:
            data = events[
                keys
            ].copy()  # needed to avoid Pandas complaining about empty dataframe
        else:
            data = pd.concat([data, events[keys]])

    data["Reward"] = data["Reward"].astype(float)
    data[score_dimensions] = data[score_dimensions].astype(float)

    plots = data.groupby(["Run_id", "Episode", "Agent_id"]).mean()

    fig = plt.figure()
    plt.plot(plots[["Reward"] + score_dimensions].to_numpy())
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.legend("RS")

    if save_path:
        save_plot(fig, save_path)
    return fig


def plot_heatmap(agent, env):
    """
    Plot how the agent sees the values in an environment.
    """
    plot = "NYI"
    return plot


def save_plot(plot, save_path):
    """
    Save plot to file. Will get deprecated if nothing else comes here.
    """
    plot.savefig(save_path)
