import os
import glob
#import sys
#import copy
#import distutils
#import shutil
#import json
import pandas as pd
#import numpy as np
#from collections import OrderedDict

# from command_runner.elevate import elevate

#import hydra
#from omegaconf import DictConfig, OmegaConf
#from flatten_dict import flatten, unflatten
#from flatten_dict.reducers import make_reducer

# this one is cross-platform
from filelock import FileLock

import seaborn as sns
from matplotlib import pyplot as plt

from aintelope.analytics.plotting import save_plot
# from aintelope.config.config_utils import register_resolvers
# from aintelope.utils import wait_for_enter, try_df_to_csv_write, RobustProgressBar


# need to specify config_path since we are in a subfolder and hydra does not automatically pay attention to current working directory. By default, hydra uses the directory of current file instead.
#@hydra.main(
#    version_base=None,
#    config_path=os.path.join(os.getcwd(), "aintelope", "config"),
#    config_name="config_experiment",
#)
# def boxplot(cfg: DictConfig) -> None:
def boxplot() -> None:

    source_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", ".."
    )
    datafiles = glob.glob(os.path.join(source_dir, "*_best_parameters_by_score_all_cycles.csv"))

    dfs = []
    for filepath in datafiles:
        print(f"Reading {filepath}")
        with FileLock(
                str(filepath)
                + ".lock"  # filepath may be PosixPath, so need to convert to str
            ):  # NB! take the lock inside the loop, not outside, so that when we are waiting for user confirmation for retry, we do not block other processes during that wait
                df = pd.read_csv(filepath)  
                dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)

    # df = df[df["experiment_name"] != "e_5_sustainability"]
    # df["experiment_name"] = df["experiment_name"].str.replace('e_5_sustainability2', 'e_5_sustainability')

    df["experiment_name"] = df["experiment_name"].str.replace('_', ' ')

    plt.rcParams['figure.constrained_layout.use'] = True    # ensure that plot labels fit to the image and do not overlap

    axes = sns.boxplot(
        data=df,
        x="test_averages.Score",
        y="experiment_name",
        hue="params_set_title",
        showfliers=False,
        orient="h",
    )  # "y" means grouping by experiment, "hue" means bars inside a group of experiment
    axes.set(xlabel='score', ylabel="experiment name")    

    save_path = os.path.join(source_dir, "boxplots")
    save_plot(axes.figure, save_path)

    plt.ion()
    axes.figure.show()
    plt.draw()
    plt.pause(
        60
    )  # render the plot. Usually the plot is rendered quickly but sometimes it may require up to 60 sec. Else you get just a blank window

    qqq = True


if __name__ == "__main__":
    boxplot()
