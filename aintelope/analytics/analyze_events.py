# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import os

from typing import Optional

import dateutil.parser as dparser
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import math
import numpy as np
import pandas as pd

import recording

from matplotlib import pyplot as plt

'''
Exporatory hacking ensues

Instructions:
When the models failed, then which dimensions of objectives they failed
Which dimensions they still maximised?
Did they at least try to follow homeostasis at all or just treated it as an unbounded objective?
In case of balancing gold and silver, did they at least try to balance or just maximised one objective?
Were there any unusual patterns in their action sequences? For example, repeating some action sequences needlessly?

'''



if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    folder = "/home/joel/project/ant_paper_results/ppo3/"
    df = recording.read_events(folder,"events.csv")[0]
    
    # Print column titles
    print(list(df.head(1)))
    # ['Run_id', 'Pipeline cycle', 'Episode', 'Trial', 'Step', 'IsTest', 
    # 'Agent_id', 'Action', 'Reward', 'Done', 'COOPERATION', 'FOOD', 
    # 'FOOD_DEFICIENCY', 'MOVEMENT']

    # Print max value of trial
    max_trial = df.iloc[df.Trial.argmax(), 3]
    print(max_trial)
    
    # Test if all of the values are the same between trial and episode. Yes
    all_equal = (df['Trial'] == df['Episode']).all()
    print(all_equal)

    # Break by trials to find the end-game
    # NEEDS NEW PLAN: trials same as episodes
    episode = df.loc[df['Trial'] == 10]
    print(episode.iloc[500:505,0:3])

    if False:
        for trial in range(0,max_trial):
            episodes = df.loc[df['Trial'] == trial]
            max_episode = episodes.iloc[episodes.Episode.argmax(), 2]
            print("Trial and max_episode",trial,max_episode)
            #episodes.loc[episodes['Episode'] == max_trial]

    # Assess when they fail and when not

    #

    