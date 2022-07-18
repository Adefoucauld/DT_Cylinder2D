# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:37:52 2022

@author: Utilisateur
"""

"""Source: decision-transformer/gym/data/download_d4rl_datasets.py"""


''' -----obj :  return list of episode with {obs, next_obs, reward, action} -------------'''

# will do it from results obtained through test on cluster"

from IPython import get_ipython
get_ipython().run_line_magic('reset','-sf')


import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import csv
import pickle

cwd = os.getcwd()
path_to_data = os.path.join(cwd,'2006_sqrt_base_pres_DTset')
# print (path)

#path_to_data = 'C:/Users/Utilisateur/Documents/Imperial/Research project/results_simulation/simulations_done/2006_sqrt_base_pres_DTset/'

'''import and clean data'''

def collect_dataset(path_to_data):

    dt_obs = pd.read_csv(path_to_data+'/test_strategy.csv',delimiter = ';')
    dt_reward = pd.read_csv(path_to_data + '/rewards.csv', delimiter = ';')
    dt_action = pd.read_csv(path_to_data + '/actions.csv',delimiter = ';')
    #only save action steps
    dt_obs = dt_obs[dt_obs['Step'].isin(list(dt_reward['Step']))].reset_index(drop = True)
    dt_obs = dt_obs.drop(columns = ['Name','Step','Drag','Lift','RecircArea','Jet0','Jet1'])
    
    #build dataset
    dataset = pd.concat([dt_obs,dt_reward.Reward,dt_action.Action_0,dt_action.Action_1], axis = 1)
    return (dt_obs,dt_reward,dt_action)


(dt_obs,dt_reward,dt_action) = collect_dataset(path_to_data)

def build_paths(dt_obs,dt_reward,dt_action):
    N = dt_reward.Reward.shape[0]
    paths = []
    for i in range(N-1):
        dataset = {'observations':dt_obs.loc[i].to_numpy().reshape(1,len(dt_obs.loc[i])),
                   'actions' : np.array([dt_action.Action_0.loc[i],dt_action.Action_1.loc[i]]).reshape(1,2),
                   'next_observations' : dt_obs.loc[i+1].to_numpy().reshape(1,len(dt_obs.loc[i])),
                   'rewards' : np.array([[dt_reward.Reward.loc[i]]])
                   }
        paths.append(dataset)
    return paths
        
paths = build_paths(dt_obs,dt_reward,dt_action)

with open('test.pkl', 'wb') as f:
            pickle.dump(paths, f)
