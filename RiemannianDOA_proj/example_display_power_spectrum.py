# %%
import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
# %matplotlib ipympl

from RunSingleMCIteration import run_single_mc_iteration
from utils import *
import os
import pickle
# %%
import utils
import ToolsMC
import importlib
importlib.reload(utils)
importlib.reload(ToolsMC)
from utils import *
from ToolsMC import *
# %%
def example_display_power_spectrum():
    # %%
    filepath_results_file = '/home/or.cohen/thesis_code/RiemannianDOA_proj/Exp_N_y2025-m07-d25_20-49-58_indp' + '/results.pkl'
    with open(filepath_results_file, 'rb') as f:
        results = pickle.load(f)
    # %%
    for i_config in range(len(results)):
        print(f"-------------- Config {i_config}:")
        print(results[i_config][0]["config"])
    # %%
    algo_list = get_algo_dict_list()
    i_config = 0
    i_mc = 10
    i_algo = len(algo_list) - 1  # Last algo
    print(results[i_config][0]["config"])
    ax = display_power_spectrum(results[i_config][i_mc]["config"], results[i_config][i_mc]["p_vec_list"])

    doas = results[i_config][i_mc]["config"]["doa"]
    power_doa_db = results[i_config][i_mc]["config"]["power_doa_db"]
    ax.set_xlim([np.min(doas)-10, np.max(doas)+10])
    ax.set_ylim([-20, np.max(power_doa_db)+3])
    
    # %%
    
    

# %%
def display_power_spectrum_tmp():
    # %%
    config_dict = create_config(
        m=12, snr=20, N=60, power_doa_db=np.array([0,0]), doa=np.array([35, 50]), cohr_flag=False,
    )
    result= run_single_mc_iteration(
        i_mc=0,
        config=config_dict,
        algo_list=list(get_algo_dict_list().keys()))

    ax = display_power_spectrum(result["config"], result["p_vec_list"])

    # doas = result["config"]["doa"]
    # power_doa_db = result["config"]["power_doa_db"]
    # ax.set_xlim([np.min(doas)-10, np.max(doas)+10])
    # ax.set_ylim([-20, np.max(power_doa_db)+3])
    # %%

if __name__ == "__main__":
    example_display_power_spectrum()
    
# %%
