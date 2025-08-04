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
    path_results_dir = '/home/or.cohen/thesis_code/RiemannianDOA_proj/Exp_Rho_y2025-m07-d28_22-11-29'
    with open(path_results_dir + '/results.pkl', 'rb') as f:
        results = pickle.load(f)
    # %%
    for i_config in range(len(results)):
        print(f"-------------- Config {i_config}:")
        print(results[i_config][0]["config"])
    # %%
    algo_list = get_algo_dict_list()
    # i_config = 2
    # i_mc = inds[5] #468 #3
    i_config = 1; i_mc =  inds[31] # 71 # 216 # 253# 3 # 468
    # i_config = 5; i_mc = 6
    # i_config = 3; i_mc = inds[3]
    print(results[i_config][0]["config"])
    ax = display_power_spectrum(results[i_config][i_mc]["config"], results[i_config][i_mc]["p_vec_list"])

    doas = results[i_config][i_mc]["config"]["doa"]
    power_doa_db = results[i_config][i_mc]["config"]["power_doa_db"]
    ax.set_xlim([np.min(doas)-10, np.max(doas)+10])
    ax.set_ylim([-20, np.max(power_doa_db)+3])
    # %%
    plt.gcf().savefig(os.path.join(path_results_dir, 'Power_Spectrum_i_config_' + str(i_config) + '_i_mc_' + str(i_mc) + '.png'), dpi=300)
    # %%
    algo_list = get_algo_dict_list()
    i_config = 1
    
    fig = plt.figure()
    fig.suptitle(f"Config {i_config}")
    sqerr_dict = {}
    for i_algo in range(len(algo_list)):
        name_algo = list(algo_list.keys())[i_algo]
        sqerr_dict[name_algo] = np.array([np.sum(results[i_config][i_mc]["selected_doa_error"][i_algo] ** 2)
                     for i_mc in range(len(results[i_config]))])
        fig.add_subplot(2, 2, i_algo + 1)
        plt.hist(sqerr_dict[name_algo], bins=50)
        plt.title(name_algo + f", Median={np.median(sqerr_dict[name_algo]):.2f}")
    # space out the subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    inds = np.argsort(sqerr_dict["AIRM"] - (sqerr_dict["SAMV"] + sqerr_dict["SPICE"])/2)

    # %%
    
    

# %%
def display_power_spectrum_tmp():
    # %%
    config_dict = create_config(
        m=12, snr=0, N=40, power_doa_db=np.array([0, 0]), doa=np.array([35, 42]), cohr_flag=True, noncircular_coeff=0.0
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
