# %%
import importlib

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import List, Optional

import utils
importlib.reload(utils)
from utils import *

import ToolsMC
importlib.reload(ToolsMC)
from ToolsMC import *

# %%

def exp_N(doa: np.ndarray = np.array([35.25, 43.25, 51.25]), power_doa_db: np.ndarray = np.array([0, 0, -5]),
          snr: float = -3,
          cohr_flag: bool = False, basedir:str = '') -> None:

    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    str_indp_cohr = 'cohr' if cohr_flag else 'indp'
    name_results_dir = f'Exp_N_{timestamp}_{str_indp_cohr}'
    name_results_dir = os.path.join(basedir, name_results_dir)
    path_results_dir = os.path.abspath(name_results_dir)
    print(f"Results will be saved in: {path_results_dir}")
    if not os.path.exists(path_results_dir):
        os.makedirs(path_results_dir)
    # %%
    num_mc = NUM_MC
    m=12
    # vec_n = np.arange(m, 7*m + 1, m)
    vec_n = np.arange(m, 11*m + 1, 2*m)
    num_configs = len(vec_n)
    config_list = []
    for i in range(num_configs):
        config_list.append(
            create_config(
                m=m, snr=snr, N=vec_n[i], 
                power_doa_db=power_doa_db,
                doa=doa,
                cohr_flag=cohr_flag,
                )
        )
    # %% Run the configurations
    results = RunDoaConfigsPBS(path_results_dir, config_list, num_mc)
    # %%
    results, algos_error_data = analyze_algo_errors(results)
    #
    fig_doa_errors = plot_doa_errors(algos_error_data, r'$N/M$', "", vec_n/m, normalize_rmse_by_parameter=False, do_ylogscale=True)
    # %% 
    fig_power_errors = plot_power_errors(algos_error_data, r'$N/M$', "", vec_n/m, normalize_rmse_by_parameter=False)
    # 
    fig_prob_detection = plot_prob_detection(algos_error_data, r'$N/M$', "", vec_n/m)
    #
    fig_qeigvals = plot_Qeigvals(results, r'$N/M$', "", vec_n/m, do_ylogscale=False)
    # %%
    experiment_configs_string_to_file(num_mc=num_mc, config_list=config_list, directory=path_results_dir)
    str_desc_name = os.path.basename(name_results_dir)
    save_figure(fig_doa_errors, path_results_dir, str_desc_name+ "_DOA")
    save_figure(fig_power_errors, path_results_dir, str_desc_name+ "_Power")
    save_figure(fig_prob_detection, path_results_dir, str_desc_name+ "_Prob")
    save_figure(fig_qeigvals, path_results_dir, str_desc_name+ "_Qeigvals")
    # %%

if __name__ == "__main__":
    exp_N()


# %%
