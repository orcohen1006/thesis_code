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

def exp_rho(doa: np.ndarray = np.array([35, 45]), power_doa_db: np.ndarray = np.array([0, 0]),
            basedir:str = '') -> None:

    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    name_results_dir = f'Exp_Rho_{timestamp}'
    name_results_dir = os.path.join(basedir, name_results_dir)
    path_results_dir = os.path.abspath(name_results_dir)
    print(f"Results will be saved in: {path_results_dir}")
    if not os.path.exists(path_results_dir):
        os.makedirs(path_results_dir)
    # %%
    num_mc = 500
    vec_rho = np.arange(0.0, 1.01, 0.2)
    vec_rho[-1] = 0.99
    num_configs = len(vec_rho)
    config_list = []
    for i in range(num_configs):
        config_list.append(
            create_config(
                m=12, snr=0, N=50, 
                power_doa_db=power_doa_db,
                doa=doa,
                cohr_flag=True,
                cohr_coeff=vec_rho[i]
                )
        )
    # %% Run the configurations
    results = RunDoaConfigsPBS(path_results_dir, config_list, num_mc)
    # %%
    results, algos_error_data = analyze_algo_errors(results)
    #
    fig_doa_errors = plot_doa_errors(algos_error_data, r'Coherence Coeff - $\rho$', "", vec_rho, normalize_rmse_by_parameter=False, do_ylogscale=False)
    # 
    fig_power_errors = plot_power_errors(algos_error_data, r'Coherence Coeff - $\rho$', "", vec_rho, normalize_rmse_by_parameter=False)
    # 
    fig_prob_detection = plot_prob_detection(algos_error_data, r'Coherence Coeff - $\rho$', "", vec_rho)
    # %%
    str_desc_name = os.path.basename(name_results_dir)
    fig_doa_errors.savefig(os.path.join(path_results_dir, 'DOA_' + str_desc_name +  '.png'), dpi=300)
    fig_power_errors.savefig(os.path.join(path_results_dir, 'Power_' + str_desc_name +  '.png'), dpi=300)
    fig_prob_detection.savefig(os.path.join(path_results_dir, 'Prob_' + str_desc_name +  '.png'), dpi=300)
    # %%
    experiment_configs_string_to_file(num_mc=num_mc, config_list=config_list, directory=path_results_dir)

if __name__ == "__main__":
    exp_rho(doa=np.array([35.25, 41.25]), power_doa_db=np.array([0, 0]))


# %%
