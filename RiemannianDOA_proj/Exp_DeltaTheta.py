# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import List, Optional
from utils import *
from ToolsMC import *
# %%

def exp_DeltaTheta(n: int, cohr_flag: bool, theta0: float = 35, basedir:str = '') -> None:
    
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    str_indp_cohr = 'cohr' if cohr_flag else 'indp'
    name_results_dir = f'Exp_DeltaTheta_{timestamp}_{str_indp_cohr}_N{n}_theta0{theta0}'
    name_results_dir = os.path.join(basedir, name_results_dir)
    path_results_dir = os.path.abspath(name_results_dir)
    print(f"Results will be saved in: {path_results_dir}")
    if not os.path.exists(path_results_dir):
        os.makedirs(path_results_dir)
    # %%
    num_mc = 100
    vec_delta_theta = np.arange(2, 11)
    num_configs = len(vec_delta_theta)
    config_list = []
    for i in range(num_configs):
        config_list.append(
            create_config(
                m=12, snr=0, N=n, 
                power_doa_db=np.array([0, 0]),
                doa=np.array([theta0, theta0+vec_delta_theta[i]]),
                cohr_flag=cohr_flag,
                )
        )
    # %% Run the configurations
    results = RunDoaConfigsPBS(path_results_dir, config_list, num_mc)
    # %%
    results, algos_error_data = analyze_algo_errors(results)
    #
    fig_doa_errors = plot_doa_errors(algos_error_data, r'$\Delta \theta$', "(degrees)", vec_delta_theta, normalize_rmse_by_parameter=True)
    # 
    fig_power_errors = plot_power_errors(algos_error_data, r'$\Delta \theta$', "(degrees)", vec_delta_theta, normalize_rmse_by_parameter=False)
    # 
    fig_prob_detection = plot_prob_detection(algos_error_data, r'$\Delta \theta$', "(degrees)", vec_delta_theta)
    # %%
    str_desc_name = os.path.basename(name_results_dir)
    fig_doa_errors.savefig(os.path.join(path_results_dir, 'DOA_' + str_desc_name +  '.png'), dpi=300)
    fig_power_errors.savefig(os.path.join(path_results_dir, 'Power_' + str_desc_name +  '.png'), dpi=300)
    fig_prob_detection.savefig(os.path.join(path_results_dir, 'Prob_' + str_desc_name +  '.png'), dpi=300)


if __name__ == "__main__":
    # Example usage
    exp_DeltaTheta(n=30, cohr_flag=False, theta0=99)


# %%
