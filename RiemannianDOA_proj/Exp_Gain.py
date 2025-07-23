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

def exp_Gain(cohr_flag: bool, basedir:str = '') -> None:
    
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    str_indp_cohr = 'cohr' if cohr_flag else 'indp'
    name_results_dir = f'Exp_Gain_{timestamp}_{str_indp_cohr}'
    name_results_dir = os.path.join(basedir, name_results_dir)
    path_results_dir = os.path.abspath(name_results_dir)
    print(f"Results will be saved in: {path_results_dir}")
    if not os.path.exists(path_results_dir):
        os.makedirs(path_results_dir)
    # %%
    num_mc = 100
    vec_gain = np.array([0.92, 0.95, 0.98, 1, 1.02, 1.05, 1.08])# p.linspace(start=0.92, stop=1.08, num=6)
    num_configs = len(vec_gain)
    config_list = []
    for i in range(num_configs):
        config_list.append(
            create_config(
                m=12, snr=0, N=30, 
                power_doa_db=np.array([0, 0]),
                doa=np.array([35, 40]),
                cohr_flag=cohr_flag,
                first_sensor_linear_gain=vec_gain[i]  # Gain for the first sensor
                )
        )
    # %% Run the configurations
    results = RunDoaConfigsPBS(path_results_dir, config_list, num_mc)
    # %%
    results, algos_error_data = analyze_algo_errors(results)
    #
    fig_doa_errors = plot_doa_errors(algos_error_data, r'$G$', "()", vec_gain, normalize_rmse_by_parameter=False)

    # 
    fig_power_errors = plot_power_errors(algos_error_data, r'$G$', "()", vec_gain, normalize_rmse_by_parameter=False)
    # 
    fig_prob_detection = plot_prob_detection(algos_error_data, r'$G$', "()", vec_gain)
    # %%
    str_desc_name = os.path.basename(name_results_dir)
    fig_doa_errors.savefig(os.path.join(path_results_dir, 'DOA_' + str_desc_name +  '.png'), dpi=300)
    fig_power_errors.savefig(os.path.join(path_results_dir, 'Power_' + str_desc_name +  '.png'), dpi=300)
    fig_prob_detection.savefig(os.path.join(path_results_dir, 'Prob_' + str_desc_name +  '.png'), dpi=300)
    # %%
    configs_string_to_file(config_list=config_list, directory=path_results_dir)

if __name__ == "__main__":
    # Example usage
    exp_Gain(cohr_flag=False)
