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

def exp_DeltaSNR(n: int, cohr_flag: bool) -> None:
    
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    str_indp_cohr = 'cohr' if cohr_flag else 'indp'
    name_results_dir = f'Exp_DeltaSNR_{timestamp}_{str_indp_cohr}_N{n}'
    path_results_dir = os.path.abspath(name_results_dir)
    print(f"Results will be saved in: {path_results_dir}")
    if not os.path.exists(path_results_dir):
        os.makedirs(path_results_dir)
    # %%
    num_mc = 100
    vec_delta_snr = np.arange(start=-8, stop=0+1, step=1)
    num_configs = len(vec_delta_snr)
    config_list = []
    for i in range(num_configs):
        config_list.append(
            create_config(
                m=12, snr=0, N=n, 
                power_doa_db=np.array([0, 0+vec_delta_snr[i]]),
                doa=np.array([35, 43]),
                cohr_flag=False,
                )
        )
    # %% Run the configurations
    results = RunDoaConfigsPBS(path_results_dir, config_list, num_mc)
    # %%
    results, algos_error_data = analyze_algo_errors(results)
    #
    fig_doa_errors = plot_doa_errors(algos_error_data, r'$\Delta SNR$', "(dB)", vec_delta_snr, normalize_rmse_by_parameter=False)
    # 
    fig_power_errors = plot_power_errors(algos_error_data, r'$\Delta SNR$', "(dB)", vec_delta_snr, normalize_rmse_by_parameter=False)
    # 
    fig_prob_detection = plot_prob_detection(algos_error_data, r'$\Delta SNR$', "(dB)", vec_delta_snr)
    # %%
    fig_doa_errors.savefig(os.path.join(path_results_dir, 'DOA_' + name_results_dir +  '.png'), dpi=300)
    fig_power_errors.savefig(os.path.join(path_results_dir, 'Power_' + name_results_dir +  '.png'), dpi=300)
    fig_prob_detection.savefig(os.path.join(path_results_dir, 'Prob_' + name_results_dir +  '.png'), dpi=300)

if __name__ == "__main__":
    # Example usage
    exp_DeltaSNR(n=30, cohr_flag=False)


# %%
