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

def exp_SNR(cohr_flag: bool, basedir:str = '', N=40, secondsourcesnr:float = 0.0) -> None:
    
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    str_indp_cohr = 'cohr' if cohr_flag else 'indp'
    name_results_dir = f'Exp_SNR_{timestamp}_{str_indp_cohr}_N_{N}_secondsourcesnr_{secondsourcesnr}'
    name_results_dir = os.path.join(basedir, name_results_dir)
    path_results_dir = os.path.abspath(name_results_dir)
    print(f"Results will be saved in: {path_results_dir}")
    if not os.path.exists(path_results_dir):
        os.makedirs(path_results_dir)
    # %%
    num_mc = 100
    vec_snr = np.arange(-10,10+1, 2.5)
    num_configs = len(vec_snr)
    config_list = []
    for i in range(num_configs):
        config_list.append(
            create_config(
                m=12, snr=vec_snr[i], N=N, 
                power_doa_db=np.array([0, secondsourcesnr]),
                doa=np.array([35, 40]),
                cohr_flag=cohr_flag,
                )
        )
    # %% Run the configurations
    results = RunDoaConfigsPBS(path_results_dir, config_list, num_mc)
    # %%
    results, algos_error_data = analyze_algo_errors(results)
    # 
    #
    fig_doa_errors = plot_doa_errors(algos_error_data, r'$SNR$', "", vec_snr, normalize_rmse_by_parameter=False, do_ylogscale=True)
    #  
    fig_power_errors = plot_power_errors(algos_error_data, r'$SNR$', "", vec_snr, normalize_rmse_by_parameter=False)
    #  
    fig_prob_detection = plot_prob_detection(algos_error_data, r'$SNR$', "", vec_snr)
    #
    i_config = np.where(np.array(vec_snr) == 0)[0][0]
    fig_l0_norm = plot_l0_norm(results[i_config])
    fig_hpbw = plot_hpbw(results[i_config])
    # %%
    str_desc_name = os.path.basename(name_results_dir)
    fig_doa_errors.savefig(os.path.join(path_results_dir, 'DOA_' + str_desc_name +  '.png'), dpi=300)
    fig_power_errors.savefig(os.path.join(path_results_dir, 'Power_' + str_desc_name +  '.png'), dpi=300)
    fig_prob_detection.savefig(os.path.join(path_results_dir, 'Prob_' + str_desc_name +  '.png'), dpi=300)
    fig_l0_norm.savefig(os.path.join(path_results_dir, 'L0_' + str_desc_name +  '.png'), dpi=300)
    fig_hpbw.savefig(os.path.join(path_results_dir, 'HPBW_' + str_desc_name +  '.png'), dpi=300)
    # %%
    experiment_configs_string_to_file(num_mc=num_mc, config_list=config_list, directory=path_results_dir)

if __name__ == "__main__":
    # Example usage
    exp_SNR(cohr_flag=False, secondsourcesnr=-5)


# %%
