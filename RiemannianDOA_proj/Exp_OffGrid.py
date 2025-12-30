# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import List, Optional
import utils
from utils import *
from ToolsMC import *

# %%

def exp_OffGrid(doa: np.ndarray = np.array([35, 43, 51]), power_doa_db: np.ndarray = np.array([0, 0, -5]),
                  N=40, M=12,
                  cohr_flag: bool = False, basedir:str = '') -> None:
    
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    str_indp_cohr = 'cohr' if cohr_flag else 'indp'
    name_results_dir = f'Exp_OffGrid_{timestamp}_{str_indp_cohr}_N_{N}_M_{M}'
    name_results_dir = os.path.join(basedir, name_results_dir)
    path_results_dir = os.path.abspath(name_results_dir)
    print(f"Results will be saved in: {path_results_dir}")
    if not os.path.exists(path_results_dir):
        os.makedirs(path_results_dir)
    # %%
    utils.globalParams.GRID_STEP_DEGREES = 1.0
    utils.globalParams.WANTED_ALGO_NAMES = {*utils.globalParams.WANTED_ALGO_NAMES, "ESPRIT"}
    num_mc = NUM_MC
    vec_delta = np.linspace(0, utils.globalParams.GRID_STEP_DEGREES/2, 4)
    # %%
    num_configs = len(vec_delta)
    config_list = []
    for i in range(num_configs):
        config_list.append(
            create_config(
                m=M, snr=0, N=N, 
                power_doa_db=power_doa_db,
                doa=doa+vec_delta[i],
                cohr_flag=cohr_flag,
                )
        )
    # %% Run the configurations
    results = RunDoaConfigsPBS(path_results_dir, config_list, num_mc)
    # %%
    results, algos_error_data = analyze_algo_errors(results)
    #
    fig_doa_errors = plot_doa_errors(algos_error_data, r'$\Delta$', "", vec_delta, normalize_rmse_by_parameter=False, do_ylogscale=False)
    # tmp = plot_doa_boxplots(algos_error_data, vec_snr, parameter_vals_to_show=[-3, 0], do_ylogscale=True)
    #  %%
    fig_power_errors = plot_power_errors(algos_error_data, r'$\Delta$', "", vec_delta, normalize_rmse_by_parameter=False, do_ylogscale=False)
    
    # %%
    experiment_configs_string_to_file(num_mc=num_mc, config_list=config_list, directory=path_results_dir)
    str_desc_name = os.path.basename(name_results_dir)
    save_figure(fig_doa_errors, path_results_dir, str_desc_name+ "_DOA")
    save_figure(fig_power_errors, path_results_dir, str_desc_name+ "_Power")
    
    plt.close()

    utils.globalParams = GlobalParms()  # reset global params to default values
    # %%

if __name__ == "__main__":
    exp_OffGrid(doa=np.array([35.0, 51.0]), power_doa_db=np.array([0, 0]) , N=50, M=12, cohr_flag=False, basedir='')

# %%
