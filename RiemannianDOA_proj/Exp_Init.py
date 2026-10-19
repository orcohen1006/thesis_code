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

def exp_Init(doa: np.ndarray = np.array([35.0, 43.0, 51.0]), power_doa_db: np.ndarray = np.array([0, 0, -5]),
                  N=50, snr = 0, M=12,
                  cohr_flag: bool = False, basedir:str = '') -> None:
    utils.globalParams = GlobalParms()  # reset global params to default values
    utils.globalParams.DO_STORE_CONVERGENCE = True  # Set to True to store convergence data for each algorithm
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    str_indp_cohr = 'cohr' if cohr_flag else 'indp'
    name_results_dir = f'Exp_Init_{timestamp}_{str_indp_cohr}_N_{N}_M_{M}'
    name_results_dir = os.path.join(basedir, name_results_dir)
    path_results_dir = os.path.abspath(name_results_dir)
    print(f"Results will be saved in: {path_results_dir}")
    if not os.path.exists(path_results_dir):
        os.makedirs(path_results_dir)
    # %%
    num_mc = NUM_MC
    list_method_for_p_init =["PER", "MVDR", "LE_ss"]
    num_configs = len(list_method_for_p_init)
    config_list = []
    for i in range(num_configs):
        curr_config = create_config(
                m=M, snr=snr, N=N, 
                power_doa_db=power_doa_db,
                doa=doa,
                cohr_flag=cohr_flag,                
                )
        curr_config["method_for_p_init"] = list_method_for_p_init[i]
        config_list.append(curr_config)
    # %% Run the configurations
    results = RunDoaConfigsPBS(path_results_dir, config_list, num_mc)
    # %%
    results, algos_error_data = analyze_algo_errors(results)
    # 
    parameter_name = r'Method for $p_{init}$'
    fig_combined = plt.figure(figsize=(8, 8))
    ax1 = fig_combined.add_subplot(2, 1, 1)
    ax2 = fig_combined.add_subplot(2, 1, 2)
    plot_doa_errors(algos_error_data, parameter_name, "", list_method_for_p_init, plot_on_ax=ax1, do_legend=True)
    plot_power_errors(algos_error_data, parameter_name, "", list_method_for_p_init,plot_on_ax=ax2, do_legend=False)

    fig_init_loss, fig_init_rel_change = plot_initialization_convergence(results, list_method_for_p_init)
    fig_init_doa_diff, _, _, df_doa_table = plot_initialization_paired_difference(
        results, list_method_for_p_init, algo_name="JBLD", plot_type="cdf", metric="doa")
    # %%
    # %%
    experiment_configs_string_to_file(num_mc=num_mc, config_list=config_list, directory=path_results_dir)
    str_desc_name = os.path.basename(name_results_dir)
    save_figure(fig_combined, path_results_dir, str_desc_name+ "_Combined")
    save_figure(fig_init_loss, path_results_dir, str_desc_name+ "_InitConvergenceLoss")
    save_figure(fig_init_rel_change, path_results_dir, str_desc_name+ "_InitConvergenceRelChange")
    save_figure(fig_init_doa_diff, path_results_dir, str_desc_name+ "_InitPairedDoaDiff")
    with open(os.path.join(path_results_dir, str_desc_name + "_InitPairedDoaDiffTable.txt"), "w") as f:
        f.write(df_doa_table.to_latex(float_format="%.1f"))
    plt.close()
    # %%

if __name__ == "__main__":
    # 1/0
    exp_Init()


# %%
