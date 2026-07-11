# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import List, Optional
from commit_repo_git import git_commit_and_push
from utils import *
from ToolsMC import *

# %%

def exp(config, scanned_param_name, scanned_param_vals, scanned_param_string_to_display, basedir:str = '') -> None:
    utils.globalParams = GlobalParms()  # reset global params to default values
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    name_results_dir = f'Exp_{scanned_param_name}_{timestamp}'
    name_results_dir = os.path.join(basedir, name_results_dir)
    path_results_dir = os.path.abspath(name_results_dir)
    print(f"Results will be saved in: {path_results_dir}")
    if not os.path.exists(path_results_dir):
        os.makedirs(path_results_dir)
    # %%
    num_mc = NUM_MC
    num_configs = len(scanned_param_vals)
    config_list = [{**config, scanned_param_name: v} for v in scanned_param_vals]
    # %% Run the configurations
    results = RunDoaConfigsPBS(path_results_dir, config_list, num_mc)
    # results = RunDoaConfigsLocally(path_results_dir, config_list, num_mc)
    # %%
    results, algos_error_data = analyze_algo_errors(results)
    # %%
    fig_doa_errors = plot_doa_errors(algos_error_data, scanned_param_string_to_display, "", scanned_param_vals, normalize_rmse_by_parameter=False, do_ylogscale=True, do_legend=True, do_colorbar=True)
    fig_sir = plot_sir(results, scanned_param_string_to_display, "", scanned_param_vals, do_ylogscale=True, do_legend=True, do_colorbar=True)
    fig_sir_per_config = plot_sir_per_config(results)
    # fig_eigsG = plot_eigsG_per_config(results, do_ylogscale=False)
    # fig_directivity = plot_directivity(results, f'{scanned_param_name}', "", scanned_param_vals, do_ylogscale=True, do_legend=False, do_colorbar=True)
    # fig_dnismse = plot_dnismse(results, f'{scanned_param_name}', "", scanned_param_vals, do_ylogscale=False, do_legend=False, do_colorbar=True)
    # fig_inismse = plot_inismse(results, f'{scanned_param_name}', "", scanned_param_vals, do_ylogscale=False, do_legend=False, do_colorbar=True)
    # %%
    experiment_configs_string_to_file(num_mc=num_mc, config_list=config_list, directory=path_results_dir)
    str_desc_name = os.path.basename(name_results_dir)
    save_figure(fig_doa_errors, path_results_dir, str_desc_name+ "_DOA")
    save_figure(fig_sir, path_results_dir, str_desc_name+ "_SIR")
    save_figure(fig_sir_per_config, path_results_dir, str_desc_name+ "_SIR_per_config")
    # save_figure(fig_directivity, path_results_dir, str_desc_name+ "_Directivity")
    # save_figure(fig_dnismse, path_results_dir, str_desc_name+ "_DNISMSE")
    # save_figure(fig_inismse, path_results_dir, str_desc_name+ "_INISMSE")
    plt.close()
    # %%


# %% ---------------------------------------
#    ---------------------------------------
#    ---------------------------------------
#    ---------------------------------------



# %% ---------------------------------------
def run_on_SNR(basedir, config):
    # %% 
    scanned_param_name = 'snr'
    scanned_param_vals = np.arange(-8.0, 8.1, 2.0)
    scanned_param_string_to_display = f'SNR (dB)'
    # %% 
    exp(config, scanned_param_name, scanned_param_vals, scanned_param_string_to_display, basedir=basedir) 
   
def run_on_inputSIR(basedir, config):
    # %% 
    scanned_param_name = 'power_doa_interf_db'
    scanned_param_vals = [np.array([0, 0]) + k for k in np.arange(-10, 10+1, 2)]
    scanned_param_string_to_display = f'Interference Power (dB)'
    # %% 
    exp(config, scanned_param_name, scanned_param_vals, scanned_param_string_to_display, basedir=basedir) 


if __name__ == "__main__":

    # 1/0 

    t0_overall = time.time()
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    basedir = f'zRunExpMPM_{timestamp}'
    print(f"run exp basedir: {basedir}")
    if not os.path.exists(basedir):
        os.makedirs(basedir) 

    # %%
    M = 12
    L = 4
    N = int(2.5*M*L)
    print(f"Running with M={M}, L={L}, N={N}, W=floor({N/L})")
    doa_desired=np.array([70.0, 135.0])
    power_doa_desired_db=np.array([0.0, 0.0])

    doa_interf = np.array([63.0, 110.0])
    power_doa_interf_db = np.array([0, 0]) + 4

    basic_config = create_config(
        m=M, snr=0, N=N, power_doa_db=power_doa_desired_db, doa=doa_desired, 
        power_doa_interf_db=power_doa_interf_db, doa_interf=doa_interf, L=L)
    # %% ---------------------------------------
    run_on_SNR(basedir, basic_config)             
    # %% ---------------------------------------
    run_on_inputSIR(basedir, basic_config) 




    # %% ---------------------------------------
    print(f'Total Running Time: {time.time() - t0_overall} sec.')
    # %% ---------------------------------------
    # git_commit_and_push(commit_message=basedir)

