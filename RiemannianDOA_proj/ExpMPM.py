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

def exp(config, scanned_param_name, scanned_param_vals, basedir:str = '') -> None:
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
    fig_doa_errors = plot_doa_errors(algos_error_data, f'{scanned_param_name}', "", scanned_param_vals, normalize_rmse_by_parameter=False, do_ylogscale=False, do_legend=False, do_colorbar=True)
    fig_sir = plot_sir(results, f'{scanned_param_name}', "", scanned_param_vals, do_ylogscale=False, do_legend=False, do_colorbar=True)
    fig_sir_per_config = plot_sir_per_config(results)
    # %%
    experiment_configs_string_to_file(num_mc=num_mc, config_list=config_list, directory=path_results_dir)
    str_desc_name = os.path.basename(name_results_dir)
    save_figure(fig_doa_errors, path_results_dir, str_desc_name+ "_DOA")
    save_figure(fig_sir, path_results_dir, str_desc_name+ "_SIR")
    save_figure(fig_sir_per_config, path_results_dir, str_desc_name+ "_SIR_per_config")
    plt.close()
    # %%


# %% ---------------------------------------
#    ---------------------------------------
#    ---------------------------------------
#    ---------------------------------------



# %% ---------------------------------------
def run_on_N_no_interf(basedir):
    # %% 
    M = 12
    L = 2
    N = int(M*2*L)
    doa_desired=np.array([70.0])
    power_doa_desired_db=np.array([0])


    doa_interf = np.array([])
    power_doa_interf_db = np.array([])

    config = create_config(
        m=M, snr=10, N=N, power_doa_db=power_doa_desired_db, doa=doa_desired, 
        power_doa_interf_db=power_doa_interf_db, doa_interf=doa_interf, L=L)
    
    scanned_param_name = 'N'
    scanned_param_vals = np.arange(int(M*1*L), int(M*3*L) + 1, int(M*0.5*L))
    # %% 
    exp(config, scanned_param_name, scanned_param_vals, basedir=basedir)

# %% ---------------------------------------
def run_on_SNR(basedir):
    # %% 
    M = 12
    L = 2
    N = int(M*3*L)
    doa_desired=np.array([70.0])
    power_doa_desired_db=np.array([0])

    doa_interf = np.array([50.0, 120.0])
    power_doa_interf_db = np.array([5.0, 5.0])

    config = create_config(
        m=M, snr=10, N=N, power_doa_db=power_doa_desired_db, doa=doa_desired, 
        power_doa_interf_db=power_doa_interf_db, doa_interf=doa_interf, L=L)
    scanned_param_name = 'snr'
    scanned_param_vals = np.array([0, 5, 10])
    # %% 
    exp(config, scanned_param_name, scanned_param_vals, basedir=basedir) 
   
def run_on_N(basedir):
    # %% 
    M = 12
    L = 2
    N = int(M*3*L)
    doa_desired=np.array([70.0])
    power_doa_desired_db=np.array([0])

    doa_interf = np.array([50.0, 120.0])
    power_doa_interf_db = np.array([5.0, 5.0])

    config = create_config(
        m=M, snr=5, N=N, power_doa_db=power_doa_desired_db, doa=doa_desired, 
        power_doa_interf_db=power_doa_interf_db, doa_interf=doa_interf, L=L)
    scanned_param_name = 'N'
    scanned_param_vals = np.array([int(M*1*L), int(M*3*L)])
    # %% 
    exp(config, scanned_param_name, scanned_param_vals, basedir=basedir) 


if __name__ == "__main__":

    # 1/0 

    t0_overall = time.time()
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    basedir = f'zRunExpMPM_{timestamp}'
    print(f"run exp basedir: {basedir}")
    if not os.path.exists(basedir):
        os.makedirs(basedir) 
    # %% ---------------------------------------
    run_on_N_no_interf(basedir)
    # %% ---------------------------------------
    run_on_SNR(basedir)        
    # %% ---------------------------------------
    run_on_N(basedir)        





    # %% ---------------------------------------
    print(f'Total Running Time: {time.time() - t0_overall} sec.')
    # %% ---------------------------------------
    git_commit_and_push(commit_message=basedir)

