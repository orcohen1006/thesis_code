# %%
import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
# %matplotlib ipympl

from RunSingleMCIteration import run_single_mc_iteration
from utils import *
import os
import pickle
# 
import utils
import ToolsMC
import importlib
importlib.reload(utils)
importlib.reload(ToolsMC)
from utils import *
from ToolsMC import *
# %%
def example_display_power_spectrum():
    # %%
    path_results_dir = '/home/or.cohen/thesis_code/RiemannianDOA_proj/run_exp_y2025-m08-d20_23-01-43/Exp_N_y2025-m08-d20_23-01-43_indp'
    with open(path_results_dir + '/results.pkl', 'rb') as f:
        results = pickle.load(f)
    # %%
    for i_config in range(len(results)):
        print(f"-------------- Config {i_config}:")
        print(results[i_config][0]["config"])
    # %%
    algo_list = get_algo_dict_list()
    # i_config = 2
    # i_mc = inds[5] #468 #3
    # i_config = 1; i_mc =  inds[31] # 71 # 216 # 253# 3 # 468
    # i_config = 5; i_mc = 6
    # i_config = 3; i_mc = inds[3]
    # i_config = 2; i_mc = 14
    i_config = 1; i_mc = 0

    print(results[i_config][0]["config"])
    ax = display_power_spectrum(results[i_config][i_mc]["config"], results[i_config][i_mc]["p_vec_list"])

    doas = results[i_config][i_mc]["config"]["doa"]
    power_doa_db = results[i_config][i_mc]["config"]["power_doa_db"]
    ax.set_xlim([np.min(doas)-10, np.max(doas)+10])
    ax.set_ylim([-20, np.max(power_doa_db)+3])
    # %%
    plt.gcf().savefig(os.path.join(path_results_dir, 'Power_Spectrum_i_config_' + str(i_config) + '_i_mc_' + str(i_mc) + '.png'), dpi=300)
    # %%
    algo_list = get_algo_dict_list()
    i_config = 1
    
    fig = plt.figure()
    fig.suptitle(f"Config {i_config}")
    sqerr_dict = {}
    for i_algo in range(len(algo_list)):
        name_algo = list(algo_list.keys())[i_algo]
        sqerr_dict[name_algo] = np.array([np.sum(results[i_config][i_mc]["selected_doa_error"][i_algo] ** 2)
                     for i_mc in range(len(results[i_config]))])
        fig.add_subplot(2, 2, i_algo + 1)
        plt.hist(sqerr_dict[name_algo], bins=50)
        plt.title(name_algo + f", Median={np.median(sqerr_dict[name_algo]):.2f}")
    # space out the subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    inds = np.argsort(sqerr_dict["AIRM"] - (sqerr_dict["SAMV"] + sqerr_dict["SPICE"])/2)

    # %%
    path_fig = '/home/or.cohen/thesis_code/RiemannianDOA_proj/TestHyperparams_y2025-m08-d19_20-14-13/JBLD_cccp_adam_learning_rate_comparison.pkl'
    with open(path_fig, 'rb') as f:
        my_fig = pickle.load(f)
    plt.show()
    # %%
    fig = plt.gcf()
    curr_path_results_dir = os.path.dirname(path_fig)
    filename_no_ext = os.path.basename(path_fig).replace('.pkl', '')
    save_figure(fig, curr_path_results_dir, filename_no_ext)


    # %%
    algo_list = get_algo_dict_list()
    num_configs = len(results)
    num_mc = len(results[0])
    num_algos = len(algo_list)
    num_iters_mat = np.zeros((num_configs, num_mc, num_algos))
    runtime_mat = np.zeros((num_configs, num_mc, num_algos))
    for i_config in range(len(results)):
        num_iters_mat[i_config,:,:] = np.array([results[i_config][i_mc]["num_iters_list"] for i_mc in range(len(results[i_config]))])
        runtime_mat[i_config,:,:] = np.array([results[i_config][i_mc]["runtime_list"] for i_mc in range(len(results[i_config]))])


    algo_names = list(algo_list.keys())
    # --- Plot: Number of Iterations ---
    fig_iters, axes_iters = plt.subplots(1, num_algos, figsize=(4 * num_algos, 4), sharey=True, sharex=True)
    fig_iters.suptitle("Number of Iterations per Configuration")

    for a in range(num_algos):
        data = [num_iters_mat[c, :, a] for c in range(num_configs)]  # list of arrays, each of shape (num_mc,)
        axes_iters[a].boxplot(data, showfliers=False)
        axes_iters[a].set_title(algo_names[a])
        axes_iters[a].set_xlabel("Config Index")
        axes_iters[a].set_xticks(range(1, num_configs + 1))
        if a == 0:
            axes_iters[a].set_ylabel("Num Iters")
        plt.tight_layout()


    # --- Plot: Runtime ---
    fig_runtime, axes_runtime = plt.subplots(1, num_algos, figsize=(4 * num_algos, 4), sharey=True, sharex=True)
    fig_runtime.suptitle("Runtime per Configuration")

    for a in range(num_algos):
        data = [runtime_mat[c, :, a] for c in range(num_configs)]
        axes_runtime[a].boxplot(data, showfliers=False)
        axes_runtime[a].set_title(algo_names[a])
        axes_runtime[a].set_xlabel("Config Index")
        axes_runtime[a].set_xticks(range(1, num_configs + 1))
        if a == 0:
            axes_runtime[a].set_ylabel("Runtime [s]")
        plt.tight_layout()



    iter_runtime_mat = runtime_mat / num_iters_mat  # Shape: (num_configs, num_mc, num_algos)
    fig_iterationruntime, axes_iterationruntime = plt.subplots(1, num_algos, figsize=(4 * num_algos, 4), sharey=True, sharex=True)
    fig_iterationruntime.suptitle("Iteration Runtime per Configuration")
    for a in range(num_algos):
        data = [iter_runtime_mat[c, :, a] for c in range(num_configs)]

        axes_iterationruntime[a].boxplot(data, showfliers=False)
        axes_iterationruntime[a].set_title(algo_names[a])
        axes_iterationruntime[a].set_xlabel("Config Index")
        axes_iterationruntime[a].set_xticks(range(1, num_configs + 1))
        if a == 0:
            axes_iterationruntime[a].set_ylabel("Runtime [s]")
        plt.tight_layout()

    plt.tight_layout()
    
    plt.show()
    # %%

    def print_table(table_name, mean_mat, std_mat):
        print("================   " + table_name + "   ================")
        header = ["Config {}".format(i+1) for i in range(num_configs)]
        print("{:<15}".format("Algorithm"), end="")
        for h in header:
            print("{:>20}".format(h), end="")
        print()

        # Print rows
        for alg_idx in range(num_algos):
            print("{:<15}".format(algo_names[alg_idx]), end="")
            for cfg_idx in range(num_configs):
                mean = mean_mat[alg_idx, cfg_idx]
                std = std_mat[alg_idx, cfg_idx]
                print("{:>20}".format(f"{mean:.4f} ± {std:.4f}"), end="")
            print()
        print("==========================================================")

    print_table("Runtime", runtime_mat.mean(axis=1).T, runtime_mat.std(axis=1).T)
    num_hundreds_iters_mat = num_iters_mat / 100
    print_table("Num Iters (hundreds)", num_hundreds_iters_mat.mean(axis=1).T, num_hundreds_iters_mat.std(axis=1).T)
    print_table("Iteration Runtime", iter_runtime_mat.mean(axis=1).T, iter_runtime_mat.std(axis=1).T)

# %%
def display_power_spectrum_tmp():
    # %%
    config_dict = create_config(
        m=12, snr=0, N=40, power_doa_db=np.array([0, 0]), doa=np.array([35, 42]), cohr_flag=True, noncircular_coeff=0.0
    )
    
    algo_list = get_algo_dict_list()
    algo_list = {k: v for k, v in algo_list.items() if k in ['AIRM', 'JBLD']}
    result= run_single_mc_iteration(
        i_mc=0,
        config=config_dict,
        algo_list=list(algo_list.keys()))

    ax = display_power_spectrum(result["config"], result["p_vec_list"], algo_list=algo_list)

    # doas = result["config"]["doa"]
    # power_doa_db = result["config"]["power_doa_db"]
    # ax.set_xlim([np.min(doas)-10, np.max(doas)+10])
    # ax.set_ylim([-20, np.max(power_doa_db)+3])
    # %%

if __name__ == "__main__":
    example_display_power_spectrum()
    
# %%
