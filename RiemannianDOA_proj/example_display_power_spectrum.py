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
plt.close('all')
# %%
def example_display_power_spectrum():
    # %%
    path_results_dir = '/home/or.cohen/thesis_code/RiemannianDOA_proj/run_exp_y2025-m12-d14_01-58-53/Exp_SNR_Large_y2025-m12-d14_01-58-53_indp_N_50_M_12'
    name_results_dir = os.path.basename(path_results_dir)
    with open(path_results_dir + '/results.pkl', 'rb') as f:
        results = pickle.load(f)
    # %%
    m=12
    vec_n = np.arange(m, 7*m + 1, m)
    vec_snr = np.arange(-4.5, 4.5 + 1, 1.5)

    plot_iterruntime_boxplot(results, vec_snr, 'SNR', DO_BOXPLOT=True)
    tmp = 123
    # %%
    for i_config in range(len(results)):
        print(f"-------------- Config {i_config}:")
        print(results[i_config][0]["config"])
    # %%
    plt.close('all')
    algo_list = get_algo_dict_list()
    # i_config = 2; i_mc = 29 #23 #8 #5 #4
    i_config = 4; i_mc = inds[16]
    ax1, ax2 = None, None
    print(results[i_config][0]["config"])
    algo_ids_group1 = [0,1]
    algo_ids_group2 = [2,3]
    algos_group1 = {k: v for i, (k, v) in enumerate(algo_list.items()) if i in algo_ids_group1}
    algos_group2 = {k: v for i, (k, v) in enumerate(algo_list.items()) if i in algo_ids_group2}
    p_vec_list_group1 = [results[i_config][i_mc]["p_vec_list"][i] for i in algo_ids_group1]
    p_vec_list_group2 = [results[i_config][i_mc]["p_vec_list"][i] for i in algo_ids_group2]
    # fig_spec = plt.figure(figsize=(10, 5))
    # ax1 = fig_spec.add_subplot(1, 2, 1)
    # ax2 = fig_spec.add_subplot(1, 2, 2)
    
    fig_spec = plt.figure(figsize=(6, 7))
    ax1 = fig_spec.add_subplot(2, 1, 1)
    ax2 = fig_spec.add_subplot(2, 1, 2)
    
    # fig_spec = plt.figure(figsize=(7, 5))
    # ax1 = plt.gca()
    # ax2 = ax1

    ax1 = display_power_spectrum(results[i_config][i_mc]["config"], p_vec_list_group1, algo_list=algos_group1, ax=ax1)
    ax2 = display_power_spectrum(results[i_config][i_mc]["config"], p_vec_list_group2, algo_list=algos_group2, ax=ax2)

    doas = results[i_config][i_mc]["config"]["doa"]
    power_doa_db = results[i_config][i_mc]["config"]["power_doa_db"]
    DELTA_X = 10
    ax1.set_xlim([np.min(doas)-DELTA_X, np.max(doas)+DELTA_X])
    ax2.set_xlim([np.min(doas)-DELTA_X, np.max(doas)+DELTA_X])
    ax1.set_ylim([-20, np.max(power_doa_db)+3])
    ax2.set_ylim([-20, np.max(power_doa_db)+3])
    # %%
    # plt.gcf().savefig(os.path.join(path_results_dir, 'Power_Spectrum_i_config_' + str(i_config) + '_i_mc_' + str(i_mc) + '.png'), dpi=300)
    save_figure(fig_spec, path_results_dir, f'Power_Spectrum_i_config_{i_config}_i_mc_ {i_mc}')
    # %%
    algo_list = get_algo_dict_list()
    i_config = 4
    
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
    path_fig = '/home/or.cohen/thesis_code/RiemannianDOA_proj/run_exp_y2025-m09-d09_11-43-32/Exp_N_y2025-m09-d09_11-46-35_indp/Exp_N_y2025-m09-d09_11-46-35_indp_DOA.pkl'
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
    print_table("Num Iters", num_iters_mat.mean(axis=1).T, num_iters_mat.std(axis=1).T)
    print_table("Iteration Runtime", iter_runtime_mat.mean(axis=1).T, iter_runtime_mat.std(axis=1).T)

# %%
def display_power_spectrum_tmp():
    # %%
    plt.close('all')
    # doa=np.array([35.0, 43.0, 51.0])
    # power_doa_db=np.array([0, 0, -5])

    # doa=np.array([35.0, 41.0])
    # power_doa_db=np.array([0, 0])

    # config = create_config(
    #     m=12, snr=0, N=20, power_doa_db=power_doa_db, doa=doa, cohr_flag=True, noncircular_coeff=0.0
    # )
    
    
    m = 50
    config = create_config(
        doa=np.array([35.0, 37.0, 39.0]), power_doa_db=np.array([0, 0, -5]) + convert_linear_to_db(12) - convert_linear_to_db(m), N=150, m=m, snr=0 ,
    )
    
    algo_list = get_algo_dict_list(flag_get_all=True)
    # algo_list = {k: v for k, v in algo_list.items() if k in ['AIRM', 'JBLD', 'LE_ss', 'PER', 'ESPRIT']}
    algo_list = {k: v for k, v in algo_list.items() if k in ['SPICE', 'JBLD','LE']}
    # algo_list = {k: v for k, v in algo_list.items() if k in ['JBLD','SPICE']}
    result= run_single_mc_iteration(
        i_mc= 1,
        config=config,
        algo_list=list(algo_list.keys()))
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # R_hat = result["R_hat"]
    # A = get_steering_matrix(get_doa_grid(),R_hat.shape[0])
    # invR_hat_A = np.linalg.solve(R_hat, A)
    # p_MVDR = 1/np.sum(A.conj() * invR_hat_A, axis=0)
    # result["p_vec_list"][0] = p_MVDR
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    ax = display_power_spectrum(result["config"], result["p_vec_list"], algo_list=algo_list)

    doas = result["config"]["doa"]
    power_doa_db = result["config"]["power_doa_db"]
    ax.set_xlim([np.min(doas)-10, np.max(doas)+10])
    ax.set_ylim([-20, np.max(power_doa_db)+3])
    # %%

if __name__ == "__main__":
    example_display_power_spectrum()
    
# %%
def Foo(x):
    x = x / 2
    print(x)

x = np.array([10.0,20.0,30.0])
print(x)
Foo(x)
print(x)
# %%