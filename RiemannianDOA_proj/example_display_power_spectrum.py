# %%
import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
%matplotlib ipympl

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
    path_results_dir = '/home/or.cohen/thesis_code/RiemannianDOA_proj/zRunExpMPM_y2026-m06-d30_12-22-03/Exp_power_doa_interf_db_y2026-m06-d30_12-22-58'
    name_results_dir = os.path.basename(path_results_dir)
    with open(path_results_dir + '/results.pkl', 'rb') as f:
        results = pickle.load(f)
    # %%
    m=12
    vec_n = np.arange(m, 7*m + 1, m)
    vec_snr = np.arange(-4.5, 4.5 + 1, 1.5)
    # vec_m = np.array([10,100])
    tmp = [[item for sublist in results for item in sublist]]
    tmp = [results[3]]
    plot_iteration_and_runtime_boxplot(tmp, np.array([1]), 'tmp', DO_BOXPLOT=True, logscale_y=False)
    1+1
    # %%
    for i_config in range(len(results)):
        print(f"-------------- Config {i_config}:")
        print(results[i_config][0]["config"])
    # %%
    i_config = 2; 
    range_jj = range(10, 11)
    for jj in range_jj:
        # 
        # plt.close('all')
        algo_list = get_algo_dict_list()
        # i_config = 2; i_mc = 29 #23 #8 #5 #4
        # i_config = 4; i_mc = inds[16]
        i_mc = inds[jj]
        ax1, ax2 = None, None
        print(results[i_config][0]["config"])
        algo_ids_group1 = [0,1]
        algo_ids_group2 = [2,3,4]
        algos_group1 = {k: v for i, (k, v) in enumerate(algo_list.items()) if i in algo_ids_group1}
        algos_group2 = {k: v for i, (k, v) in enumerate(algo_list.items()) if i in algo_ids_group2}
        p_vec_list_group1 = [results[i_config][i_mc]["p_vec_list"][i] for i in algo_ids_group1]
        p_vec_list_group2 = [results[i_config][i_mc]["p_vec_list"][i] for i in algo_ids_group2]
        # fig_spec = plt.figure(figsize=(10, 5))
        # ax1 = fig_spec.add_subplot(1, 2, 1)
        # ax2 = fig_spec.add_subplot(1, 2, 2)
        
        fig_spec = plt.figure(figsize=(8, 8))
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
        if len(range_jj) > 1:
            fig_spec.suptitle(f"Config {i_config}, MC Iteration {i_mc}, jj={jj}")
    # %%
    # plt.gcf().savefig(os.path.join(path_results_dir, 'Power_Spectrum_i_config_' + str(i_config) + '_i_mc_' + str(i_mc) + '.png'), dpi=300)
    save_figure(fig_spec, path_results_dir, f'Power_Spectrum_i_config_{i_config}_i_mc_ {i_mc}')
    # %%
    algo_list = get_algo_dict_list()
    i_config = 2
    
    fig = plt.figure()
    fig.suptitle(f"Config {i_config}")
    sqerr_dict = {}
    for i_algo in range(len(algo_list)):
        name_algo = list(algo_list.keys())[i_algo]
        sqerr_dict[name_algo] = np.array([np.sum(results[i_config][i_mc]["selected_doa_error"][i_algo] ** 2)
                     for i_mc in range(len(results[i_config]))])
        fig.add_subplot(2, 3, i_algo + 1)
        plt.hist(sqerr_dict[name_algo], bins=50)
        plt.title(name_algo + f", Median={np.median(sqerr_dict[name_algo]):.2f}")
    # space out the subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
    inds = np.argsort(sqerr_dict["AIRM"] - (sqerr_dict["SAMV"] + sqerr_dict["SPICE"])/2)

    # %%
    path_fig = '/home/or.cohen/thesis_code/RiemannianDOA_proj/run_exp_y2026-m01-d13_15-59-09/Exp_OffGrid_y2026-m01-d13_16-28-38_indp_N_50_M_12_SNR_0/Exp_OffGrid_y2026-m01-d13_16-28-38_indp_N_50_M_12_SNR_0_DOA.pkl'
    with open(path_fig, 'rb') as f:
        my_fig = pickle.load(f)
    plt.show()
    # %%
    fig = plt.gcf()
    curr_path_results_dir = os.path.dirname(path_fig)
    filename_no_ext = os.path.basename(path_fig).replace('.pkl', '')
    save_figure(fig, curr_path_results_dir, filename_no_ext)
    # %%
    path_fig_pkl_prefix = '/home/or.cohen/thesis_code/RiemannianDOA_proj/run_exp_y2026-m01-d13_15-59-09/Exp_SNR_y2026-m01-d13_15-59-09_indp_N_50_M_12/Exp_SNR_y2026-m01-d13_15-59-09_indp_N_50_M_12_'
    with open(path_fig_pkl_prefix + '_DOA.pkl', 'rb') as f:
        fig_DOA = pickle.load(f)
    plt.show()
    with open(path_fig_pkl_prefix + '_Power.pkl', 'rb') as f:
        fig_Power = pickle.load(f)    
    plt.show()

    # now i want to combine the two figures into one figure with two subplots. only the first subplot will have the legend
    # the combined figure should be the size of the two figures stacked horizontally.
    # keep all the original figures properties and plots exactly the same, except for the legend.
    fig_combined = plt.figure(figsize=(fig_DOA.get_size_inches()[0], fig_DOA.get_size_inches()[1] + fig_Power.get_size_inches()[1]))
    ax1 = fig_combined.add_subplot(1, 2, 1)
    ax2 = fig_combined.add_subplot(1, 2, 2)
        
    # %%
    tmp = plot_iteration_and_runtime_boxplot(results, vec_snr, 'SNR', logscale_y=False)
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

import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
%matplotlib ipympl


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

M = 12
L = 4
N = int(2.5*M*L)
snr = 5
doa_desired=np.array([70.0, 135.0])
power_doa_desired_db=np.array([0.0, 0.0])

doa_interf = np.array([63.0, 110.0])
power_doa_interf_db = np.array([0.0, 0.0]) + 4

# doa_interf = np.array([])
# power_doa_interf_db = np.array([])

config = create_config(
    m=M, snr=snr, N=N, power_doa_db=power_doa_desired_db, doa=doa_desired, 
    power_doa_interf_db=power_doa_interf_db, doa_interf=doa_interf, L=L)

algo_list = define_all_algo_dict_list()
result= run_single_mc_iteration(
    i_mc= 0,
    config=config,
    algo_list=list(algo_list.keys()),
    do_save_G_tensor_results= True)

ax = display_power_spectrum(result["config"], result["p_vec_list"], algo_list=algo_list,
                            normalize_power=NormalizePowerType.NONE, do_legend=False, do_colorbar=True, 
                            algos_to_leave_out = ["MinSpectrum","qstar"])
fig_q_spectrum = plt.gcf()
# save_figure(fig_q_spectrum, ".", "q_spectrum_example")


# ax = display_power_spectrum(result["config"], result["list_p_vec_for_G_tensor"], algo_list=get_segements_dict_list(config["L"]),
#                             normalize_power=NormalizePowerType.NONE, do_legend=True, do_colorbar=False)
# fig_segments_spectrum = plt.gcf()


# create figure with 2 subplots:
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
# display_power_spectrum(result["config"], result["list_p_vec_for_G_tensor"][0], algo_list=algo_list,
#                             normalize_power=NormalizePowerType.NONE, ax=ax1)

# %%
from datetime import datetime
path_dir = "Figs_Spectrum_" + datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
os.makedirs(path_dir)
save_figure(fig_q_spectrum, path_dir, name="q_spectrum")
save_figure(fig_segments_spectrum, path_dir, name="segments_spectrum")


# %%

import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
%matplotlib ipympl

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

M = 12
L = 4
W = 1*M
N = int(W*L)
doa_desired=np.array([70.0, 135.0])
power_doa_desired_db=np.array([0.0, 0.0])

doa_interf = np.array([65.0, 110.0])
power_doa_interf_db = np.array([0.0, 0.0]) + 4


# doa_interf = np.array([40.0, 60.0, 120.0, 150.0])
# power_doa_interf_db = np.array([3.0, 3.0, 3.0, 3.0])


# doa_interf = np.array([])
# power_doa_interf_db = np.array([])

config = create_config(
    m=M, snr=10, N=N, power_doa_db=power_doa_desired_db, doa=doa_desired, 
    power_doa_interf_db=power_doa_interf_db, doa_interf=doa_interf, L=L)

configs_to_display = []

# N_vals = [int(1*M*L), int(2*M*L), int(3*M*L)]
# snr_vals = [10, 0, -5]

# N_vals = [int(1.5*M*L)]
# snr_vals = [5, 0, -2.5, -5]

# N_vals = [int(1.5*M*L), int(3*M*L)]
# N_vals = [int(1.2*M*L), int(2*M*L), int(3*M*L), int(6*M*L)]
N_vals = [int(2*M*L), int(10*M*L)]
snr_vals = [10, 0, -2.5, -5]

for N in N_vals:
    for snr in snr_vals:
        currconfig = config.copy()
        currconfig["N"] = N
        currconfig["snr"] = snr
        configs_to_display.append(currconfig)

 
fig, axes = plt.subplots(len(N_vals), len(snr_vals), figsize=(len(snr_vals)*4, len(N_vals)*4))
axes = axes.flatten()

for i, currconfig in enumerate(configs_to_display):
    algo_list = define_all_algo_dict_list()
    result = run_single_mc_iteration(
        i_mc=0,
        config=currconfig,
        algo_list=list(algo_list.keys()))

    ax = display_power_spectrum(result["config"], result["p_vec_list"], algo_list=algo_list, ax=axes[i],
                            do_colorbar=False,
                            normalize_power=NormalizePowerType.NONE, 
                            algos_to_leave_out = ["OptimalNI","OptimalCMPM"])
    axes[i].set_title(f"W/M={currconfig['N']/M/config['L']}, SNR={currconfig['snr']}")

fig.suptitle("Power Spectrum Analysis")
plt.tight_layout()
plt.show()




# %%
def Foo(x):
    x = x / 2
    print(x)

x = np.array([10.0,20.0,30.0])
print(x)
Foo(x)
print(x)
# %%

# %%
import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
%matplotlib ipympl

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
path_results_dir = '/home/or.cohen/thesis_code/RiemannianDOA_proj/zRunExpMPM_y2026-m07-d11_21-47-44/Exp_power_doa_interf_db_y2026-m07-d11_21-48-38'
path_results_dir = '/home/or.cohen/thesis_code/RiemannianDOA_proj/zRunExpMPM_y2026-m07-d11_21-47-44/Exp_snr_y2026-m07-d11_21-47-44'
name_results_dir = os.path.basename(path_results_dir)
with open(path_results_dir + '/results.pkl', 'rb') as f:
    results = pickle.load(f)
# %%
from mpm import *
num_configs = len(results)
num_mc = len(results[0])
costMatrix = np.zeros(shape=(num_configs, num_mc))
for i_config in range(num_configs):
    for i_mc in range(num_mc):
        G_tensor = results[i_config][i_mc]["G_tensor"]
        L = G_tensor.shape[0]
        X = karcher_mean(G_tensor,  epsilon=1e-4, max_iter= 10, delta=utils.globalParams.DELTA_FOR_DIAG_LOADING)
        # wX, VX = eigh_clip(X)
        # _, invsqrtX = sqrt_invsqrt_from_eigh(wX, VX)
        # cost = np.sum([normsquared_logm_invsqrtX_Y_invsqrtX(invsqrtX, G_tensor[l,:,:]) for l in range(L)])
        cost = np.mean([riemann_dist2(X, G_tensor[l,:,:]) for l in range(L)])
        costMatrix[i_config,i_mc] = cost
        
        print(f"i_config={i_config}, i_mc={i_mc}")
# %%
parameter_values = scanned_param_vals
if type(parameter_values[0]) == np.ndarray:
        parameter_values = np.array([param[0] for param in parameter_values])


M, W = results[0][0]["config"]["m"], results[0][0]["config"]["N"]/results[0][0]["config"]["L"]
normalizedCostMatrix = costMatrix * (W / M**2)
normalizedCostMatrix = np.sqrt(normalizedCostMatrix)
mean_cost_vec = np.mean(normalizedCostMatrix, axis=1)
qlow_cost_vec = np.percentile(normalizedCostMatrix, 25, axis=1)
qhigh_cost_vec = np.percentile(normalizedCostMatrix, 75, axis=1)

fig = plt.figure()
ax = plt.gca()
ax.plot(parameter_values, mean_cost_vec)
ax.fill_between(parameter_values, qlow_cost_vec, qhigh_cost_vec, alpha=0.20, linewidth=0.5)



def get_qstar(normalized_dispersion):
    normalized_dispersion_min = 1.0
    normalized_dispersion_max = 1.05
    normalized_dispersion_clipped = np.clip(normalized_dispersion, normalized_dispersion_min, normalized_dispersion_max)
    dtilde = (normalized_dispersion_clipped - normalized_dispersion_min) / (normalized_dispersion_max - normalized_dispersion_min)
    qstar = 1 - 2*dtilde
    # qstar = 1 - 2/(1 + np.exp(-10*(dtilde - 0.5)))
    return qstar

qstar_mean = get_qstar(mean_cost_vec)
qstar_low = get_qstar(qlow_cost_vec)
qstar_high = get_qstar(qhigh_cost_vec)
ax.plot(parameter_values, qstar_mean)
ax.fill_between(parameter_values, qstar_low, qstar_high, alpha=0.20, linewidth=0.5)

plt.show()




# %% --------------

from mpm import *
num_configs = len(results)
num_mc = 100 #len(results[0])
q_vals = np.arange(1, -1.01, -0.25)
resMatrix = np.zeros(shape=(len(q_vals)-1, num_configs, num_mc))
for i_config in range(num_configs):
    for i_mc in range(num_mc):
        G_tensor = results[i_config][i_mc]["G_tensor"]
        prev_G_q = None
        dists = []
        for q in q_vals:
            G_q = mpm(G_tensor, q, delta=globalParams.DELTA_FOR_DIAG_LOADING)
            if not (prev_G_q is None):
                dists.append(riemann_dist(G_q, prev_G_q))
            prev_G_q = G_q
        
        dists = np.array(dists)
        path_length = dists.sum()
        normalized_dists = dists / path_length
        resMatrix[:, i_config, i_mc] = normalized_dists
        print(f"i_config={i_config}, i_mc={i_mc}")

# %%
fig = plt.figure()
ax = plt.gca()
for i_jump in range(len(q_vals)-1):
    currMatrix = resMatrix[i_jump,:,:]
    mean_cost_vec = np.mean(currMatrix, axis=1)
    qlow_cost_vec = np.percentile(currMatrix, 25, axis=1)
    qhigh_cost_vec = np.percentile(currMatrix, 75, axis=1)
    ax.plot(range(num_configs), mean_cost_vec, label=f"{q_vals[i_jump]} to {q_vals[i_jump+1]}")
    ax.fill_between(range(num_configs), qlow_cost_vec, qhigh_cost_vec, alpha=0.20, linewidth=0.5)

ax.legend()