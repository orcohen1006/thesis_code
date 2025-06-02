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

def exp_M(cohr_flag: bool, basedir:str = '') -> None:
    
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    str_indp_cohr = 'cohr' if cohr_flag else 'indp'
    name_results_dir = f'Exp_M_{timestamp}_{str_indp_cohr}'
    name_results_dir = os.path.join(basedir, name_results_dir)
    path_results_dir = os.path.abspath(name_results_dir)
    print(f"Results will be saved in: {path_results_dir}")
    if not os.path.exists(path_results_dir):
        os.makedirs(path_results_dir)
    # %%
    num_mc = 50
    vec_m = np.array([10, 50, 100, 500])
    num_configs = len(vec_m)
    config_list = []
    for i in range(num_configs):
        config_list.append(
            create_config(
                m=vec_m[i], snr=-20, N=vec_m[i]*3, 
                # power_doa_db=np.array([0, 0, 0, 0, 0, 0]),
                # doa=np.array([35, 35.5, 40, 40.5, 45, 45.5]),
                power_doa_db=np.array([-20, -20]),
                doa=np.array([35,150]),
                cohr_flag=False,
                )
        )
    # %% Run the configurations
    results = RunDoaConfigsPBS(path_results_dir, config_list, num_mc)
    # %%
    results, algos_error_data = analyze_algo_errors(results)
    #
    fig_doa_errors = plot_doa_errors(algos_error_data, r'$M$', "", vec_m, normalize_rmse_by_parameter=False)
    # 
    fig_power_errors = plot_power_errors(algos_error_data, r'$M$', "", vec_m, normalize_rmse_by_parameter=False)
    # 
    fig_prob_detection = plot_prob_detection(algos_error_data, r'$M$', "", vec_m)
    # %%
    import matplotlib.pyplot as plt   
    algo_list = get_algo_dict_list()
 
    fig_runtime = plt.figure()
    plt.xlabel(r'$M$', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    for i_algo,algo_name in enumerate(algo_list.keys()):
        mean_runtime = [np.mean([results[i_config][i_mc]["runtime_list"][i_algo] for i_mc in range(num_mc)]) for i_config in range(num_configs)]
        std_runtime = [np.std([results[i_config][i_mc]["runtime_list"][i_algo] for i_mc in range(num_mc)]) for i_config in range(num_configs)]
        plt.errorbar(vec_m, mean_runtime, yerr=std_runtime, label=algo_name, **algo_list[algo_name])
    plt.legend()

    fig_iterRuntime = plt.figure()
    plt.xlabel(r'$M$', fontsize=12)
    plt.ylabel('Iteration Runtime (seconds)', fontsize=12)
    for i_algo,algo_name in enumerate(algo_list.keys()):
        mean_runtime = [np.mean([results[i_config][i_mc]["runtime_list"][i_algo]/results[i_config][i_mc]["num_iters_list"][i_algo] for i_mc in range(num_mc)]) for i_config in range(num_configs)]
        std_runtime = [np.std([results[i_config][i_mc]["runtime_list"][i_algo]/results[i_config][i_mc]["num_iters_list"][i_algo] for i_mc in range(num_mc)]) for i_config in range(num_configs)]
        plt.errorbar(vec_m, mean_runtime, yerr=std_runtime, label=algo_name, **algo_list[algo_name])
    plt.legend()
    
    fig_numIters = plt.figure()
    plt.xlabel(r'$M$', fontsize=12)
    plt.ylabel('Iterations', fontsize=12)
    for i_algo,algo_name in enumerate(algo_list.keys()):
        mean_iterations = [np.mean([results[i_config][i_mc]["num_iters_list"][i_algo] for i_mc in range(num_mc)]) for i_config in range(num_configs)]
        std_iterations = [np.std([results[i_config][i_mc]["num_iters_list"][i_algo] for i_mc in range(num_mc)]) for i_config in range(num_configs)]
        plt.errorbar(vec_m, mean_iterations, yerr=std_iterations, label=algo_name, **algo_list[algo_name])
    plt.legend()
    # %%
    str_desc_name = os.path.basename(name_results_dir)

    fig_runtime.savefig(os.path.join(path_results_dir, 'Runtime_' + str_desc_name +  '.png'), dpi=300)
    fig_iterRuntime.savefig(os.path.join(path_results_dir, 'IterRuntime_' + str_desc_name +  '.png'), dpi=300)
    fig_numIters.savefig(os.path.join(path_results_dir, 'NumIters_' + str_desc_name +  '.png'), dpi=300)
    fig_doa_errors.savefig(os.path.join(path_results_dir, 'DOA_' + str_desc_name +  '.png'), dpi=300)
    fig_power_errors.savefig(os.path.join(path_results_dir, 'Power_' + str_desc_name +  '.png'), dpi=300)
    fig_prob_detection.savefig(os.path.join(path_results_dir, 'Prob_' + str_desc_name +  '.png'), dpi=300)

if __name__ == "__main__":
    # Example usage
    exp_M(cohr_flag=False)


# %%
