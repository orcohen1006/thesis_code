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

def exp_DeltaSNR(n: int=30, cohr_flag: bool = False, basedir:str = '') -> None:
    
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    str_indp_cohr = 'cohr' if cohr_flag else 'indp'
    name_results_dir = f'Exp_DeltaSNR_{timestamp}_{str_indp_cohr}_N{n}'
    name_results_dir = os.path.join(basedir, name_results_dir)
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
                doa=np.array([30, 40]),
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
    str_desc_name = os.path.basename(name_results_dir)
    fig_doa_errors.savefig(os.path.join(path_results_dir, 'DOA_' + str_desc_name +  '.png'), dpi=300)
    fig_power_errors.savefig(os.path.join(path_results_dir, 'Power_' + str_desc_name +  '.png'), dpi=300)
    fig_prob_detection.savefig(os.path.join(path_results_dir, 'Prob_' + str_desc_name +  '.png'), dpi=300)


if __name__ == "__main__":
    # Example usage
    exp_DeltaSNR(n=30, cohr_flag=False)


# %%
if False:
    # %%
    import numpy as np
    from scipy.linalg import sqrtm, logm, norm

    def AIRM(R1, R2):
        R2inv_sqrt = np.linalg.inv(sqrtm(R2))
        prod = R2inv_sqrt @ R1 @ R2inv_sqrt
        return norm(logm(prod), 'fro')

    M = 12
    N = 20
    # generate a random M signals with N samples, calculate the true covariance matrix (MxM) and the sample covariance matrix (R_p)
    
    


    alphas_vals = np.arange(1, 10, 0.1)
    clean_euclidean_distance = norm(R_true - R_p, 'fro')
    clean_airm_distance = AIRM(R_true, R_p)
    euclidan_distances = np.zeros(len(alphas_vals))
    airm_distances = np.zeros(len(alphas_vals))
    for i, alpha in enumerate(alphas_vals):
        T = np.diag([1,1,1,1,alpha])
        R_obs = T @ R_true @ T
        euclidan_distances[i] = norm(R_obs - R_p, 'fro')
        airm_distances[i] = AIRM(R_obs, R_p)
    # plotting
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(alphas_vals, euclidan_distances, label='Euclidean Distance')
    plt.plot(alphas_vals, airm_distances, label='AIRM Distance')
    plt.xlabel('Alpha')
    plt.ylabel('Distance')
    plt.legend()
# %%
