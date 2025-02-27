import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import List, Optional
# from ComputeAlgosStdErr import compute_algos_std_err
from ComputeAlgosStdErr_parallel import compute_algos_std_err_parallel

def exp_DeltaTheta(n: int, cohr_flag: bool, large_scale_flag: bool) -> None:
    """
    Experiment to evaluate algorithm performance with varying angle separations.
    
    Parameters:
    -----------
    n : int
        Number of samples
    cohr_flag : bool
        Flag indicating if sources are coherent
    large_scale_flag : bool
        Flag indicating if this is a large-scale experiment
    """
    np.random.seed(42)
    flag_save_fig = True
    m = 12
    
    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    str_indp_cohr = 'cohr' if cohr_flag else 'indp'
    results_dir = f'Exp_DeltaTheta_{timestamp}_{str_indp_cohr}_M{m}_N{n}_L{int(large_scale_flag)}'
    
    if not os.path.exists(results_dir) and flag_save_fig:
        os.makedirs(results_dir)
    
    if large_scale_flag:
        print('==== Large SCALE MC tests Running...')
        num_mc = 500
        vec_delta_theta = np.arange(2, 11)
    else:
        print('=========== SMALL SCALE MC tests@@@ !!! =======')
        num_mc = 1 #50
        vec_delta_theta = np.arange(8, 9) #np.arange(2, 11)

    
    snr = 0
    algo_list = ["PER", "SPICE", "SAMV", "AFFINV"]
    
    num_algos = len(algo_list)
    
    se_mean = np.zeros((len(vec_delta_theta), num_algos))
    failing_rate = np.zeros((len(vec_delta_theta), num_algos))
    
    crb_list = np.zeros(len(vec_delta_theta))
    
    # Source powers in dB
    power_doa_db = np.array([3, 4])
    first_doa = 35.11
    
    for delta_theta_ind, delta_theta in enumerate(vec_delta_theta):
        print(f'=== Computing DeltaTheta == {delta_theta}')
        doa = np.array([first_doa, first_doa + delta_theta])
        
        se_mean_per_algo, failing_rate_per_algo, crb_val = compute_algos_std_err_parallel(
            algo_list, num_mc, snr, n, m, cohr_flag, power_doa_db, doa
        )
        
        se_mean[delta_theta_ind, :] = se_mean_per_algo
        failing_rate[delta_theta_ind, :] = failing_rate_per_algo
        crb_list[delta_theta_ind] = crb_val
    
    if flag_save_fig:
        np.savez(
            os.path.join(results_dir, 'Algos_Data'),
            vec_delta_theta=vec_delta_theta,
            se_mean=se_mean,
            failing_rate=failing_rate,
            algo_list=algo_list,
            crb_list=crb_list
        )
    
    # Plot figures
    color_set = ['r-->', 'm--p', 'b-^', 'g--s', 'k--']

    # Plot SE figure
    plt.figure()
    
    for i_algo in range(num_algos):
        prepare = np.sqrt(se_mean[:, i_algo] + np.finfo(float).eps)
        plt.semilogy(vec_delta_theta, prepare, color_set[i_algo], label=algo_list[i_algo])
    
    # CRB plot
    plt.semilogy(vec_delta_theta, np.sqrt(crb_list), color_set[-1], label='CRB')
    
    plt.xlabel(r'$\Delta \theta$ (degrees)')
    plt.ylabel('Angle RMSE (degree)')
    plt.title(f'{str_indp_cohr}, M={m}, N={n}')
    plt.legend()
    plt.grid(True)
    if flag_save_fig:
        plt.savefig(os.path.join(results_dir, 'MSE.png'), dpi=300)
        plt.savefig(os.path.join(results_dir, 'MSE.pdf'))

if __name__ == "__main__":
    # Example usage
    exp_DeltaTheta(n=100, cohr_flag=False, large_scale_flag=False)
