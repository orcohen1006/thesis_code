import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import List, Optional
from ComputeAlgosStdErr_parallel import compute_algos_std_err_parallel
from utils import get_algo_dict_list, get_doa_grid


def exp_N(cohr_flag: bool, large_scale_flag: bool) -> None:
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
    results_dir = f'Exp_N_{timestamp}_{str_indp_cohr}_M{m}_L{int(large_scale_flag)}'
    
    if not os.path.exists(results_dir) and flag_save_fig:
        os.makedirs(results_dir)
    
    if large_scale_flag:
        print('==== Large SCALE MC tests Running...')
        num_mc = 500
        vec_n = np.arange(16, 50, 5)
    else:
        print('=========== SMALL SCALE MC tests@@@ !!! =======')
        num_mc = 100 #50
        vec_n = np.arange(20, 50+1, 10)

    
    snr = 0
    algo_list = get_algo_dict_list()
    
    num_algos = len(algo_list)
    
    se_mean = np.zeros((len(vec_n), num_algos))
    failing_rate = np.zeros((len(vec_n), num_algos))
    
    crb_list = np.zeros(len(vec_n))
    
    # Source powers in dB
    power_doa_db = np.array([0, 0])
    doa = np.array([40.2, 45.3])

    for n_ind, n in enumerate(vec_n):
        print(f'=== Computing N == {n}')

        se_mean_per_algo, failing_rate_per_algo, crb_val = compute_algos_std_err_parallel(
            list(algo_list.keys()), num_mc, snr, n, m, cohr_flag, power_doa_db, doa
        )
        
        se_mean[n_ind, :] = se_mean_per_algo
        failing_rate[n_ind, :] = failing_rate_per_algo
        crb_list[n_ind] = crb_val
    
    if flag_save_fig:
        np.savez(
            os.path.join(results_dir, 'Algos_Data'),
            vec_n=vec_n,
            se_mean=se_mean,
            failing_rate=failing_rate,
            algo_list=algo_list,
            crb_list=crb_list
        )

    # Plot SE figure
    plt.figure()

    for i_algo, algo_name in enumerate(algo_list.keys()):
        prepare = np.sqrt(se_mean[:, i_algo] + np.finfo(float).eps)
        plt.plot(vec_n, prepare, label=algo_name, **algo_list[algo_name])

        # CRB plot
    plt.plot(vec_n, np.sqrt(crb_list), 'k--', label='CRB')
    
    plt.xlabel(r'$N$ (samples)')
    plt.ylabel('Angle RMSE (degree)')
    #plt.title(f'{str_indp_cohr}, M={m}, N={n}')
    plt.legend()
    plt.grid(True)
    if flag_save_fig:
        plt.savefig(os.path.join(results_dir, 'MSE_' + results_dir + '.png'), dpi=300)
        plt.savefig(os.path.join(results_dir, 'MSE_' + results_dir + '.pdf'))

if __name__ == "__main__":
    # Example usage
    exp_N(cohr_flag=False, large_scale_flag=False)
