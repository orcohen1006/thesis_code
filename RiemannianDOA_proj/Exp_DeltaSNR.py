import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import List, Optional
from ComputeAlgosStdErr_parallel import compute_algos_std_err_parallel
from utils import get_algo_dict_list


def exp_DeltaSNR(n: int, cohr_flag: bool, large_scale_flag: bool) -> None:
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
    results_dir = f'Exp_DeltaSNR_{timestamp}_{str_indp_cohr}_M{m}_N{n}_L{int(large_scale_flag)}'
    
    if not os.path.exists(results_dir) and flag_save_fig:
        os.makedirs(results_dir)
    
    if large_scale_flag:
        print('==== Large SCALE MC tests Running...')
        num_mc = 500
        vec_delta_snr = np.arange(start=-10, stop=0, step=2)
    else:
        print('=========== SMALL SCALE MC tests@@@ !!! =======')
        num_mc = 100
        # vec_delta_snr = np.arange(start=-10, stop=1, step=2)
        vec_delta_snr = np.arange(start=-5, stop=1, step=1)

    
    snr = 0
    algo_list = get_algo_dict_list()
    
    num_algos = len(algo_list)
    
    se_mean = np.zeros((len(vec_delta_snr), num_algos))
    failing_rate = np.zeros((len(vec_delta_snr), num_algos))
    
    crb_list = np.zeros(len(vec_delta_snr))
    
    # Source powers in dB
    doa = np.array([35, 40])
    firstSourcePower_db = 0
    for delta_snr_ind, delta_snr in enumerate(vec_delta_snr):
        print(f'=== Computing DeltaSNR == {delta_snr}')
        power_doa_db = np.array([firstSourcePower_db, firstSourcePower_db+delta_snr])

        se_mean_per_algo, failing_rate_per_algo, crb_val = compute_algos_std_err_parallel(
            list(algo_list.keys()), num_mc, snr, n, m, cohr_flag, power_doa_db, doa
        )
        
        se_mean[delta_snr_ind, :] = se_mean_per_algo
        failing_rate[delta_snr_ind, :] = failing_rate_per_algo
        crb_list[delta_snr_ind] = crb_val
    
    if flag_save_fig:
        np.savez(
            os.path.join(results_dir, 'Algos_Data'),
            vec_delta_snr=vec_delta_snr,
            se_mean=se_mean,
            failing_rate=failing_rate,
            algo_list=algo_list,
            crb_list=crb_list
        )

    # Plot SE figure
    plt.figure()
    
    for i_algo, algo_name in enumerate(algo_list.keys()):
        prepare = np.sqrt(se_mean[:, i_algo] + np.finfo(float).eps)
        plt.plot(vec_delta_snr, prepare, label=algo_name, **algo_list[algo_name])
    
    # CRB plot
    plt.plot(vec_delta_snr, np.sqrt(crb_list), 'k--', label='CRB')
    
    plt.xlabel(r'$\Delta SNR$ (dB)')
    plt.ylabel('Angle RMSE (degree)')
    #plt.title(f'{str_indp_cohr}, M={m}, N={n}')
    plt.legend()
    plt.grid(True)
    if flag_save_fig:
        plt.savefig(os.path.join(results_dir, 'MSE_' + results_dir + '.png'), dpi=300)
        plt.savefig(os.path.join(results_dir, 'MSE_' + results_dir + '.pdf'))

if __name__ == "__main__":
    # Example usage
    exp_DeltaSNR(n=30, cohr_flag=False, large_scale_flag=False)
