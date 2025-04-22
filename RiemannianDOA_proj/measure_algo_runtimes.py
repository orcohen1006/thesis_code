import numpy as np
from time import time
import matplotlib.pyplot as plt
import os
from ComputeAlgosStdErr_parallel import run_single_mc_iteration
from utils import get_doa_grid, get_algo_dict_list


def measure_algo_times():
    snr = 0

    # power_doa_db = np.array([0, -2, -4])
    # doa = np.array([40, 47, 55])
    vec_m = np.array([10, 40, 70, 100])
    NUM_TIMES = 20

    power_doa_db = np.array([0, 0, 0])
    doa = np.array([40, 45, 50])

    sources_power = 10.0 ** (power_doa_db / 10.0)
    algo_list = get_algo_dict_list()

    num_algos = len(algo_list)
    num_sources = len(doa)  # # of sources

    doa_scan = get_doa_grid()

    doa = np.sort(doa)



    noise_power_db = np.max(power_doa_db) - snr
    noise_power = 10.0 ** (noise_power_db / 10.0)
    runtime_data = np.zeros((len(vec_m),NUM_TIMES, num_algos))
    num_iters_data = np.zeros((len(vec_m),NUM_TIMES, num_algos))
    for ii, m in enumerate(vec_m):
        N = int(3*m)
        delta_vec = np.arange(m)
        A_true = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa * np.pi / 180)))
        A = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa_scan * np.pi / 180)))
        for jj in range(NUM_TIMES):
            print(f"========================================>   ========================================>       m={m}, {jj}/{NUM_TIMES}")
            _,_,p_vec_cell,runtime_list, num_iters_list = run_single_mc_iteration(i_mc=jj, algo_list=list(algo_list.keys()), snr=snr, t_samples=N,
                                                     m=m,cohr_flag=False,power_doa_db=power_doa_db,doa=doa, A_true= A_true, A=A,
                                                     noise_power=noise_power,doa_scan=doa_scan, seed=jj)
            runtime_data[ii,jj,:] = np.array(runtime_list)
            num_iters_data[ii,jj,:] = np.array(num_iters_list)

    print('yep')
    np.savez(
        os.path.join('data_measure_runtime'),
        runtime_data=runtime_data,
        num_iters_data=num_iters_data,
    )

    mean_iter_time, std_iter_time = np.mean(runtime_data / num_iters_data, axis=1), np.std(runtime_data / num_iters_data, axis=1)




if __name__ == "__main__":
    measure_algo_times()