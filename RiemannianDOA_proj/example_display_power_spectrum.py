import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional

from ComputeAlgosStdErr_parallel import run_single_mc_iteration
from utils import get_doa_grid, get_algo_dict_list


def display_power_spectrum():
    seed = 42
    m = 12
    snr = 0
    N = 20


    # power_doa_db = np.array([0, -2, -4])
    # doa = np.array([40, 47, 55])

    # power_doa_db = np.array([0, 0])
    # doa = np.array([40.2, 45.3])

    power_doa_db = np.array([3, 4])
    doa = np.array([35, 40])

    sources_power = 10.0 ** (power_doa_db / 10.0)
    algo_list = get_algo_dict_list()

    num_algos = len(algo_list)
    num_sources = len(doa)  # # of sources

    doa_scan = get_doa_grid()

    doa = np.sort(doa)

    delta_vec = np.arange(m)
    A_true = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa * np.pi / 180)))
    A = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa_scan * np.pi / 180)))

    noise_power_db = np.max(power_doa_db) - snr
    noise_power = 10.0 ** (noise_power_db / 10.0)

    _,_,p_vec_cell,runtime_list, num_iters_list = run_single_mc_iteration(i_mc=seed, algo_list=list(algo_list.keys()), snr=snr, t_samples=N,
                                             m=m,cohr_flag=False,power_doa_db=power_doa_db,doa=doa, A_true= A_true, A=A,
                                             noise_power=noise_power,doa_scan=doa_scan, seed=seed)

    plt.figure()
    plt.grid(True)
    plts = []
    epsilon_power = 10.0 ** (-20 / 10.0)
    for i_algo, algo_name in enumerate(algo_list.keys()):
        spectrum = p_vec_cell[i_algo]
        spectrum[spectrum < epsilon_power] = epsilon_power
        spectrum = 10 * np.log10(spectrum)
        plt_line, = plt.plot(doa_scan, spectrum, label=algo_name, **algo_list[algo_name])
        plts.append(plt_line)


    plt_doa, = plt.plot(doa, power_doa_db, 'x',color='red', label='DOA')
    plts.append(plt_doa)
    plt.legend(handles=plts)
    plt.xlabel(r"$\theta$ [degrees]", fontsize=14)
    plt.ylabel("power [dB]", fontsize=14)
    plt.title('Directions Power Spectrum Estimation')
    # plt.ylim([-15,15])
    plt.show()
    print('ok...')

if __name__ == "__main__":
    display_power_spectrum()