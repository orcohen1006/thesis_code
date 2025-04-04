import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional

from ComputeAlgosStdErr_parallel import run_single_mc_iteration


def display_power_spectrum():
    seed = 0
    m = 12
    snr = 0
    N = 36


    power_doa_db = np.array([0, -5])
    sources_power = 10.0 ** (power_doa_db / 10.0)
    doa = np.array([40, 45])

    algo_list = ["PER", "SPICE", "SAMV", "AIRM", "JBLD"]
    color_set = ['r:>', 'm:p', 'b:^', 'g:s', 'y:o', 'k--']

    num_algos = len(algo_list)
    num_sources = len(doa)  # # of sources

    # doa_scan = np.arange(0, 180.5, 0.5)  # doa grid
    doa_scan = np.arange(0, 181, 1)  # doa grid

    doa = np.sort(doa)

    delta_vec = np.arange(m)
    A_true = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa * np.pi / 180)))
    A = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa_scan * np.pi / 180)))

    noise_power_db = np.max(power_doa_db) - snr
    noise_power = 10.0 ** (noise_power_db / 10.0)

    _,_,p_vec_cell = run_single_mc_iteration(i_mc=seed, algo_list=algo_list, snr=snr, t_samples=N,
                                             m=m,cohr_flag=False,power_doa_db=power_doa_db,doa=doa, A_true= A_true, A=A,
                                             noise_power=noise_power,doa_scan=doa_scan, seed=seed)

    plt.figure()
    plt.grid(True)
    plts = []
    epsilon_power = 10.0 ** (-30 / 10.0)
    for i_algo in range(1,num_algos):
        spectrum = p_vec_cell[i_algo]
        spectrum[spectrum < epsilon_power] = 0
        spectrum = 10*np.log10(spectrum)
        plt_line, = plt.plot(doa_scan, spectrum, color_set[i_algo], label=algo_list[i_algo])
        plts.append(plt_line)

    plt_doa, = plt.plot(doa, power_doa_db, 'x',color='red', label='DOA')
    plts.append(plt_doa)
    plt.legend(handles=plts)
    plt.xlabel(r"$\theta$ [degrees]", fontsize=14)
    plt.ylabel("power [dB]", fontsize=14)
    plt.title('Directions Power Spectrum Estimation')
    # plt.ylim([-15,15])
    plt.show()

if __name__ == "__main__":
    display_power_spectrum()