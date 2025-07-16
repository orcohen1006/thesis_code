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
# %%
import utils
import ToolsMC
import importlib
importlib.reload(utils)
importlib.reload(ToolsMC)
from utils import *
from ToolsMC import *
# %%
def example_display_power_spectrum():
    # %%
    # filepath_results_file = os.path.abspath('Exp_N_y2025-m06-d03_09-28-47_indp/results.pkl')
    filepath_results_file = '/home/or.cohen/thesis_code/RiemannianDOA_proj/run_exp_y2025-m07-d15_16-52-19/Exp_SNR_y2025-m07-d15_16-52-19_indp_N_40_secondsourcesnr_0' + '/results.pkl'
    with open(filepath_results_file, 'rb') as f:
        results = pickle.load(f)
    # %%
    for i_config in range(len(results)):
        print(f"-------------- Config {i_config}:")
        print(results[i_config][0]["config"])
    # %%
    algo_list = get_algo_dict_list()
    i_config = 0 
    i_mc = 2
    ax = display_power_spectrum(results[i_config][i_mc]["config"], results[i_config][i_mc]["p_vec_list"])

    doas = results[i_config][i_mc]["config"]["doa"]
    power_doa_db = results[i_config][i_mc]["config"]["power_doa_db"]
    ax.set_xlim([np.min(doas)-10, np.max(doas)+10])
    ax.set_ylim([-20, np.max(power_doa_db)+3])
    
    # %%
    
    

    # %%
def display_power_spectrum_tmp():
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

    config_dict = create_config(
        m=m, snr=snr, N=N, power_doa_db=power_doa_db, doa=doa, cohr_flag=False,
    )
    _, _, p_vec_cell, runtime_list, num_iters_list = run_single_mc_iteration(
        i_mc=seed, algo_list=list(algo_list.keys()), config=config_dict, A_true=A_true, A=A,
        doa_scan=doa_scan, seed=seed
    )

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
    example_display_power_spectrum()
    
# %%
