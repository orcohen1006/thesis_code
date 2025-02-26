import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional

from fun_DASRes import *
from fun_SAM3Res import *
from fun_SPICEplusRes import *
from fun_Affinv import *
from SAM_CRB import *

def compute_algos_std_err(
    algo_list: List[str], 
    num_mc: int, 
    snr: float, 
    t_samples: int, 
    m: int, 
    cohr_flag: bool, 
    power_doa_db: np.ndarray, 
    doa: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute standard error for different DOA algorithms.
    
    Parameters:
    -----------
    algo_list : List[str]
        List of algorithm names to evaluate
    num_mc : int
        Number of Monte Carlo iterations
    snr : float
        Signal-to-noise ratio in dB
    t_samples : int
        Number of time samples
    m : int
        Number of array elements
    cohr_flag : bool
        Flag indicating if sources are coherent
    power_doa_db : np.ndarray
        Source powers in dB
    doa : np.ndarray
        Direction of arrival angles in degrees
        
    Returns:
    --------
    Tuple containing:
    - SE_mean_per_algo: Mean squared error for each algorithm
    - Failing_rate_per_algo: Failure rate for each algorithm
    - CRB_val: Cramer-Rao bound value
    """
    num_algos = len(algo_list)
    failed_total_times = np.zeros(num_algos)
    se_history = np.zeros((num_algos, num_mc))
    
    # Fixed Source powers
    num_sources = len(doa)  # # of sources
    
    power_doa = 10.0 ** (power_doa_db / 10.0)
    amplitude_doa = np.sqrt(power_doa)
    
    doa_scan = np.arange(0, 180.5, 0.5)  # doa grid
    
    doa = np.sort(doa)
    
    delta_vec = np.arange(m)
    # True steering vector matrix
    A_true = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa * np.pi / 180)))
    # Steering vector matrix w.r.t all possible scanning DOA's
    A = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa_scan * np.pi / 180)))
    
    noise_power_db = np.mean(power_doa_db) - snr
    noise_power = 10.0 ** (noise_power_db / 10.0)
    
    for i_mc in range(num_mc):
        t0 = time()
        
        # Generate signal
        noise = np.sqrt(noise_power / 2) * (np.random.randn(m, t_samples) + 1j * np.random.randn(m, t_samples))
        
        if not cohr_flag:  # independent sources
            waveform = np.exp(1j * 2 * np.pi * np.random.rand(num_sources, t_samples))
            waveform = waveform * np.tile(amplitude_doa, (t_samples, 1)).T
        else:  # coherent sources
            waveform = np.exp(1j * 2 * np.pi * np.random.rand(num_sources - 1, t_samples))
            waveform = np.vstack([waveform, waveform[0, :]])
            waveform = waveform * np.tile(amplitude_doa, (t_samples, 1)).T
        
        y_noisefree = A_true @ waveform  # ideal noiseless measurements
        y_noisy = y_noisefree + noise  # noisy measurements
        
        modulus_hat_das = np.sum(np.abs(A.conj().T @ (y_noisy / m)), axis=1) / t_samples
        
        # Run on all algorithms
        sqr_err = [None] * num_algos
        power_se = [None] * num_algos
        p_vec_cell = [None] * num_algos
        
        for i_algo in range(num_algos):
            detected_powers = np.nan
            distance = np.nan
            normal = np.nan
            if algo_list[i_algo] == "PER":
                detected_powers, distance, p_vec, normal, _ = fun_DASRes(y_noisy, A, modulus_hat_das, doa_scan, doa)
            elif algo_list[i_algo] == "SAMV":
                detected_powers, distance, p_vec, normal, _ = fun_SAM3Res(y_noisy, A, modulus_hat_das, doa_scan, doa, noise_power)
            elif algo_list[i_algo] == "SPICE":
                detected_powers, distance, p_vec, normal, _ = fun_SPICEplusRes(y_noisy, A, modulus_hat_das, doa_scan, doa, noise_power)
            elif algo_list[i_algo] == "AFFINV":
                detected_powers, distance, p_vec, normal, _ = fun_Affinv(y_noisy, A, modulus_hat_das, doa_scan, doa, noise_power)
            else:
                raise ValueError("Algorithm not implemented")
            p_vec_cell[i_algo] = p_vec
            
            if not normal:
                sqr_err[i_algo] = np.nan
                power_se[i_algo] = np.nan
            else:
                power_dif = detected_powers - power_doa
                sqr_err[i_algo] = np.dot(distance, distance)
                power_se[i_algo] = np.dot(power_dif, power_dif)
        
        if False:
            plt.figure()
            plt.grid(True)
            plts = []
            
            for i_algo in range(num_algos):
                plt_line, = plt.plot(doa_scan, 10 * np.log10(p_vec_cell[i_algo]), '-o', label=algo_list[i_algo])
                plts.append(plt_line)
            
            plt_doa, = plt.plot(doa, power_doa_db, 'x', label='DOA')
            plts.append(plt_doa)
            plt.legend(handles=plts)
            plt.show()
        
        # Convert list to matrix for processing
        se_all_m = np.array([se if se is not None else np.nan for se in sqr_err])
        
        # Use the flag to track failures
        nan_flag_col_vec = np.isnan(se_all_m)
        failed_total_times = failed_total_times + nan_flag_col_vec
        
        # Change NaN to inf
        se_all_m[nan_flag_col_vec] = np.inf
        se_history[:, i_mc] = se_all_m
        
        dt = time() - t0
        print(f"i_MC = {i_mc+1}, elapsed time: {dt:.2f} [sec]")
    
    # Prepare output
    se_history_sorted = np.sort(se_history, axis=1)[:, ::-1]  # Sort in descending order
    
    # Discard 2% bad results
    percent = 2
    actual_del_entries = int(np.ceil(num_mc * percent / 100))
    
    se_mean_per_algo = np.zeros(num_algos)
    failing_rate_per_algo = np.zeros(num_algos)
    
    for i_algo in range(num_algos):
        # Find first non-infinite value
        non_inf_indices = np.where(se_history_sorted[i_algo, :] < np.inf)[0]
        if len(non_inf_indices) > 0:
            start_pointer = non_inf_indices[0]
            count_pointer = start_pointer + actual_del_entries
            
            # Make sure count_pointer doesn't exceed array bounds
            count_pointer = min(count_pointer, num_mc - 1)
            
            if count_pointer < num_mc:
                se_mean_per_algo[i_algo] = np.mean(se_history_sorted[i_algo, count_pointer:])
        
        failing_rate_per_algo[i_algo] = failed_total_times[i_algo] / num_mc
    
    crb_val = SAM_CRB(snr, t_samples, cohr_flag, power_doa_db, doa)
    
    return se_mean_per_algo, failing_rate_per_algo, crb_val
