import numpy as np
import multiprocessing as mp
import time
from functools import partial
from . import fun_DASRes, fun_SAM3Res, fun_SPICEplusRes, fun_Affinv, SAM_CRB

def run_single_mc(i_MC, algo_list, SNR, t_samples, M, cohr_flag, PowerDOAdB, DOA, A_true, A, DOAscan, noisePower):
    """Helper function for parallel processing - runs a single Monte Carlo iteration"""
    t0 = time.time()
    num_algos = len(algo_list)
    num_sources = len(DOA)
    
    PowerDOA = 10 ** (PowerDOAdB / 10)
    amplitudeDOA = np.sqrt(PowerDOA)
    
    # Generate signal
    noise = np.sqrt(noisePower/2) * (np.random.randn(M, t_samples) + 1j * np.random.randn(M, t_samples))
    
    if not cohr_flag:  # independent sources
        waveform = np.exp(1j * 2 * np.pi * np.random.rand(num_sources, t_samples)) * amplitudeDOA.reshape(-1, 1)
    else:  # coherent sources
        waveform = np.exp(1j * 2 * np.pi * np.random.rand(num_sources - 1, t_samples))
        waveform = np.vstack((waveform, waveform[0, :]))
        waveform = waveform * amplitudeDOA.reshape(-1, 1)
    
    y_noisefree = A_true @ waveform  # ideal noiseless measurements
    y_noisy = y_noisefree + noise  # noisy measurements
    
    modulus_hat_das = np.sum(np.abs(A.conj().T @ y_noisy / M), axis=1) / t_samples
    
    # Run on all algorithms
    SqrErr = [None] * num_algos
    PowerSE = [None] * num_algos
    p_vec_cell = [None] * num_algos
    
    for i_algo in range(num_algos):
        Detected_powers = np.nan
        Distance = np.nan
        normal = np.nan
        
        if algo_list[i_algo] == "PER":
            Detected_powers, Distance, p_vec, normal, _ = fun_DASRes(y_noisy, A, modulus_hat_das, DOAscan, DOA)
        elif algo_list[i_algo] == "SAMV":
            Detected_powers, Distance, p_vec, normal, _ = fun_SAM3Res(y_noisy, A, modulus_hat_das, DOAscan, DOA, noisePower)
        elif algo_list[i_algo] == "SPICE":
            Detected_powers, Distance, p_vec, normal, _ = fun_SPICEplusRes(y_noisy, A, modulus_hat_das, DOAscan, DOA, noisePower)
        elif algo_list[i_algo] == "AFFINV":
            Detected_powers, Distance, p_vec, normal, _ = fun_Affinv(y_noisy, A, modulus_hat_das, DOAscan, DOA, noisePower)
        else:
            raise ValueError("Algorithm not implemented")
        
        p_vec_cell[i_algo] = p_vec
        
        if not normal:
            SqrErr[i_algo] = np.nan
            PowerSE[i_algo] = np.nan
        else:
            power_dif = Detected_powers - PowerDOA
            SqrErr[i_algo] = np.dot(Distance, Distance)
            PowerSE[i_algo] = np.dot(power_dif, power_dif)
    
    # Convert list to array for easier handling
    SE_all_m = np.array([se if not np.isnan(se) else np.inf for se in SqrErr])
    NaN_flag_col_vec = np.isnan(SqrErr)
    
    dt = time.time() - t0
    print(f"i_MC = {i_MC}, elapsed time: {dt:.2f} [sec]")
    
    return SE_all_m, NaN_flag_col_vec

def ComputeAlgosStdErr(algo_list, NUM_MC, SNR, t_samples, M, cohr_flag, PowerDOAdB, DOA):
    """
    Compute the standard error for different algorithms using Monte Carlo simulations.
    
    Parameters:
    algo_list: list of algorithms to evaluate
    NUM_MC: number of Monte Carlo simulations
    SNR: signal-to-noise ratio
    t_samples: number of time samples
    M: number of sensors
    cohr_flag: coherent sources flag
    PowerDOAdB: source powers in dB
    DOA: true DOA angles
    
    Returns:
    SE_mean_perAlgo: mean squared error per algorithm
    Failing_rate_perAlgo: rate of failing detections per algorithm
    CRB_val: Cramer-Rao Bound value
    """
    num_algos = len(algo_list)
    num_sources = len(DOA)
    
    PowerDOA = 10 ** (PowerDOAdB / 10)
    amplitudeDOA = np.sqrt(PowerDOA)
    
    DOAscan = np.arange(0, 180.5, 0.5)  # DOA grid
    DOA = np.sort(DOA)
    
    delta_vec = np.arange(M)
    A_true = np.exp(1j * np.pi * np.outer