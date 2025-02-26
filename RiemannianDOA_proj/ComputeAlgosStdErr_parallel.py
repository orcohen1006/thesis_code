import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
import multiprocessing as mp
from joblib import Parallel, delayed
from functools import partial

from fun_DASRes import *
from fun_SAM3Res import *
from fun_SPICEplusRes import *
from fun_Affinv import *
from SAM_CRB import *

def run_single_mc_iteration(
        i_mc: int,
        algo_list: List[str],
        snr: float,
        t_samples: int,
        m: int,
        cohr_flag: bool,
        power_doa_db: np.ndarray,
        doa: np.ndarray,
        A_true: np.ndarray,
        A: np.ndarray,
        noise_power: float,
        doa_scan: np.ndarray,
        seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a single Monte Carlo iteration with a specific random seed.

    Parameters:
    -----------
    i_mc : int
        Monte Carlo iteration index
    ... (other parameters same as compute_algos_std_err)
    seed : int
        Random seed for this iteration

    Returns:
    --------
    Tuple containing:
    - se_all_m: Array of squared errors for each algorithm
    - nan_flag_col_vec: Boolean array indicating NaN values
    """
    # Set random seed for this process
    np.random.seed(seed)

    t0 = time()
    num_algos = len(algo_list)
    num_sources = len(doa)
    amplitude_doa = np.sqrt(10.0 ** (power_doa_db / 10.0))

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
            detected_powers, distance, p_vec, normal, _ = fun_SAM3Res(y_noisy, A, modulus_hat_das, doa_scan, doa,
                                                                      noise_power)
        elif algo_list[i_algo] == "SPICE":
            detected_powers, distance, p_vec, normal, _ = fun_SPICEplusRes(y_noisy, A, modulus_hat_das, doa_scan, doa,
                                                                           noise_power)
        elif algo_list[i_algo] == "AFFINV":
            detected_powers, distance, p_vec, normal, _ = fun_Affinv(y_noisy, A, modulus_hat_das, doa_scan, doa,
                                                                     noise_power)
        else:
            raise ValueError("Algorithm not implemented")

        p_vec_cell[i_algo] = p_vec

        if not normal:
            sqr_err[i_algo] = np.nan
            power_se[i_algo] = np.nan
        else:
            power_dif = detected_powers - 10.0 ** (power_doa_db / 10.0)
            sqr_err[i_algo] = np.dot(distance, distance)
            power_se[i_algo] = np.dot(power_dif, power_dif)

    # Convert list to array for processing
    se_all_m = np.array([se if se is not None else np.nan for se in sqr_err])

    # Use the flag to track failures
    nan_flag_col_vec = np.isnan(se_all_m)

    dt = time() - t0
    print(f"i_MC = {i_mc + 1}, elapsed time: {dt:.2f} [sec]")

    return se_all_m, nan_flag_col_vec


def compute_algos_std_err_parallel(
        algo_list: List[str],
        num_mc: int,
        snr: float,
        t_samples: int,
        m: int,
        cohr_flag: bool,
        power_doa_db: np.ndarray,
        doa: np.ndarray,
        method: str = 'joblib',
        n_jobs: int = -1,
        random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute standard error for different DOA algorithms using parallel processing.

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
    method : str
        Parallelization method: 'joblib' or 'multiprocessing'
    n_jobs : int
        Number of processes to use. -1 means use all available cores.
    random_seed : int
        Base random seed for reproducibility

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

    doa_scan = np.arange(0, 180.5, 0.5)  # doa grid

    doa = np.sort(doa)

    delta_vec = np.arange(m)
    # True steering vector matrix
    A_true = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa * np.pi / 180)))
    # Steering vector matrix w.r.t all possible scanning DOA's
    A = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa_scan * np.pi / 180)))

    noise_power_db = np.mean(power_doa_db) - snr
    noise_power = 10.0 ** (noise_power_db / 10.0)

    # Generate a seed for each MC iteration for reproducibility
    np.random.seed(random_seed)
    seeds = np.arange(0, num_mc)

    # Create a partial function with fixed parameters
    run_mc_iteration = partial(
        run_single_mc_iteration,
        algo_list=algo_list,
        snr=snr,
        t_samples=t_samples,
        m=m,
        cohr_flag=cohr_flag,
        power_doa_db=power_doa_db,
        doa=doa,
        A_true=A_true,
        A=A,
        noise_power=noise_power,
        doa_scan=doa_scan
    )

    # Run parallel execution using the chosen method
    if method == 'joblib':
        print(f"Using joblib with {n_jobs} jobs")
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_mc_iteration)(i_mc=i, seed=seeds[i]) for i in range(num_mc)
        )
    elif method == 'multiprocessing':
        print(f"Using multiprocessing with {n_jobs if n_jobs > 0 else mp.cpu_count()} processes")
        # Adjust n_jobs if it's -1
        if n_jobs < 0:
            n_jobs = mp.cpu_count()

        # Create a pool of workers
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.starmap(
                run_mc_iteration,
                [(i, seeds[i]) for i in range(num_mc)]
            )
    else:
        raise ValueError(f"Unsupported parallelization method: {method}")

    # Process results
    for i_mc, (se_all_m, nan_flag_col_vec) in enumerate(results):
        failed_total_times = failed_total_times + nan_flag_col_vec

        # Change NaN to inf for sorting
        se_all_m_copy = se_all_m.copy()
        se_all_m_copy[nan_flag_col_vec] = np.inf
        se_history[:, i_mc] = se_all_m_copy

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