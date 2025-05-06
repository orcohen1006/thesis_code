import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
import multiprocessing as mp
from joblib import Parallel, delayed
from functools import partial
from dataclasses import dataclass

from utils import create_config
from fun_DAS import *
from fun_SAMV import *
from fun_SPICE import *
from fun_Riemannian import *
from RiemannianDOA_proj.CRB import *

def run_single_mc_iteration(
        i_mc: int,
        algo_list: List[str],
        config: dict,
        A_true: np.ndarray,
        A: np.ndarray,
        doa_scan: np.ndarray,
        seed: int
):
    """
    Run a single Monte Carlo iteration with a specific random seed.

    Parameters:
    -----------
    i_mc : int
        Monte Carlo iteration index
    config : Config
        Configuration structure containing simulation parameters
    ... (other parameters same as compute_algos_std_err)
    seed : int
        Random seed for this iteration

    Returns:
    --------
    Tuple containing:
    - se_all_m: Array of squared errors for each algorithm
    - nan_flag_col_vec: Boolean array indicating NaN values
    """

    t0 = time()
    num_algos = len(algo_list)
    num_sources = len(config["doa"])
    amplitude_doa = np.sqrt(10.0 ** (config["power_doa_db"] / 10.0))

    noise_power_db = np.max(config["power_doa_db"]) - config["snr"]
    noise_power = 10.0 ** (noise_power_db / 10.0)

    y_noisy = generate_signal(A_true, config["power_doa_db"], config["N"], noise_power, cohr_flag=False, seed=seed)

    modulus_hat_das = np.sum(np.abs(A.conj().T @ (y_noisy / config["m"])), axis=1) / config["N"]

    # Run on all algorithms
    sqr_err = [None] * num_algos
    power_se = [None] * num_algos
    p_vec_cell = [None] * num_algos
    runtime_list = [None] * num_algos
    num_iters_list = [None] * num_algos

    for i_algo in range(num_algos):
        t_algo_start = time()
        if algo_list[i_algo] == "PER":
            p_vec, num_iters, _ = fun_DAS(y_noisy, A, modulus_hat_das, doa_scan, config["doa"])
        elif algo_list[i_algo] == "SAMV":
            p_vec, num_iters, _ = fun_SAMV(y_noisy, A, modulus_hat_das, doa_scan, config["doa"], noise_power)
        elif algo_list[i_algo] == "SPICE":
            p_vec, num_iters, _ = fun_SPICE(y_noisy, A, modulus_hat_das, doa_scan, config["doa"], noise_power)
        elif algo_list[i_algo] == "AIRM":
            p_vec, num_iters, _ = fun_Riemannian(y_noisy, A, modulus_hat_das, doa_scan, config["doa"], noise_power, loss_name="AIRM")
        elif algo_list[i_algo] == "JBLD":
            p_vec, num_iters, _ = fun_Riemannian(y_noisy, A, modulus_hat_das, doa_scan, config["doa"], noise_power, loss_name="JBLD")
        else:
            raise ValueError("Algorithm not implemented")

        runtime_list[i_algo] = time() - t_algo_start
        print(f"{algo_list[i_algo]}: #iters= {num_iters}, runtime= {runtime_list[i_algo]} [sec]")
        num_iters_list[i_algo] = num_iters

        p_vec_cell[i_algo] = p_vec
        detected_powers, distance, normal = detect_DOAs(p_vec, doa_scan, config["doa"])

        if not normal:
            sqr_err[i_algo] = np.nan
            power_se[i_algo] = np.nan
        else:
            power_dif = detected_powers - 10.0 ** (config["power_doa_db"] / 10.0)
            distance = distance.astype(float)
            sqr_err[i_algo] = np.dot(distance, distance)
            power_se[i_algo] = np.dot(power_dif, power_dif)

    if False:
        plt.figure()
        plt.grid(True)
        plts = []

        for i_algo in range(num_algos):
            plt_line, = plt.plot(doa_scan, 10 * np.log10(p_vec_cell[i_algo]), '-o', label=algo_list[i_algo])
            plts.append(plt_line)

        plt_doa, = plt.plot(config["doa"], config["power_doa_db"], 'x', label='DOA')
        plts.append(plt_doa)
        plt.legend(handles=plts)
        plt.ylim([-15,15])
        plt.show()
    # Convert list to array for processing
    se_all_m = np.array([se if se is not None else np.nan for se in sqr_err])

    # Use the flag to track failures
    nan_flag_col_vec = np.isnan(se_all_m)

    dt = time() - t0
    print(f"i_MC = {i_mc + 1}, elapsed time: {dt:.2f} [sec]")

    return se_all_m, nan_flag_col_vec, p_vec_cell, runtime_list, num_iters_list


def compute_algos_std_err_parallel(
        algo_list: List[str],
        num_mc: int,
        config: dict,
        method: str = 'joblib',
        n_jobs: int = 20,
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
    config : Config
        Configuration structure containing simulation parameters
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
    num_sources = len(config["doa"])  # # of sources

    power_doa = 10.0 ** (config["power_doa_db"] / 10.0)

    doa_scan = get_doa_grid()

    config["doa"] = np.sort(config["doa"])

    delta_vec = np.arange(config["m"])
    A_true = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(config["doa"] * np.pi / 180)))
    A = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa_scan * np.pi / 180)))

    noise_power_db = np.max(config["power_doa_db"]) - config["snr"]
    noise_power = 10.0 ** (noise_power_db / 10.0)

    # Generate a seed for each MC iteration for reproducibility
    np.random.seed(random_seed)
    seeds = np.arange(0, num_mc)

    # Create a partial function with fixed parameters
    run_mc_iteration = partial(
        run_single_mc_iteration,
        algo_list=algo_list,
        config=config,
        A_true=A_true,
        A=A,
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
    for i_mc, (se_all_m, nan_flag_col_vec,_, _, _) in enumerate(results):
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

    crb_val = cramer_rao_lower_bound(config)

    return se_mean_per_algo, failing_rate_per_algo, crb_val