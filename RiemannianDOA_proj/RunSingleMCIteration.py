# %%
import datetime
import sys
import numpy as np
from time import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
from functools import partial
from dataclasses import dataclass
from collections import defaultdict
import pickle
from utils import *
from fun_DAS import *
from fun_SAMV import *
from fun_SPICE import *
from fun_Riemannian import *
import os
import logging
# %%
def run_single_mc_iteration(
        i_mc: int,
        config: dict,
        algo_list : Optional[List[str]] = None,
        do_log: bool = False
):

    t0 = time()
    num_sources = len(config["doa"])

    power_doa_db = config["power_doa_db"]
    power_doa = 10.0 ** (power_doa_db / 10.0)
    noise_power_db = np.max(config["power_doa_db"]) - config["snr"]
    noise_power = 10.0 ** (noise_power_db / 10.0)


    doa_scan = get_doa_grid()
    if do_log:
        logging.info(f"- got doa_scan.")
    A_true = get_steering_matrix(config["doa"], config["m"])
    A = get_steering_matrix(doa_scan, config["m"])
    if do_log:
        logging.info(f"- got A_true and A.")


    if algo_list is None:
        algo_list = list(get_algo_dict_list().keys())

    num_algos = len(algo_list)

    y_noisy = generate_signal(A_true, power_doa_db, config["N"], noise_power, cohr_flag=config["cohr_flag"], 
                              cohr_coeff = config["cohr_coeff"], noncircular_coeff=config["noncircular_coeff"],
                              seed=i_mc)

    if do_log:
        logging.info(f"- generated noisy signal y_noisy with shape {y_noisy.shape}.")
        
    modulus_hat_das = np.sum(np.abs(A.conj().T @ (y_noisy / config["m"])), axis=1) / config["N"]
    
    # Run on all algorithms
    p_vec_list = [None] * num_algos
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
        num_iters_list[i_algo] = num_iters
        p_vec_list[i_algo] = p_vec
        msg = f"{algo_list[i_algo]}: #iters= {num_iters}, runtime= {runtime_list[i_algo]} [sec]"
        if do_log:
            logging.info(msg)
        else:
            print(msg)

    msg = f"i_mc = {i_mc + 1}, elapsed time: {time() - t0 :.2f} [sec]"
    if do_log:
        logging.info(msg)
    else:
        print(msg)
    
    result = {}
    result["i_mc"] = i_mc
    result['config'] = config
    # result["R_hat"] = (y_noisy @ y_noisy.conj().T) / config["N"]
    result['runtime_list'] = runtime_list
    result['num_iters_list'] = num_iters_list
    result['p_vec_list'] = p_vec_list

    
    return result

# def load_all_results(dirpath: str, prefix_result_files: str) -> None:
#     """
#     Load all result files with the given prefix, delete them, and save the overall results.

#     Args:
#         dirpath (str): Directory path where the result files are stored.
#         prefix_result_files (str): Prefix of the result files.
#     """

#     overall_results = []
#     for filename in os.listdir(dirpath):
#         if filename.startswith(prefix_result_files) and filename.endswith('.pkl'):
#             filepath = os.path.join(dirpath, filename)
#             with open(filepath, 'rb') as f:
#                 overall_results.append(pickle.load(f))
#             os.remove(filepath)

#     overall_filepath = os.path.join(dirpath, "overall_results.pkl")
#     with open(overall_filepath, 'wb') as f:
#         pickle.dump(overall_results, f)

if __name__ == "__main__":

    job_id = int(sys.argv[1])  # PBS job index
    log_dir = "/home/or.cohen/thesis_code/RiemannianDOA_proj/job_logs"
    log_path = os.path.join(log_dir, f"mylog_job_{job_id}.log")

    # Configure logging to write immediately
    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Also print to stdout in real-time if desired
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger("").addHandler(console)

    # Disable buffering on stdout
    sys.stdout.reconfigure(line_buffering=True)

    datetime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logging.info(f"~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ {datetime_str} : RunSingleMCIteration {job_id} ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")


    with open(FILENAME_PBS_METADATA, "rb") as f:
        metadata = pickle.load(f)

    logging.info(f"Metadata loaded: {metadata}")

    num_configs = metadata["num_configs"]
    num_mc = metadata["num_mc"]
    num_jobs = metadata["num_jobs"]
    workdir = metadata["workdir"]

    total_tasks = num_configs * num_mc
    tasks_per_job = (total_tasks + num_jobs - 1) // num_jobs  # ceil division

    start_task = job_id * tasks_per_job
    end_task = min(start_task + tasks_per_job, total_tasks)
    logging.info(f"Job {job_id} will process tasks from {start_task} to {end_task} (total: {end_task - start_task})")
    for task_id in range(start_task, end_task):
        i_config = task_id // num_mc
        i_mc = task_id % num_mc

        config_path = f"{workdir}/config_{i_config}.pkl"
        result_path = f"{workdir}/config_{i_config}_mc_{i_mc}.pkl"
        logging.info(f"Processing config file: {config_path}  , result will be saved to: {result_path}")
        # Load config and run
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        logging.info(f"Loaded config: {config}")
        result = run_single_mc_iteration(i_mc=i_mc, config=config, do_log = True)

        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"(task_id:{task_id}) successfully saved result file: {result_path}")
    # %% Example usage
    # i_mc = 0
    # filepath_config = "tmp_config_file.pkl"
    # filepath_result = f"tmp_result_{i_mc}.pkl"
    # config = create_config(
    #     m=12, snr=0, N=20, power_doa_db=np.array([3, 4]), doa=np.array([35, 40]), cohr_flag=False,
    # )
    # with open(filepath_config, 'wb') as f:
    #     pickle.dump(config, f)

    # result = run_single_mc_iteration(i_mc, filepath_config, filepath_result)

    # with open(filepath_result, 'rb') as f:
    #     loaded_result = pickle.load(f)

def tmp():
    # %%
    from RunSingleMCIteration import run_single_mc_iteration
    from example_display_power_spectrum import display_power_spectrum
    
    config = create_config(
        m=12, snr=0, N=40, power_doa_db=np.array([3, 4]), doa=np.array([35, 40]), cohr_flag=False,
    )
    algo_list = get_algo_dict_list(flag_also_use_PER=True)
    # remain only with keys "PER" and "JBLD"
    algo_list = {k: algo_list[k] for k in ["PER", "JBLD"] if k in algo_list}
    # %%
    result = run_single_mc_iteration(0, config, algo_list=list(algo_list.keys()))
    
    ax = display_power_spectrum(result["config"], result["p_vec_list"], algo_list=algo_list)

    # doas = result["config"]["doa"]
    # power_doa_db = result["config"]["power_doa_db"]
    # ax.set_xlim([np.min(doas)-10, np.max(doas)+10])
    # ax.set_ylim([-20, np.max(power_doa_db)+3])

    

# %%
# total_tasks = 1000
# num_jobs = 334
# tasks_per_job = (total_tasks + num_jobs - 1) // num_jobs  # ceil division
# print(tasks_per_job)
# %%
