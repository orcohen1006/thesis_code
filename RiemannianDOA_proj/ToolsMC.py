# %%
import os
import numpy as np
import pickle
import subprocess
import time
from pathlib import Path
from utils import *
from collections import defaultdict
DEFAULT_NUM_JOBS = 334
# %%
def save_job_metadata(workdir: str, config_list: list, num_mc: int, num_jobs: int):
    if os.path.exists(FILENAME_PBS_METADATA):
        os.remove(FILENAME_PBS_METADATA)
    metadata = {}
    metadata["num_configs"] = len(config_list)
    metadata["num_jobs"] = num_jobs
    metadata["num_mc"] = num_mc
    metadata["workdir"] = workdir
    with open(FILENAME_PBS_METADATA, "wb") as f:
        pickle.dump(metadata, f)

def save_doa_configs(workdir: str, config_list: list):
    for i in range(len(config_list)):
        cfg = config_list[i]
        with open(f"{workdir}/config_{i}.pkl", "wb") as f:
            pickle.dump(cfg, f)

def submit_job_array(workdir: str, config_list: list, num_mc: int, num_jobs: int):

    res = subprocess.run(["qsub", "-J", f"0-{num_jobs-1}", FILENAME_PBS_SCRIPT], capture_output=True, text=True)

    if res.returncode != 0:
        print("Error submitting job array:", res.stderr)
        raise RuntimeError("Job submission failed.")

    print("Job submission output:", res.stdout)
    return res.stdout.strip()

def wait_for_results(workdir: str, config_list: list, num_mc:int, job_id:str, t0: float):
    expected = len(config_list) * num_mc
    while True:
        done = len(list(Path(workdir).glob("config_*_mc_*.pkl")))
        # workdir is a string of a path, get string of the last two parts of the path
        workdir_parts = Path(workdir).parts[-2:] 
        print(f"Job {job_id} Waiting... {workdir_parts},   {done}/{expected} results ready. elapsed time: {time.time() - t0:.2f} [sec]")
        if done >= expected:
            break
        time.sleep(5) 

def collect_results(workdir: str, config_list: list, num_mc:int):
    results = []
    for i in range(len(config_list)):
        curr_config_results = []
        for j in range(num_mc):
            result_file = Path(workdir) / f"config_{i}_mc_{j}.pkl"
            with open(result_file, "rb") as f:
                curr_config_results.append(pickle.load(f))
            os.remove(result_file)
        results.append(curr_config_results)
    return results

def RunDoaConfigsPBS(workdir: str, config_list: list, num_mc:int, num_jobs: int = DEFAULT_NUM_JOBS):
    t0 = time.time()
    print("Starting job array with ", num_jobs, " jobs:", len(config_list), "configurations and", num_mc, "Monte Carlo iterations.")
    
    save_job_metadata(workdir, config_list, num_mc, num_jobs)

    save_doa_configs(workdir, config_list)
    
    job_id = submit_job_array(workdir, config_list, num_mc, num_jobs)
    print("Submitted job ID:", job_id)

    wait_for_results(workdir, config_list, num_mc, job_id, t0)
    print("All results ready. Collecting...")

    results = collect_results(workdir, config_list, num_mc)
    print(f"Collected {len(results)} results.")
    with open(f"{workdir}/results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {workdir}/results.pkl")
    print(f"RunDoaConfigsPBS: Total elapsed time: {time.time() - t0:.2f} [sec]")
    return results

# %%
def analyze_algo_errors(results: list):
    from CRB import cramer_rao_lower_bound
    algo_list = get_algo_dict_list()
    num_algo = len(algo_list)
    grid_doa = get_doa_grid()
    num_configs = len(results)
    num_mc = len(results[0])
    for i_config in range(num_configs):
        config = results[i_config][0]["config"]
        tmp_doa = np.expand_dims(config["doa"],-1)
        threshold_theta_detect = 5 # np.abs(tmp_doa - tmp_doa.T).max()
        print(f"Config {i_config}: threshold_theta_detect = {threshold_theta_detect}")
        A = np.exp(1j * np.pi * np.outer(np.arange(config["m"]), np.cos(grid_doa * np.pi / 180)))
        noise_power = 10.0 ** ((np.max(config["power_doa_db"]) - config["snr"]) / 10.0)
        for i_mc in range(num_mc):
            result = results[i_config][i_mc]
            num_detected = [None] * num_algo
            selected_doa_error = [None] * num_algo
            selected_power_error = [None] * num_algo
            succ_match_detected_doa = [None] * num_algo
            succ_match_true_doa = [None] * num_algo
            # num_detected_aic = [None] * num_algo
            # num_detected_mdl = [None] * num_algo
            for i_alg in range(num_algo):
                num_detected[i_alg], _, _, selected_doa_error[i_alg], selected_power_error[i_alg], succ_match_detected_doa[i_alg], succ_match_true_doa[i_alg] =\
                    estimate_doa_calc_errors(
                        result["p_vec_list"][i_alg], grid_doa,
                        result["config"]["doa"],
                        convert_db_to_linear(result["config"]["power_doa_db"]), threshold_theta_detect=threshold_theta_detect)
                # R = A @ np.diag(result["p_vec_list"][i_alg]) @ A.conj().T + noise_power* np.eye(config["m"])
                # num_detected_aic[i_alg], num_detected_mdl[i_alg] = model_order_selection(R, config["N"])
            result["num_detected"] = num_detected
            result["selected_doa_error"] = selected_doa_error
            result["selected_power_error"] = selected_power_error
            result["succ_match_detected_doa"] = succ_match_detected_doa
            result["succ_match_true_doa"] = succ_match_true_doa
            # result["num_detected_aic"] = num_detected_aic
            # result["num_detected_mdl"] = num_detected_mdl


    algos_error_data = {key: defaultdict(lambda: [None]*num_configs) for key in 
                        ["mean_doa_errors", "mean_power_errors", "mean_square_doa_errors", "mean_square_power_errors", 
                         "prob_detect","prob_false_detection", "prob_full_detection"]}

    for i_config in range(len(results)):
        config = results[i_config][0]["config"]
        indices_mc_all_algos_detected_enough_sources = [
            i_mc for i_mc in range(num_mc)
            if all([num_det >= len(config["doa"]) for num_det in results[i_config][i_mc]["num_detected"]])
            ]

        print(f"Config {i_config}: {len(indices_mc_all_algos_detected_enough_sources)}/{num_mc} MC iterations where all algos detected enough sources.")
        for i_algo, algo_name in enumerate(algo_list.keys()):
            if len(indices_mc_all_algos_detected_enough_sources) < 0.4 * num_mc:
                # If not enough MC iterations to look at, this config is irelevant
                doa_errors = np.expand_dims(results[i_config][0]["selected_doa_error"][i_algo] * np.nan, axis=0)
                power_errors = np.expand_dims(results[i_config][0]["selected_power_error"][i_algo] * np.nan, axis=0)
            else:
                doa_errors = np.stack([results[i_config][i_mc]["selected_doa_error"][i_algo] 
                                            for i_mc in indices_mc_all_algos_detected_enough_sources])
                power_errors = np.stack([results[i_config][i_mc]["selected_power_error"][i_algo] 
                                            for i_mc in indices_mc_all_algos_detected_enough_sources])
                mean_square_errors = np.mean(doa_errors**2, axis=1)
                prcnt_worst_results_to_ignore = 2
                index = np.ceil((100 - prcnt_worst_results_to_ignore) / 100 * len(mean_square_errors)).astype(int)
                bottom_q_percent_indices = np.argsort(mean_square_errors)[:index]
                doa_errors = np.stack([doa_errors[i] for i in bottom_q_percent_indices])
                power_errors = np.stack([power_errors[i] for i in bottom_q_percent_indices])         

            algos_error_data["mean_doa_errors"][algo_name][i_config] = np.mean(doa_errors, axis=0)
            algos_error_data["mean_power_errors"][algo_name][i_config] = np.mean(power_errors, axis=0)
            algos_error_data["mean_square_doa_errors"][algo_name][i_config] = np.mean(doa_errors**2, axis=0)
            algos_error_data["mean_square_power_errors"][algo_name][i_config] = np.mean(power_errors**2, axis=0)
            
            # inds_detected_enough = [results[i_config][i_mc]["num_detected"][i_algo] >= len(config["doa"]) for i_mc in range(num_mc)]
            prob_detection =  [np.sum(results[i_config][i_mc]["succ_match_true_doa"][i_algo])/len(config["doa"])
                                for i_mc in range(num_mc)]
            prob_false_detection = [1 - np.sum(results[i_config][i_mc]["succ_match_detected_doa"][i_algo])/(1e-10+results[i_config][i_mc]["num_detected"][i_algo])
                                for i_mc in range(num_mc)]
            # is_full_detection = [np.sum(results[i_config][i_mc]["succ_match_true_doa"][i_algo]) == len(config["doa"])
            #                     for i_mc in range(num_mc)]
            is_full_detection = [np.sum(results[i_config][i_mc]["num_detected"][i_algo]) == len(config["doa"])
                                for i_mc in range(num_mc)]
            algos_error_data["prob_detect"][algo_name][i_config] = np.mean(prob_detection)
            algos_error_data["prob_false_detection"][algo_name][i_config] = np.mean(prob_false_detection)
            algos_error_data["prob_full_detection"][algo_name][i_config] = np.mean(is_full_detection)
        algos_error_data["mean_square_doa_errors"]["CRB"][i_config] = cramer_rao_lower_bound(config)
        # --
    return results, algos_error_data
# %%
def tmp123():
    print("yep")
# %%
def plot_prob_detection(algos_error_data: dict, parameter_name: str, parameter_units: str, parameter_values: list):
    # import matplotlib.pyplot as plt
    # import matplotlib.gridspec as gridspec
    
    # algo_list = get_algo_dict_list()    

    # fig = plt.figure()
    # ax1 = fig.gca()
    # ax1.grid(True)
    # ax1.set_xlabel("FPR")
    # ax1.set_ylabel("TPR")
    # for algo_name in algo_list.keys():
    #     tpr = np.stack(algos_error_data["prob_detect"][algo_name])
    #     fpr = np.stack(algos_error_data["prob_false_detection"][algo_name])
    #     ax1.plot(fpr, tpr, label=algo_name, **algo_list[algo_name])
   
    # return fig

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    algo_list = get_algo_dict_list()    

    fig = plt.figure()
    ax1 = fig.gca()
    ax1.set_title("Probability of Detection")
    ax1.grid(True)
    ax1.set_xlabel(parameter_name + f" {parameter_units}")
    ax1.set_ylabel("$P_{D}$")
    for algo_name in algo_list.keys():
        prob_detect = np.stack(algos_error_data["prob_detect"][algo_name])
        ax1.plot(parameter_values, prob_detect, label=algo_name, **algo_list[algo_name])

    return fig

    # import matplotlib.pyplot as plt
    # import matplotlib.gridspec as gridspec
    
    # algo_list = get_algo_dict_list()    

    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.set_title("Probability of Detection")
    # ax1.grid(True)
    # ax1.set_xlabel(parameter_name + f" {parameter_units}")
    # ax1.set_ylabel("$P_{D}$")
    # for algo_name in algo_list.keys():
    #     prob_detect = np.stack(algos_error_data["prob_detect"][algo_name])
    #     ax1.plot(parameter_values, prob_detect, label=algo_name, **algo_list[algo_name])
    # ax2 = fig.add_subplot(212)
    # ax2.set_title("Probability of False Detection")
    # ax2.grid(True)
    # ax2.set_xlabel(parameter_name + f" {parameter_units}")
    # ax2.set_ylabel("$P_{FD}$")
    # for algo_name in algo_list.keys():
    #     prob_false_detection = np.stack(algos_error_data["prob_false_detection"][algo_name])
    #     ax2.plot(parameter_values, prob_false_detection, label=algo_name, **algo_list[algo_name])
    # ax2.legend()
    # plt.tight_layout()
    # return fig

# def plot_doa_errors(algos_error_data: dict, parameter_name: str, parameter_units: str, parameter_values: list, normalize_rmse_by_parameter: bool = False):
#     import matplotlib.pyplot as plt
#     import matplotlib.gridspec as gridspec
    
#     algo_list = get_algo_dict_list()
#     fig = plt.figure(figsize=(12, 10))
#     gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])  # 3 rows, 2 columns

#     # Axes for the 2x2 part
#     axs = [
#         [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
#         [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
#     ]

#     # Axis for the bottom full-width plot
#     ax_total = fig.add_subplot(gs[2, :])  # spans both columns

#     for algo_name in algo_list.keys():
#         mean_doa_errors = np.stack(algos_error_data["mean_doa_errors"][algo_name]) 
#         mse_doa_errors = np.stack(algos_error_data["mean_square_doa_errors"][algo_name])
#         rmse_doa_errors = np.sqrt(mse_doa_errors)
        
#         all_sources_doa_rmse = np.sqrt(np.sum(mse_doa_errors, axis=1))

#         # Source 1
#         axs[0][0].plot(parameter_values, mean_doa_errors[:, 0], label=algo_name, **algo_list[algo_name])
#         axs[1][0].plot(parameter_values, rmse_doa_errors[:, 0], label=algo_name, **algo_list[algo_name])

#         # Source 2
#         axs[0][1].plot(parameter_values, mean_doa_errors[:, 1], label=algo_name, **algo_list[algo_name])
#         axs[1][1].plot(parameter_values, rmse_doa_errors[:, 1], label=algo_name, **algo_list[algo_name])

#         # Total RMSE (source 1 + source 2)
#         if normalize_rmse_by_parameter:
#             all_sources_doa_rmse = all_sources_doa_rmse / parameter_values
#         ax_total.plot(parameter_values, all_sources_doa_rmse, label=algo_name, **algo_list[algo_name])
#     crb_values = np.sqrt(np.stack(algos_error_data["mean_square_doa_errors"]["CRB"]))
#     if normalize_rmse_by_parameter:
#         crb_values = crb_values / parameter_values
#     ax_total.plot(parameter_values, crb_values, 'k--', label='CRB')
#     # Titles and labels
#     axs[0][0].set_title("Source 1: Mean DOA Error (Bias)")
#     axs[0][0].set_ylabel("DOA Error [degrees]")
    
#     axs[0][1].set_title("Source 2: Mean DOA Error (Bias)")
#     axs[0][1].set_ylabel("DOA Error [degrees]")

#     axs[1][0].set_title("Source 1: RMSE DOA Error")
#     axs[1][0].set_ylabel("DOA RMSE [degrees]")

#     axs[1][1].set_title("Source 2: RMSE DOA Error")
#     axs[1][1].set_ylabel("DOA RMSE [degrees]")

#     ax_total.set_title("Both Sources DOA RMSE")
#     if normalize_rmse_by_parameter:
#         ax_total.set_ylabel("DOA RMSE / " + parameter_name)
#     else:
#         ax_total.set_ylabel("DOA RMSE [degrees]")
#     ax_total.set_xlabel(parameter_name + f" {parameter_units}")
#     # Add legends
#     for ax_row in axs:
#         for ax in ax_row:
#             ax.legend()
#             ax.grid(True)
#     ax_total.legend()
#     ax_total.grid(True)
#     plt.tight_layout()
#     return fig
def plot_doa_errors(algos_error_data: dict, parameter_name: str, parameter_units: str, parameter_values: list, normalize_rmse_by_parameter: bool = False):
    import matplotlib.pyplot as plt
    
    algo_list = get_algo_dict_list()
    fig = plt.figure()
    ax = plt.gca()
    for algo_name in algo_list.keys():
        mean_doa_errors = np.stack(algos_error_data["mean_doa_errors"][algo_name]) 
        mse_doa_errors = np.stack(algos_error_data["mean_square_doa_errors"][algo_name])
        rmse_doa_errors = np.sqrt(mse_doa_errors)
        
        all_sources_doa_rmse = np.sqrt(np.sum(mse_doa_errors, axis=1))
        if normalize_rmse_by_parameter:
            all_sources_doa_rmse = all_sources_doa_rmse / parameter_values
        ax.plot(parameter_values, all_sources_doa_rmse, label=algo_name, **algo_list[algo_name])
    crb_values = np.sqrt(np.stack(algos_error_data["mean_square_doa_errors"]["CRB"]))
    if normalize_rmse_by_parameter:
        crb_values = crb_values / parameter_values
    ax.plot(parameter_values, crb_values, 'k--', label='CRB')
    if normalize_rmse_by_parameter:
        ax.set_ylabel("DOA RMSE / " + parameter_name)
    else:
        ax.set_ylabel("DOA RMSE [degrees]")
    ax.set_xlabel(parameter_name + f" {parameter_units}")
    ax.legend()
    ax.grid(True)
    return fig

# def plot_power_errors(algos_error_data: dict, parameter_name: str, parameter_units: str, parameter_values: list, normalize_rmse_by_parameter: bool = False):
#     import matplotlib.pyplot as plt
#     import matplotlib.gridspec as gridspec

#     algo_list = get_algo_dict_list()
#     fig = plt.figure(figsize=(12, 10))
#     gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])  # 3 rows, 2 columns

#     # Axes for the 2x2 part
#     axs = [
#         [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
#         [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
#     ]

#     # Axis for the bottom full-width plot
#     ax_total = fig.add_subplot(gs[2, :])  # spans both columns

#     for algo_name in algo_list.keys():
#         mean_power_errors = np.stack(algos_error_data["mean_power_errors"][algo_name]) 
#         mse_power_errors = np.stack(algos_error_data["mean_square_power_errors"][algo_name])
#         rmse_power_errors = np.sqrt(mse_power_errors)
        
#         all_sources_power_rmse = np.sqrt(np.sum(mse_power_errors, axis=1))

#         # Source 1
#         axs[0][0].plot(parameter_values, mean_power_errors[:, 0], label=algo_name, **algo_list[algo_name])
#         axs[1][0].plot(parameter_values, rmse_power_errors[:, 0], label=algo_name, **algo_list[algo_name])

#         # Source 2
#         axs[0][1].plot(parameter_values, mean_power_errors[:, 1], label=algo_name, **algo_list[algo_name])
#         axs[1][1].plot(parameter_values, rmse_power_errors[:, 1], label=algo_name, **algo_list[algo_name])

#         # Total RMSE (source 1 + source 2)
#         if normalize_rmse_by_parameter:
#             all_sources_power_rmse = all_sources_power_rmse / parameter_values
#         ax_total.plot(parameter_values, all_sources_power_rmse, label=algo_name, **algo_list[algo_name])

#     # Titles and labels
#     axs[0][0].set_title("Source 1: Mean power Error (Bias)")
#     axs[0][0].set_ylabel("power Error")
    
#     axs[0][1].set_title("Source 2: Mean power Error (Bias)")
#     axs[0][1].set_ylabel("power Error")

#     axs[1][0].set_title("Source 1: RMSE power Error")
#     axs[1][0].set_ylabel("power RMSE")

#     axs[1][1].set_title("Source 2: RMSE power Error")
#     axs[1][1].set_ylabel("power RMSE")

#     ax_total.set_title("Both Sources power RMSE")
#     if normalize_rmse_by_parameter:
#         ax_total.set_ylabel("power RMSE / " + parameter_name)
#     else:
#         ax_total.set_ylabel("power RMSE")
#     ax_total.set_xlabel(parameter_name + f" {parameter_units}")
#     # Add legends
#     for ax_row in axs:
#         for ax in ax_row:
#             ax.legend()
#             ax.grid(True)
#     ax_total.legend()
#     ax_total.grid(True)
#     plt.tight_layout()
#     return fig


def plot_power_errors(algos_error_data: dict, parameter_name: str, parameter_units: str, parameter_values: list, normalize_rmse_by_parameter: bool = False):
    import matplotlib.pyplot as plt

    algo_list = get_algo_dict_list()
    fig = plt.figure()
    ax = plt.gca()
    for algo_name in algo_list.keys():
        mean_power_errors = np.stack(algos_error_data["mean_power_errors"][algo_name]) 
        mse_power_errors = np.stack(algos_error_data["mean_square_power_errors"][algo_name])
        rmse_power_errors = np.sqrt(mse_power_errors)
        
        all_sources_power_rmse = np.sqrt(np.sum(mse_power_errors, axis=1))

        if normalize_rmse_by_parameter:
            all_sources_power_rmse = all_sources_power_rmse / parameter_values
        ax.plot(parameter_values, all_sources_power_rmse, label=algo_name, **algo_list[algo_name])

    if normalize_rmse_by_parameter:
        ax.set_ylabel("power RMSE / " + parameter_name)
    else:
        ax.set_ylabel("power RMSE")
    ax.set_xlabel(parameter_name + f" {parameter_units}")

    ax.legend()
    ax.grid(True)
    return fig


# %%
if __name__ == "__main__":
    # %%
    num_mc = 100
    vec_delta_theta = np.arange(2, 11)
    num_configs = len(vec_delta_theta)
    workdir = os.path.abspath('TmpWorkDir5')
    os.makedirs(workdir, exist_ok=True)
    config_list = []
    for i in range(num_configs):
        config_list.append(
            create_config(
                m=12, snr=0, N=20, 
                power_doa_db=np.array([3, 4]),
                doa=np.array([35, 35+vec_delta_theta[i]]),
                cohr_flag=False,
                )
        )
    # %% Run the configurations
    RunDoaConfigsPBS(workdir, config_list, num_mc, num_jobs=DEFAULT_NUM_JOBS)
    # %%
    with open(f'{workdir}/results.pkl', 'rb') as f:
        results = pickle.load(f)
    print("Results loaded from file.")
    # %%
    results, algos_error_data = analyze_algo_errors(results)

    # %%

    algo_list = get_algo_dict_list()
    i_config = 3
    i_mc = 1
    ax = display_power_spectrum(results[i_config][i_mc]["config"], results[i_config][i_mc]["p_vec_list"])
    # %%
    fig_doa_errors = plot_doa_errors(algos_error_data, r'$\Delta \theta$', "degrees", vec_delta_theta, normalize_rmse_by_parameter=True)
    # 
    fig_power_errors = plot_power_errors(algos_error_data, r'$\Delta \theta$', "degrees", vec_delta_theta, normalize_rmse_by_parameter=False)
    # 
    fig_prob_detection = plot_prob_detection(algos_error_data, r'$\Delta \theta$', "degrees", vec_delta_theta)
    
    

# %%
