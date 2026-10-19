# %%
import os
import numpy as np
import pickle
import subprocess
import time
from pathlib import Path
import utils
from utils import *
from collections import defaultdict
from matplotlib.font_manager import FontProperties
import time

NUM_MC = 500
DEFAULT_NUM_JOBS = 600 # 334
# %%
def save_job_metadata(workdir: str, config_list: list, num_mc: int, num_jobs: int):
    if os.path.exists(FILENAME_PBS_METADATA):
        os.remove(FILENAME_PBS_METADATA)
    metadata = {}
    metadata["num_configs"] = len(config_list)
    metadata["num_jobs"] = num_jobs
    metadata["num_mc"] = num_mc
    metadata["workdir"] = workdir
    metadata["grid_step_degrees"] = utils.globalParams.GRID_STEP_DEGREES
    metadata["grid_min_max_vals_degrees"] = utils.globalParams.GRID_MIN_MAX_VALS_DEGREES    
    metadata["wanted_algo_names"] = utils.globalParams.WANTED_ALGO_NAMES
    metadata["sensor_array_type"] = utils.globalParams.SENSOR_ARRAY_TYPE
    metadata["do_store_convergence"] = utils.globalParams.DO_STORE_CONVERGENCE
    
    with open(FILENAME_PBS_METADATA, "wb") as f:
        pickle.dump(metadata, f)

def save_doa_configs(workdir: str, config_list: list):
    for i in range(len(config_list)):
        cfg = config_list[i]
        with open(f"{workdir}/config_{i}.pkl", "wb") as f:
            pickle.dump(cfg, f)

def submit_job_array(workdir: str, config_list: list, num_mc: int, num_jobs: int):

    # delete the directory job_logs if it exists
    job_logs_dir = Path('.') / "job_logs"
    if job_logs_dir.exists():
        import shutil
        shutil.rmtree(job_logs_dir)
    #create the job_logs directory
    os.makedirs(job_logs_dir, exist_ok=True)

    res = subprocess.run(["qsub", "-J", f"0-{num_jobs-1}", FILENAME_PBS_SCRIPT], capture_output=True, text=True)

    if res.returncode != 0:
        print("Error submitting job array:", res.stderr)
        raise RuntimeError("Job submission failed.")

    print("Job submission output:", res.stdout)
    return res.stdout.strip()

# def wait_for_results(workdir: str, config_list: list, num_mc:int, job_id:str, t0: float):
#     expected = len(config_list) * num_mc
#     while True:
#         done = len(list(Path(workdir).glob("config_*_mc_*.pkl")))
#         # workdir is a string of a path, get string of the last two parts of the path
#         workdir_parts = Path(workdir).parts[-2:] 
#         print(f"Job {job_id} Waiting... {workdir_parts},   {done}/{expected} results ready. elapsed time: {time.time() - t0:.2f} [sec]")
#         if done >= expected:
#             break
#         time.sleep(5) 




from tqdm import tqdm
def wait_for_results(workdir: str, config_list: list, num_mc:int, job_id:str, t0: float):
    expected = len(config_list) * num_mc

    def count_results():
        return len(list(Path(workdir).glob("config_*_mc_*.pkl")))

    current = count_results()
    last_update_time = time.time()
    pbar = tqdm(total=expected, initial=current, desc="results ready", unit="file")

    while current < expected:
        new_count = count_results()
        now = time.time()
        # Only update by the difference
        if new_count > current:
            pbar.update(new_count - current)
            current = new_count
            last_update_time = now
        
        elapsed = now - last_update_time
        pbar.set_postfix_str(f"last update: {elapsed:0.1f}s ago")
        time.sleep(5)

    pbar.close()


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
    
    job_logs_dir = Path('.') / "job_logs"
    if job_logs_dir.exists():
        import shutil
        shutil.rmtree(job_logs_dir)
    #create the job_logs directory
    os.makedirs(job_logs_dir, exist_ok=True)

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
    algo_names = results[0][0]["algo_names"]
    algo_list = get_specific_inorder_algo_list(algo_names)
    num_algo = len(algo_list)
    grid_doa = utils.get_doa_grid()
    num_configs = len(results)
    num_mc = len(results[0])
    for i_config in range(num_configs):
        config = results[i_config][0]["config"]
        tmp_doa = np.expand_dims(config["doa"],-1)
        threshold_theta_detect = 5 # np.abs(tmp_doa - tmp_doa.T).max()
        print(f"Config {i_config}: threshold_theta_detect = {threshold_theta_detect}")
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
            l0_norm = [None] * num_algo
            list_HPBW = [None] * num_algo
            for i_alg in range(num_algo):
                num_detected[i_alg], _, _, selected_doa_error[i_alg], selected_power_error[i_alg], \
                    succ_match_detected_doa[i_alg], succ_match_true_doa[i_alg], list_HPBW[i_alg] =\
                    estimate_doa_calc_errors(
                        result["p_vec_list"][i_alg], grid_doa,
                        result["config"]["doa"],
                        convert_db_to_linear(result["config"]["power_doa_db"]), threshold_theta_detect=threshold_theta_detect)
                l0_norm[i_alg] = thresholded_l0_norm(result["p_vec_list"][i_alg], threshold=convert_db_to_linear(np.min(config["power_doa_db"]))*0.01)

            result["num_detected"] = num_detected
            result["selected_doa_error"] = selected_doa_error
            result["selected_power_error"] = selected_power_error
            result["succ_match_detected_doa"] = succ_match_detected_doa
            result["succ_match_true_doa"] = succ_match_true_doa
            result["l0_norm"] = l0_norm
            result["list_HPBW"] = list_HPBW
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
            # inds = indices_mc_all_algos_detected_enough_sources
            # inds = [
            #     i_mc for i_mc in range(num_mc)
            #     if results[i_config][i_mc]["num_detected"][i_algo] >= len(config["doa"])
            #     ]
            inds = [
                i_mc for i_mc in range(num_mc)
                ]
            if len(inds) < 0.4 * num_mc:
                # If not enough MC iterations to look at, this config is irelevant
                doa_errors = np.expand_dims(results[i_config][0]["selected_doa_error"][i_algo] * np.nan, axis=0)
                power_errors = np.expand_dims(results[i_config][0]["selected_power_error"][i_algo] * np.nan, axis=0)
            else:
                doa_errors = np.stack([results[i_config][i_mc]["selected_doa_error"][i_algo] 
                                            for i_mc in inds])
                power_errors = np.stack([results[i_config][i_mc]["selected_power_error"][i_algo] 
                                            for i_mc in inds])
                mean_square_errors = np.mean(doa_errors**2, axis=1)

                # prcnt_worst_results_to_ignore = 2
                # index = np.ceil((100 - prcnt_worst_results_to_ignore) / 100 * len(mean_square_errors)).astype(int)
                # bottom_q_percent_indices = np.argsort(mean_square_errors)[:index]
                # doa_errors = np.stack([doa_errors[i] for i in bottom_q_percent_indices])
                # power_errors = np.stack([power_errors[i] for i in bottom_q_percent_indices])

                median_ = np.median(mean_square_errors)
                inds_to_remove = np.where(mean_square_errors > 1000*(median_+1e-1))[0]
                prcnt_outliers = len(inds_to_remove) / len(mean_square_errors) * 100
                if prcnt_outliers > 15: 
                    print(f"ERROR: {prcnt_outliers} % outliers detected in config {i_config}, algo {algo_name}. ")
                    doa_errors = np.nan * np.ones_like(doa_errors)
                    power_errors = np.nan * np.ones_like(power_errors)
                elif prcnt_outliers > 0:
                    print(f"WARNING: {prcnt_outliers} % outliers detected in config {i_config}, algo {algo_name}. Removing them.")
                    doa_errors = np.delete(doa_errors, inds_to_remove, axis=0)
                    power_errors = np.delete(power_errors, inds_to_remove, axis=0)


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
def plot_l0_norm(result):
    # for each algorithm: display a boxplot of the l0_norm
    import matplotlib.pyplot as plt
    algo_list = get_algo_dict_list()
    num_sources = len(result[0]["config"]["doa"])
    fig, ax = plt.subplots()
    for i_algo, algo_name in enumerate(algo_list.keys()):
        l0_norm = np.stack([result[i_mc]["l0_norm"][i_algo] for i_mc in range(len(result))])
        ax.boxplot(l0_norm, positions=[i_algo], widths=0.5, patch_artist=True,
                   whis=[10,90],
                   showfliers=False,
                   boxprops=dict(facecolor='lightblue', color='black'),
                   medianprops=dict(color='red'), whiskerprops=dict(color='black'))
    # add a line for the number of sources
    ax.axhline(y=num_sources, color='green', linestyle=':', label=f'Number of Sources = {num_sources}')
    ax.set_xticks(range(len(algo_list)))
    ax.set_xticklabels(algo_list.keys())
    ax.set_ylabel("$l_0$ Norm")
    ax.legend()
    return fig

def plot_hpbw(result):
    # for each algorithm: display a boxplot of the hpbw
    import matplotlib.pyplot as plt
    algo_list = get_algo_dict_list()
    fig, ax = plt.subplots()
    for i_algo, algo_name in enumerate(algo_list.keys()):
        hpbw = [result[i_mc]["list_HPBW"][i_algo] for i_mc in range(len(result))]
        # hpbw is now a list of lists. create a single list of hpbw values
        hpbw = np.array([item for sublist in hpbw for item in sublist])
        ax.boxplot(hpbw, positions=[i_algo], widths=0.5, patch_artist=True, 
                   whis=[10,90],
                   showfliers=False,
                   boxprops=dict(facecolor='lightblue', color='black'),
                   medianprops=dict(color='red'), whiskerprops=dict(color='black'))
    grid_res = np.median(np.diff(get_doa_grid()))
    ax.axhline(y=grid_res, color='green', linestyle=':', label=f'Grid Resolution = {grid_res:.2f} degrees')
    ax.set_xticks(range(len(algo_list)))
    ax.set_xticklabels(algo_list.keys())
    ax.set_ylabel("HPBW (degrees)")
    ax.legend()
    return fig
    
def plot_Qeigvals(results, parameter_name: str, parameter_units: str, parameter_values: list,
                    do_ylogscale: bool = False):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.gca()
    # %%
    num_configs = len(results)
    num_mc = len(results[0])
    # Collect all eigenvalues for each config and MC
    all_eigvals = []
    for i_config in range(num_configs):
        config_eigvals = []
        for i_mc in range(num_mc):
            eigvals = eigvals_of_Q_given_result(results[i_config][i_mc])
            config_eigvals.append(eigvals)
        all_eigvals.append(np.array(config_eigvals))  # shape: (num_mc, num_eigvals)
    all_eigvals = np.array(all_eigvals)  # shape: (num_configs, num_mc, num_eigvals)
    num_eigvals = all_eigvals.shape[2]

    # Compute mean and std for each eigenvalue index across MC runs, for each config
    mean_eigvals = np.mean(all_eigvals, axis=1)  # shape: (num_configs, num_eigvals)
    std_eigvals = np.std(all_eigvals, axis=1)    # shape: (num_configs, num_eigvals)

    # # Plot each eigenvalue's mean and std as a function of parameter_values
    # # select continous color map (not viridis)
    # cmap = plt.get_cmap('plasma')  # or 'inferno', 'magma', etc.
    # for i_eig in range(num_eigvals):
    #     # ax.plot(parameter_values, mean_eigvals[:, i_eig], marker='o', label=fr"Mean $\lambda_{{{i_eig+1}}}$")
    #     ax.fill_between(parameter_values, 
    #                     mean_eigvals[:, i_eig] - std_eigvals[:, i_eig], 
    #                     mean_eigvals[:, i_eig] + std_eigvals[:, i_eig],
    #                     alpha=0.3, label=fr"$\lambda_{{{i_eig+1}}}$ ± Std", color=cmap(i_eig / num_eigvals))

    bar_eigvals_mat = np.mean(all_eigvals, axis=2)  # shape: (num_configs, num_mc)
    # print(f"bar_eigvals_mat shape: {bar_eigvals_mat.shape}")
    mean_bar_eigvals = np.mean(bar_eigvals_mat, axis=1)  # shape: (num_configs,)
    std_bar_eigvals = np.std(bar_eigvals_mat, axis=1)
    # Plot mean and std of bar eigenvalues
    ax.plot(parameter_values, mean_bar_eigvals, marker='o', label=r"$\bar{\lambda}$: Mean")
    ax.fill_between(parameter_values,
                    mean_bar_eigvals - std_bar_eigvals,
                    mean_bar_eigvals + std_bar_eigvals,
                    alpha=0.3, label=r"$\bar{\lambda}$: Mean ± Std")
    
    # %%
    ax.grid(True)
    if do_ylogscale:
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--')

    ax.set_ylabel("Q Eigenvalues")
    ax.set_xlabel(parameter_name + f" {parameter_units}")
    # ax.legend()
    return fig
    
    # # ----------------------
    # # bar_eigvals shape: (num_configs, num_mc)
    # #"selected_doa_error"
    # algo_list = get_algo_dict_list()
    # fig = plt.figure()
    # ax = plt.gca()
    # bar_lambda_vals = [bar_eigvals_mat[i_config, i_mc]
    #                    for i_config in range(num_configs)
    #                    for i_mc in range(num_mc)]
    # for i_algo, algo_name in enumerate(algo_list.keys()): 
    #     rmse_doa_vals = [np.sqrt(np.mean(results[i_config][i_mc]["selected_doa_error"][i_algo]**2)) 
    #                        for i_config in range(num_configs) 
    #                        for i_mc in range(num_mc)]
    #     ax.scatter(bar_lambda_vals, rmse_doa_vals, label=algo_name, c=algo_list[algo_name]["color"], alpha=0.1) 
    # ax.set_xlabel(r"$\bar{\lambda}$")
    # ax.set_ylabel("RMSE DOA Error (degrees)")
    
    # return fig



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

def plot_doa_errors_per_source(algos_error_data: dict, parameter_name: str, parameter_units: str, parameter_values: list):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    algo_list = get_algo_dict_list()

    num_sources = algos_error_data["mean_square_doa_errors"]["CRB"][0].shape[0]
    fig = plt.figure()
    gs = gridspec.GridSpec(num_sources, 2, height_ratios=[1, 1, 1])  # 3 rows, 2 columns

    # Axes for the 2x2 part
    axs = [
        [fig.add_subplot(gs[i_source, 0]), fig.add_subplot(gs[i_source, 1])] for i_source in range(num_sources)
    ]

    crb_values = np.stack(algos_error_data["mean_square_doa_errors"]["CRB"])
    for algo_name in algo_list.keys():
        mean_doa_errors = np.stack(algos_error_data["mean_doa_errors"][algo_name]) 
        mse_doa_errors = np.stack(algos_error_data["mean_square_doa_errors"][algo_name])        

        for i_source in range(num_sources):
            # Plot mean DOA errors and RMSE for each source
            axs[i_source][0].plot(parameter_values, mean_doa_errors[:, i_source], label=algo_name, **algo_list[algo_name])
            axs[i_source][1].plot(parameter_values, mse_doa_errors[:, i_source], label=algo_name, **algo_list[algo_name])
            axs[i_source][1].plot(parameter_values, crb_values[:, i_source], 'k--', label='CRB')

    # Titles and labels
    for i_source in range(num_sources):
        axs[i_source][0].set_title(f"Source {i_source+1}: DOA Bias")
        axs[i_source][0].set_ylabel("Bias [degrees]")
        axs[i_source][1].set_title(f"Source {i_source+1}: DOA MSE")
        axs[i_source][1].set_ylabel("MSE [degrees^2]")
        axs[i_source][0].grid(True)
        axs[i_source][1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_power_errors_per_source(algos_error_data: dict, parameter_name: str, parameter_units: str, parameter_values: list):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    algo_list = get_algo_dict_list()

    num_sources = algos_error_data["mean_square_doa_errors"]["CRB"][0].shape[0]
    fig = plt.figure()
    gs = gridspec.GridSpec(num_sources, 2, height_ratios=[1, 1, 1])  # 3 rows, 2 columns

    # Axes for the 2x2 part
    axs = [
        [fig.add_subplot(gs[i_source, 0]), fig.add_subplot(gs[i_source, 1])] for i_source in range(num_sources)
    ]

    crb_values = np.stack(algos_error_data["mean_square_doa_errors"]["CRB"])
    for algo_name in algo_list.keys():
        mean_power_errors = np.stack(algos_error_data["mean_power_errors"][algo_name]) 
        mse_power_errors = np.stack(algos_error_data["mean_square_power_errors"][algo_name])        

        for i_source in range(num_sources):
            # Plot mean DOA errors and RMSE for each source
            axs[i_source][0].plot(parameter_values, mean_power_errors[:, i_source], label=algo_name, **algo_list[algo_name])
            axs[i_source][1].plot(parameter_values, mse_power_errors[:, i_source], label=algo_name, **algo_list[algo_name])

    # Titles and labels
    for i_source in range(num_sources):
        axs[i_source][0].set_title(f"Source {i_source+1}: Power Bias")
        axs[i_source][0].set_ylabel("Bias [linear]")
        axs[i_source][1].set_title(f"Source {i_source+1}: Power MSE")
        axs[i_source][1].set_ylabel("MSE [linear^2]")
        axs[i_source][0].grid(True)
        axs[i_source][1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_doa_errors(algos_error_data: dict, parameter_name: str, parameter_units: str, parameter_values: list, normalize_rmse_by_parameter: bool = False,
                    do_ylogscale: bool = False, plot_on_ax=None, do_legend: bool = True):
    import matplotlib.pyplot as plt
    
    algo_names = list(algos_error_data["mean_doa_errors"].keys())
    algo_list = get_specific_inorder_algo_list(algo_names)
    # algo_list = get_algo_dict_list()
    if plot_on_ax is not None:
        fig = []
        ax = plot_on_ax
    else:
        fig = plt.figure()
        ax = plt.gca()
    for algo_name in algo_list.keys():
        mean_doa_errors = np.stack(algos_error_data["mean_doa_errors"][algo_name]) 
        mse_doa_errors = np.stack(algos_error_data["mean_square_doa_errors"][algo_name])
        
        doa_mse = mse_doa_errors
        doa_mse_mean = np.mean(doa_mse, axis=1)
        doa_root_mse_mean = np.sqrt(doa_mse_mean)

        if any(np.isnan(doa_root_mse_mean)):
            continue

        if normalize_rmse_by_parameter:
            doa_root_mse_mean = doa_root_mse_mean / parameter_values
        
        label = f"{ALGONAME}({algo_name})" if (algo_name == "AIRM" or algo_name == "JBLD" or algo_name == "LE") else algo_name
        pltline = ax.plot(parameter_values, doa_root_mse_mean, label=label, **algo_list[algo_name])

        qlow = np.sqrt(np.percentile(doa_mse, 25, axis=1))
        qhigh = np.sqrt(np.percentile(doa_mse, 75, axis=1))

        # doa_mse_se = np.std(doa_mse, axis=1, ddof=1)/np.sqrt(doa_mse.shape[1])
        # qlow = doa_mse_mean - 1.96*doa_mse_se
        # qhigh = doa_mse_mean + 1.96*doa_mse_se
        # qlow = np.sqrt(np.maximum(qlow, 0))
        # qhigh = np.sqrt(qhigh)

        ax.fill_between(parameter_values, qlow, qhigh, color=pltline[0].get_color(), alpha=0.10, linewidth=0.5)

        if do_ylogscale:
            ax.set_yscale('log')
            ax.grid(True, which='both', linestyle='--')
    crb_values = np.stack(algos_error_data["mean_square_doa_errors"]["CRB"])
    lower_bound_all_sources_doa_rmse = np.sqrt(np.mean(crb_values, axis=1))
    # lower_bound_all_sources_doa_rmse = np.mean(crb_values, axis=1)
    xylabel_fontsize = 12
    if normalize_rmse_by_parameter:
        lower_bound_all_sources_doa_rmse = lower_bound_all_sources_doa_rmse / parameter_values
    ax.plot(parameter_values, lower_bound_all_sources_doa_rmse, '--', color='gray', label='CRB', linewidth=2.5)
    if normalize_rmse_by_parameter:
        ax.set_ylabel("DOA RMSE / " + parameter_name, fontsize=xylabel_fontsize)
    else:
        ax.set_ylabel(r"$\mathrm{RMSE}_{\mathrm{DOA}}$ (degrees)", fontsize=xylabel_fontsize)
    ax.set_xlabel(parameter_name + f" {parameter_units}", fontsize=xylabel_fontsize)
    
    if do_legend:
        lgd = ax.legend()
        for text in lgd.get_texts():
            if "JBLD" in text.get_text():
                text.set_fontweight("bold")
    ax.grid(True)
    return fig


def plot_doa_boxplots(algos_error_data, parameter_values, parameter_vals_to_show = None, do_ylogscale=False):
    
    import matplotlib.patches as mpatches

    algo_list = get_algo_dict_list()

    indices_to_show = np.arange(len(parameter_values))
    if parameter_vals_to_show is not None:
        # find indices of parameter_values that are in parameter_vals_to_show
        indices_to_show = [i for i, val in enumerate(parameter_values) if val in parameter_vals_to_show]
        parameter_values = [parameter_values[i] for i in indices_to_show]

    algo_names = list(algo_list.keys())
    num_algos = len(algo_names)
    num_snrs = len(parameter_values)

    fig, ax = plt.subplots(figsize=(8, 4))

    x_base = np.arange(num_snrs)
    group_width = 0.8                     # total width used by all algos per SNR
    box_width  = 0.8*group_width / num_algos  # width per algo

    legend_patches = []

    for j, algo_name in enumerate(algo_names):
        # (num_snrs, num_trials)
        mse = np.stack(algos_error_data["mean_square_doa_errors"][algo_name])
        rmse = np.sqrt(mse)

        # boxplot data: list over SNR, each is vector over trials
        box_data = [rmse[i, :] for i in indices_to_show]


        # positions for this algo inside each SNR group
        positions = x_base - group_width/2 + (j + 0.5) * box_width

        # try to reuse the color from your line-style dict
        style_dict = algo_list[algo_name]
        color = style_dict.get("color", f"C{j}")

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=box_width * 0.9,
            patch_artist=True,
            showfliers=False,  # cleaner
            boxprops=dict(facecolor=color, alpha=0.5),
            medianprops=dict(color="k", linewidth=1.2),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )

        legend_patches.append(mpatches.Patch(color=color, alpha=0.5, label=algo_name))

    ax.set_xticks(x_base)
    ax.set_xticklabels(parameter_values)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSE (deg)")

    if do_ylogscale:
        ax.set_yscale("log")

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # compact legend
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    return fig

def plot_power_errors(algos_error_data: dict, parameter_name: str, parameter_units: str, parameter_values: list, normalize_rmse_by_parameter: bool = False,
                    do_ylogscale: bool = False, plot_on_ax=None, do_legend: bool = True):
    import matplotlib.pyplot as plt

    
    algo_names = list(algos_error_data["mean_doa_errors"].keys())
    algo_list = get_specific_inorder_algo_list(algo_names)
    # algo_list = get_algo_dict_list()
    if plot_on_ax is not None:
        fig = []
        ax = plot_on_ax
    else:
        fig = plt.figure()
        ax = plt.gca()
    for algo_name in algo_list.keys():
        mean_power_errors = np.stack(algos_error_data["mean_power_errors"][algo_name]) 
        mse_power_errors = np.stack(algos_error_data["mean_square_power_errors"][algo_name])
        
        power_mse = mse_power_errors
        power_mse_mean = np.mean(power_mse, axis=1)
        power_root_mse_mean = np.sqrt(power_mse_mean)

        if any(np.isnan(power_root_mse_mean)):
            continue

        if normalize_rmse_by_parameter:
            power_root_mse_mean = power_root_mse_mean / parameter_values

        label = f"{ALGONAME}({algo_name})" if (algo_name == "AIRM" or algo_name == "JBLD" or algo_name == "LE") else algo_name
        pltline = ax.plot(parameter_values, power_root_mse_mean, label=label, **algo_list[algo_name])

        qlow = np.sqrt(np.percentile(power_mse, 25, axis=1))
        qhigh = np.sqrt(np.percentile(power_mse, 75, axis=1))
        ax.fill_between(parameter_values, qlow, qhigh, color=pltline[0].get_color(), alpha=0.10, linewidth=0.5)

    if do_ylogscale:
        ax.set_yscale('log')
    xylabel_fontsize = 12
    if normalize_rmse_by_parameter:
        ax.set_ylabel("power RMSE / " + parameter_name, fontsize=xylabel_fontsize)
    else:
        ax.set_ylabel(r"$\mathrm{RMSE}_{\mathrm{Power}}$ (linear)", fontsize=xylabel_fontsize)
    ax.set_xlabel(parameter_name + f" {parameter_units}", fontsize=xylabel_fontsize)

    if do_legend:
        lgd = ax.legend()
        for text in lgd.get_texts():
            if "JBLD" in text.get_text():
                text.set_fontweight("bold")
    ax.grid(True)
    return fig

def plot_initialization_convergence(results, list_method_for_p_init, algo_name="JBLD"):
    """
    Plot the convergence (median and IQR over MC runs) of `algo_name` for each initialization method.
    results[i_config] corresponds to list_method_for_p_init[i_config].
    Each results[i_config][i_mc]["tuple_convergence_list"][algo_idx] is (loss_history, rel_change_history).
    Runs that converged early are padded with their last value.
    Returns (fig_loss, fig_rel_change).
    """
    import matplotlib.pyplot as plt

    def stack_padded(list_histories):
        max_len = max(len(h) for h in list_histories)
        mat = np.full((len(list_histories), max_len), np.nan)
        for i, h in enumerate(list_histories):
            mat[i, :len(h)] = h
            mat[i, len(h):] = h[-1]
        return mat

    xylabel_fontsize = 12
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
    fig_rel_change, ax_rel_change = plt.subplots(figsize=(6, 4))
    for i_config, method_name in enumerate(list_method_for_p_init):
        algo_idx = results[i_config][0]["algo_names"].index(algo_name)
        list_tuples = [results[i_config][i_mc]["tuple_convergence_list"][algo_idx] for i_mc in range(len(results[i_config]))]
        list_tuples = [t for t in list_tuples if t is not None and len(t[0]) > 0]
        if len(list_tuples) == 0:
            print(f"plot_initialization_convergence: no convergence data for {algo_name} with init {method_name}")
            continue

        for ax, i_hist in [(ax_loss, 0), (ax_rel_change, 1)]:
            mat = stack_padded([t[i_hist] for t in list_tuples])  # Shape: (num_mc, max_iters)
            if i_hist == 0:
                mat = mat - mat[:, [-1]] + 1e-12   # suboptimality per run
                # mat = mat / mat[:, [0]]  # normalize each loss trajectory by its first value
            q25, q50, q75 = np.percentile(mat, [25, 50, 75], axis=0)
            iters = np.arange(1, mat.shape[1] + 1)
            line, = ax.plot(iters, q50, label=method_name)
            ax.fill_between(iters, q25, q75, color=line.get_color(), alpha=0.25)

    ax_loss.set_ylabel('Normalized loss', fontsize=xylabel_fontsize)
    ax_loss.set_yscale('log')
    ax_rel_change.set_ylabel('Relative change', fontsize=xylabel_fontsize)
    ax_rel_change.set_yscale('log')
    for ax in [ax_loss, ax_rel_change]:
        ax.set_xlabel('Iteration', fontsize=xylabel_fontsize)
        ax.set_xscale('log')
        ax.set_title(f"{ALGONAME}({algo_name}): median and IQR")
        ax.legend(title=r'$p_{init}$')
        ax.grid(True, linestyle='--', alpha=0.4, which='both')
    fig_loss.tight_layout()
    fig_rel_change.tight_layout()
    return fig_loss, fig_rel_change

    

def compute_initialization_paired_metric(results, list_method_for_p_init, algo_name="JBLD", ref_method=None, metric="rel_l2"):
    """
    Paired comparison of the final result of `algo_name` across initialization methods.
    The MC seed equals i_mc in every config, so run i_mc sees the same data for all init methods.
    For each non-reference method computes, per MC run:
        metric="rel_l2":  ||p_method - p_ref|| / ||p_ref||
        metric="max_abs": max(|p_method - p_ref|)
        metric="loss":    |L_method - L_ref| / |L_ref|, L = final loss (requires DO_STORE_CONVERGENCE)
        metric="doa":     max |doa_hat_method - doa_hat_ref| [deg], after matching the two sets of
                          estimated DOAs to each other with linear_sum_assignment (ground truth is not used)
        metric="power":   max |power_hat_method - power_hat_ref| [dB] over the peaks matched as in "doa"
    Returns dict_metric where dict_metric[method] is an array of shape (num_mc,).
    """
    from scipy.optimize import linear_sum_assignment
    if ref_method is None:
        ref_method = list_method_for_p_init[0]
    i_ref = list_method_for_p_init.index(ref_method)
    num_mc = min(len(results[i_config]) for i_config in range(len(list_method_for_p_init)))
    grid_doa = utils.get_doa_grid()

    def get_result_and_algo_idx(i_config, i_mc):
        result = results[i_config][i_mc]
        assert result["i_mc"] == i_mc, "results are not ordered by i_mc, pairing is invalid"
        return result, result["algo_names"].index(algo_name)

    def get_p_vec(i_config, i_mc):
        result, algo_idx = get_result_and_algo_idx(i_config, i_mc)
        return np.ravel(np.asarray(result["p_vec_list"][algo_idx], dtype=float))

    def get_final_loss(i_config, i_mc):
        result, algo_idx = get_result_and_algo_idx(i_config, i_mc)
        tuple_convergence = result["tuple_convergence_list"][algo_idx]
        if tuple_convergence is None or len(tuple_convergence[0]) == 0:
            raise ValueError("No loss history stored, run with DO_STORE_CONVERGENCE = True")
        return tuple_convergence[0][-1]

    def get_selected_doas_and_powers(i_config, i_mc):
        config = results[i_config][i_mc]["config"]
        _, selected_detected_doas, selected_detected_powers, _, _, _, _, _ = estimate_doa_calc_errors(
            get_p_vec(i_config, i_mc), grid_doa, config["doa"], convert_db_to_linear(config["power_doa_db"]))
        return selected_detected_doas, selected_detected_powers

    def max_matched_diff(i_config_a, i_config_b, i_mc):
        # match the two sets of estimated DOAs to each other (the ground truth is not used)
        doas_a, powers_a = get_selected_doas_and_powers(i_config_a, i_mc)
        doas_b, powers_b = get_selected_doas_and_powers(i_config_b, i_mc)
        cost_matrix = np.abs(doas_a[:, np.newaxis] - doas_b[np.newaxis, :])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        max_doa_diff = np.max(cost_matrix[row_ind, col_ind])
        max_power_diff_db = np.max(np.abs(convert_linear_to_db(powers_a[row_ind]) - convert_linear_to_db(powers_b[col_ind])))
        return max_doa_diff, max_power_diff_db

    dict_metric = {}
    for i_config, method_name in enumerate(list_method_for_p_init):
        if i_config == i_ref:
            continue
        values = np.zeros(num_mc)
        for i_mc in range(num_mc):
            if metric == "rel_l2":
                p_ref, p_method = get_p_vec(i_ref, i_mc), get_p_vec(i_config, i_mc)
                values[i_mc] = np.linalg.norm(p_method - p_ref) / np.linalg.norm(p_ref)
            elif metric == "max_abs":
                p_ref, p_method = get_p_vec(i_ref, i_mc), get_p_vec(i_config, i_mc)
                values[i_mc] = np.max(np.abs(p_method - p_ref))
            elif metric == "loss":
                loss_ref, loss_method = get_final_loss(i_ref, i_mc), get_final_loss(i_config, i_mc)
                values[i_mc] = np.abs(loss_method - loss_ref) / np.abs(loss_ref)
            elif metric == "doa":
                values[i_mc] = max_matched_diff(i_config, i_ref, i_mc)[0]
            elif metric == "power":
                values[i_mc] = max_matched_diff(i_config, i_ref, i_mc)[1]
            else:
                raise ValueError(f"Unknown metric: {metric}")
        dict_metric[method_name] = values
    return dict_metric


def display_initialization_spectra(results, list_method_for_p_init, i_mc, algo_name="JBLD", str_title=None):
    """
    Display the spectra of `algo_name` for MC run i_mc, one curve per initialization method.
    """
    import matplotlib.pyplot as plt
    list_p_vec = []
    algo_list = {}
    for i_config, method_name in enumerate(list_method_for_p_init):
        result = results[i_config][i_mc]
        assert result["i_mc"] == i_mc, "results are not ordered by i_mc, pairing is invalid"
        algo_idx = result["algo_names"].index(algo_name)
        list_p_vec.append(np.array(result["p_vec_list"][algo_idx], dtype=float))  # copy, display_power_spectrum clips in place
        algo_list[f"init: {method_name}"] = {"color": f"C{i_config}", "linestyle": ["-", "--", ":", "-."][i_config % 4]}

    config = results[0][i_mc]["config"]
    fig, ax = plt.subplots(figsize=(7, 4))
    display_power_spectrum(config, list_p_vec, algo_list=algo_list, ax=ax)
    DELTA_X = 10
    ax.set_xlim([np.min(config["doa"]) - DELTA_X, np.max(config["doa"]) + DELTA_X])
    ax.set_ylim([-20, np.max(config["power_doa_db"]) + 3])
    ax.set_title(str_title if str_title is not None else f"{ALGONAME}({algo_name}), MC run {i_mc}")
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    return fig


def plot_initialization_paired_difference(results, list_method_for_p_init, algo_name="JBLD", ref_method=None,
                                          plot_type="cdf", logscale=None, metric="rel_l2", num_worst_to_display=0,
                                          show_fliers=True, doa_gross_threshold=None):
    """
    Plot the paired metric of compute_initialization_paired_metric (see there for the metric options).
    plot_type: "cdf" (empirical CDF per method) or "boxplot".
    logscale: None -> log for all metrics except "doa" (which is grid-quantized and often exactly 0).
    num_worst_to_display: for each method, also display the spectra of its num_worst_to_display worst MC runs
        (chosen from all runs, including the excluded ones).
    doa_gross_threshold: only for metric="doa" or "power". Runs with a DOA difference > doa_gross_threshold [deg] are
        counted (reported in the title) and excluded from the plot and statistics. None -> exclude nothing.
    Returns (fig, dict_metric, list_fig_worst, df_doa_table). dict_metric holds all runs (nothing excluded).
    df_doa_table: only for metric="doa" (else None). One row per non-reference method, columns are the percentage
        of all runs (nothing excluded) with a DOA difference of exactly 0 / 1 grid step, or more than 1 step.
    """
    import matplotlib.pyplot as plt

    if ref_method is None:
        ref_method = list_method_for_p_init[0]
    if logscale is None:
        logscale = (metric != "doa")
    dict_metric = compute_initialization_paired_metric(results, list_method_for_p_init, algo_name, ref_method, metric)

    # runs to exclude from the plot and statistics (gross DOA disagreements)
    do_exclude = metric in ["doa", "power"] and doa_gross_threshold is not None
    if do_exclude:
        dict_doa_diff = dict_metric if metric == "doa" else \
            compute_initialization_paired_metric(results, list_method_for_p_init, algo_name, ref_method, "doa")
    dict_is_excluded = {}
    for method_name, values in dict_metric.items():
        if do_exclude:
            dict_is_excluded[method_name] = dict_doa_diff[method_name] > doa_gross_threshold + 1e-6
        else:
            dict_is_excluded[method_name] = np.zeros(len(values), dtype=bool)
    dict_metric_kept = {m: dict_metric[m][~dict_is_excluded[m]] for m in dict_metric}

    xylabel_fontsize = 12
    str_ref = r'\mathrm{' + ref_method.replace('_', r'\_') + r'}'
    dict_str_metric = {
        "rel_l2": r'$\|\hat{p}_{\mathrm{init}} - \hat{p}_{' + str_ref + r'}\| / \|\hat{p}_{' + str_ref + r'}\|$',
        "max_abs": r'$\max|\hat{p}_{\mathrm{init}} - \hat{p}_{' + str_ref + r'}|$',
        "loss": r'$|L_{\mathrm{init}} - L_{' + str_ref + r'}| / |L_{' + str_ref + r'}|$',
        "doa": r'$\max_k |\hat{\theta}_{k,\mathrm{init}} - \hat{\theta}_{k,' + str_ref + r'}|$ (degrees)',
        "power": r'$\max_k |\hat{P}_{k,\mathrm{init}} - \hat{P}_{k,' + str_ref + r'}|$ (dB)',
    }
    str_metric = dict_str_metric[metric]
    fig, ax = plt.subplots(figsize=(6, 4))
    if plot_type == "cdf":
        for method_name, values in dict_metric_kept.items():
            x = np.sort(values)
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, y, where='post', label=f'{method_name} vs. {ref_method}')
        ax.set_xlabel(str_metric, fontsize=xylabel_fontsize)
        ax.set_ylabel('Empirical CDF', fontsize=xylabel_fontsize)
        ax.set_ylim([0, 1.02])
        if logscale:
            ax.set_xscale('log')
        ax.legend()
    elif plot_type == "boxplot":
        list_methods = list(dict_metric_kept.keys())
        ax.boxplot([dict_metric_kept[m] for m in list_methods], showfliers=show_fliers)
        for method_name in list_methods:
            values = dict_metric_kept[method_name]
            q25, q75 = np.percentile(values, [25, 75])
            num_fliers = np.sum((values < q25 - 1.5 * (q75 - q25)) | (values > q75 + 1.5 * (q75 - q25)))
            print(f"boxplot {method_name} vs. {ref_method}: {num_fliers}/{len(values)} runs are outliers (beyond 1.5 IQR)")
        ax.set_xticks(np.arange(1, len(list_methods) + 1))
        ax.set_xticklabels([f'{m} vs. {ref_method}' for m in list_methods], fontsize=xylabel_fontsize)
        ax.set_ylabel(str_metric, fontsize=xylabel_fontsize)
        if logscale:
            ax.set_yscale('log')
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    str_title = f"{ALGONAME}({algo_name}): paired difference of final results"
    if do_exclude:
        str_excluded = ", ".join(f"{m}: {np.sum(dict_is_excluded[m])}/{len(dict_is_excluded[m])}" for m in dict_is_excluded)
        str_title += f"\nexcluded runs with DOA difference > {doa_gross_threshold}°: {str_excluded}"
    ax.set_title(str_title)
    ax.grid(True, linestyle='--', alpha=0.4, which='both')
    fig.tight_layout()

    df_doa_table = None
    if metric == "doa":
        import pandas as pd
        # detected DOAs are on-grid, so the differences are integer multiples of the grid step
        step_deg = np.diff(utils.get_doa_grid())[0]
        list_columns = ["diff=0", f"diff={step_deg:g}°", f"diff>{step_deg:g}°"]
        dict_rows = {}
        for method_name, values in dict_metric.items():
            num_steps = np.round(values / step_deg).astype(int)
            dict_rows[f"{method_name} vs. {ref_method}"] = \
                [100 * np.mean(num_steps == 0), 100 * np.mean(num_steps == 1), 100 * np.mean(num_steps > 1)]
        df_doa_table = pd.DataFrame.from_dict(dict_rows, orient="index", columns=list_columns)
        print(f"{ALGONAME}({algo_name}): DOA difference vs. {ref_method}, % of all runs:")
        print(df_doa_table.to_string(float_format=lambda x: f"{x:.1f}%"))

    list_fig_worst = []
    for method_name, values in dict_metric.items():
        values_kept = dict_metric_kept[method_name]
        q25, q50, q75 = np.percentile(values_kept, [25, 50, 75])
        i_mc_worst = np.argsort(values)[::-1][:max(num_worst_to_display, 3)]
        print(f"[{metric}] {method_name} vs. {ref_method}: excluded {np.sum(dict_is_excluded[method_name])}/{len(values)} runs. "
              f"Over kept runs: median = {q50:.2e}, IQR = [{q25:.2e}, {q75:.2e}], max = {values_kept.max():.2e}, "
              f"fraction of exact zeros = {np.mean(values_kept == 0):.2f}. "
              f"Worst MC runs (all runs) = {i_mc_worst.tolist()}")
        if logscale and np.any(values_kept <= 0):
            print(f"  note: {np.sum(values_kept <= 0)} non-positive values are not shown on the log scale")
        for i_mc in i_mc_worst[:num_worst_to_display]:
            str_title = f"MC run {i_mc}: {metric} ({method_name} vs. {ref_method}) = {values[i_mc]:.2e}"
            list_fig_worst.append(display_initialization_spectra(results, list_method_for_p_init, i_mc, algo_name, str_title))
    return fig, dict_metric, list_fig_worst, df_doa_table


def plot_iteration_and_runtime_boxplot(results, param_vals, param_name, DO_BOXPLOT=True, logscale_y=True):
    
    algo_names = results[0][0]["algo_names"]
    REMOVE_ESPRIT = True
    if REMOVE_ESPRIT and "ESPRIT" in algo_names:
        esprit_index = algo_names.index("ESPRIT")
        for i_config in range(len(results)):
            for i_mc in range(len(results[i_config])):
                # remove ESPRIT data
                for key in ['runtime_list', 'num_iters_list', 'p_vec_list', 'algo_names']:
                    del results[i_config][i_mc][key][esprit_index]



    algo_list = get_specific_inorder_algo_list(algo_names)


    num_configs = len(results)
    num_mc = len(results[0])
    num_algos = len(algo_list)
    num_iters_mat = np.zeros((num_configs, num_mc, num_algos))
    runtime_mat = np.zeros((num_configs, num_mc, num_algos))
    for i_config in range(len(results)):
        num_iters_mat[i_config,:,:] = np.array([results[i_config][i_mc]["num_iters_list"] for i_mc in range(len(results[i_config]))])
        runtime_mat[i_config,:,:] = np.array([results[i_config][i_mc]["runtime_list"] for i_mc in range(len(results[i_config]))])
    iter_runtime_mat = runtime_mat / num_iters_mat  # Shape: (num_configs, num_mc, num_algos)

    # Boxplot settings
    # colors = ['skyblue', 'salmon', 'lightgreen']
    import matplotlib.colors as mcolors

    def lighten_color(color, amount=0.5):
        """
        Lightens the given color by blending it with white.
        amount=0 returns original color, amount=1 returns white.
        """
        try:
            c = mcolors.to_rgb(color)
        except ValueError:
            # if not recognized, default to black
            c = (0,0,0)
        white = np.array([1,1,1])
        return tuple(c + (white - c) * amount)
    original_colors = [algo_list[d]["color"] for d in algo_list]
    colors = [lighten_color(c, 0.5) for c in original_colors] 
    fig_runtime_boxplot, ax = plt.subplots(figsize=(6,4))
    
 
    
    x = np.arange(num_configs)  # x locations for each config
    width = 0.10  # width of each box
    gap = 0.03  # gap between boxes inside a group

    group_width = num_algos * width + (num_algos - 1) * gap
    offsets = (
        np.arange(num_algos) * (width + gap)
        - group_width / 2
        + width / 2
    )
    for algo_idx in range(num_algos):
        pos = x + offsets[algo_idx]
        data_to_plot = [runtime_mat[config_idx,:,algo_idx] for config_idx in range(num_configs)]
        
        if DO_BOXPLOT:
            bp = ax.boxplot(data_to_plot, positions=pos, widths=width, patch_artist=True, showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor(colors[algo_idx])
            for whisker in bp['whiskers']:
                whisker.set_color('black')
            for cap in bp['caps']:
                cap.set_color('black')
            for median in bp['medians']:
                median.set_color('black')
        
        else:
            vp = ax.violinplot(data_to_plot, positions=pos, widths=width, showmeans=False, showmedians=True, showextrema=False)
            #set median color black
            vp['cmedians'].set_color('black')
            vp['cmedians'].set_linewidth(2)
            # Fill color
            for body in vp['bodies']:
                body.set_facecolor(colors[algo_idx])
                body.set_alpha(0.5)  # make it transparent

    # Labels and ticks
    xylabel_fontsize = 12
    ax.set_xticks(x)
    ax.set_xticklabels([f'${param_name} = {param_vals[i]}$' for i in range(num_configs)],fontsize=xylabel_fontsize)

    ax.set_ylabel('Runtime (s)',fontsize=xylabel_fontsize)
    # ax.set_title('Algorithm runtimes by configuration')
    if logscale_y:
        ax.set_yscale('log')
    # Legend
    from matplotlib.patches import Patch
    algo_names = list(algo_list.keys())
    labels = [f"{ALGONAME}({algo_name})" if (algo_name == "AIRM" or algo_name == "JBLD" or algo_name == "LE") else algo_name 
                for algo_name in algo_names]
    legend_handles = [Patch(facecolor=colors[i], label=labels[i]) for i in range(num_algos)]
    lgd = ax.legend(handles=legend_handles)
    for text in lgd.get_texts():
        if "JBLD" in text.get_text():
            text.set_fontweight("bold")
    
    delta_x = 0.4
    ax.set_xlim([0-delta_x,1+delta_x])
    # set grid only for y axis
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, which='both')
    plt.tight_layout()
    # plt.show()



    
    def print_table_with_percentile(table_name, mat, do_only_mean = False):
        from tabulate import tabulate
        qlow_mat = np.percentile(mat, 25, axis=1).T
        qmid_mat = np.percentile(mat, 50, axis=1).T
        qhigh_mat = np.percentile(mat, 75, axis=1).T

        mean_mat = np.mean(mat, axis=1).T


        header = ["Algorithm"] + [f'${param_name} = {param_vals[i]}$' for i in range(num_configs)]
        table_data = []

        for alg_idx in range(num_algos):
            row = [algo_names[alg_idx]]
            for cfg_idx in range(num_configs):
                if do_only_mean:
                    formatted_value = f"{mean_mat[alg_idx, cfg_idx]:.4f}"
                else:
                    formatted_value = (
                        f"{qmid_mat[alg_idx, cfg_idx]:.4f} "
                        f"[ {qlow_mat[alg_idx, cfg_idx]:.4f}, "
                        f"{qhigh_mat[alg_idx, cfg_idx]:.4f} ]"
                    )
                row.append(formatted_value)
            table_data.append(row)

        print(f"================   {table_name}   ================")
        tabulate_data = tabulate(table_data, headers=header, tablefmt="grid")
        print(tabulate_data)
        print("==========================================================")
        return (table_name, tabulate_data)
        # with open(os.path.join(path_results_dir, table_name+'.txt'), "w") as f:
        #     f.write(tabulate_data)
        # return tabulate_data
    
    # print_table("Runtime", runtime_mat.mean(axis=1).T, runtime_mat.std(axis=1).T)
    # print_table("Num Iters", num_iters_mat.mean(axis=1).T, num_iters_mat.std(axis=1).T)
    # print_table("Iteration Runtime", iter_runtime_mat.mean(axis=1).T, iter_runtime_mat.std(axis=1).T)
    # print_table_with_percentile("Runtime", runtime_mat)
    table_num_iters = print_table_with_percentile("Num Iters", num_iters_mat, do_only_mean=False)
    table_iter_runtime_ms = print_table_with_percentile("Iteration Runtime [ms]", iter_runtime_mat * 1e3)  # Convert to milliseconds
    table_runtime_s = print_table_with_percentile("Runtime [s]", runtime_mat)



    return fig_runtime_boxplot, [table_num_iters, table_iter_runtime_ms, table_runtime_s]


def create_runtime_plot_M(list_resultPaths, list_M):
    #%%
    import matplotlib.pyplot as plt
    import pickle
    num_results = len(list_resultPaths)
    results = []
    for i_result in range(num_results):
        path_results_dir = list_resultPaths[i_result]
        name_results_dir = os.path.basename(path_results_dir)
        with open(path_results_dir + '/results.pkl', 'rb') as f:
            curr_results = pickle.load(f)
        results.append(curr_results[0])

    fig_runtime_boxplot, list_of_tables = plot_iteration_and_runtime_boxplot(results, list_M, 'M', DO_BOXPLOT=True, logscale_y=True)
    #%%
    save_figure(fig_runtime_boxplot, path_results_dir, "boxplot_runtime_vs_M")

    for table_name, table_data in list_of_tables:
        with open(os.path.join(path_results_dir, table_name+'.txt'), "w") as f:
            f.write(table_data)
    
    
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
