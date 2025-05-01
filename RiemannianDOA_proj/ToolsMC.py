import os
import pickle
import subprocess
import time
from pathlib import Path
from utils import *


def save_job_metadata(workdir: str, num_mc: int):
    if os.path.exists(FILENAME_PBS_METADATA):
        os.remove(FILENAME_PBS_METADATA)
    metadata = {}
    metadata["num_mc"] = num_mc
    metadata["workdir"] = workdir
    with open(FILENAME_PBS_METADATA, "wb") as f:
        pickle.dump(metadata, f)

def save_doa_configs(workdir: str, config_list: list):
    for i in range(len(config_list)):
        cfg = config_list[i]
        with open(f"{workdir}/config_{i}.pkl", "wb") as f:
            pickle.dump(cfg, f)

def submit_job_array(workdir: str, config_list: list, num_mc: int):
    total_jobs = len(config_list) * num_mc
    res = subprocess.run(["qsub", "-J", f"0-{total_jobs-1}", FILENAME_PBS_SCRIPT], capture_output=True, text=True)

    if res.returncode != 0:
        print("Error submitting job array:", res.stderr)
        raise RuntimeError("Job submission failed.")

    print("Job submission output:", res.stdout)
    return res.stdout.strip()

def wait_for_results(workdir: str, config_list: list, num_mc:int, t0: float):
    expected = len(config_list) * num_mc
    while True:
        done = len(list(Path(workdir).glob("config_*_mc*.pkl")))
        print(f"Waiting... {done}/{expected} results ready. elapsed time: {time.time() - t0:.2f} [sec]")
        if done >= expected:
            break
        time.sleep(10) 

def collect_results(workdir: str, config_list: list, num_mc:int):
    results = []
    for i in range(len(config_list)):
        for j in range(num_mc):
            result_file = Path(workdir) / f"config_{i}_mc{j}.pkl"
            with open(result_file, "rb") as f:
                results.append(pickle.load(f))
            os.remove(result_file)
    return results

def RunDoaConfigsPBS(workdir: str, config_list: list, num_mc:int):
    t0 = time.time()
    print("Starting job array with", len(config_list), "configurations and", num_mc, "Monte Carlo iterations.")
    
    save_job_metadata(workdir, num_mc)

    save_doa_configs(workdir, config_list)
    
    job_id = submit_job_array(workdir, config_list, num_mc)
    print("Submitted job ID:", job_id)

    wait_for_results(workdir, config_list, num_mc, t0)
    print("All results ready. Collecting...")

    results = collect_results(workdir, config_list, num_mc)
    print(f"Collected {len(results)} results.")
    with open(f"{workdir}/results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {workdir}/results.pkl")
    print(f"RunDoaConfigsPBS: Total elapsed time: {time.time() - t0:.2f} [sec]")
    return results

if __name__ == "__main__":
    num_mc = 100
    num_configs = 5
    workdir = 'TmpWorkDir'
    os.makedirs(workdir, exist_ok=True)

    config_list = []
    for i in range(num_configs):
        config_list.append(
            create_config(
                m=12, snr=0, N=20, power_doa_db=np.array([3, 4]), doa=np.array([35, 40+i]), cohr_flag=False,
                )
        )
    # Run the configurations
    RunDoaConfigsPBS(workdir, config_list, num_mc)
    # %%
    with open('TmpWorkDir/results.pkl', 'rb') as f:
        results = pickle.load(f)
    print("Results loaded from file.")
# %%
