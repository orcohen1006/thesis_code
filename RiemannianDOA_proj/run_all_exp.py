import numpy as np
import time
import os
import matplotlib.pyplot as plt
from Exp_N import *
from Exp_M import *
from Exp_Rho import *
from Exp_SNR_Large import *
from Exp_M import *
from commit_repo_git import git_commit_and_push
from datetime import datetime
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_all_exp():
    t0_overall = time.time()
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    basedir = f'run_exp_{timestamp}'
    print(f"run exp basedir: {basedir}")
    if not os.path.exists(basedir):
        os.makedirs(basedir) 
    # ---------------------------------------
    # exp_M(basedir=basedir, power_doa_db=np.array([0]), doa=np.array([35.25]))


    doa=np.array([35.25, 43.25, 51.25])
    # doa=np.array([35.25, 45.75, 55.75])
    power_doa_db=np.array([0, 0, -5])

    exp_N(basedir=basedir, doa=doa, power_doa_db=power_doa_db, snr=0)

    exp_SNR_Large(basedir=basedir, doa=doa, power_doa_db=power_doa_db, N=40)

    exp_rho(basedir=basedir, doa=np.array([35.25, 41.25]), power_doa_db=np.array([0, 0]), N=40)
    # ---------------------------------------
    print(f'Total Running Time: {time.time() - t0_overall} sec.')
    return basedir
if __name__ == "__main__":
    basedir = run_all_exp()
    git_commit_and_push(commit_message=basedir)
