import numpy as np
import time
import os
import matplotlib.pyplot as plt
from Exp_DeltaTheta import *
from Exp_DeltaSNR import *
from Exp_N import *
from Exp_SNR import *
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
    for cohr_flag in [False]:
        exp_N(cohr_flag=cohr_flag, basedir=basedir)
        
    for cohr_flag in [False]:
        for N in [30]:
            exp_DeltaSNR(n=N, cohr_flag=cohr_flag, basedir=basedir)

    for cohr_flag in [False]:
        for N in [40, 50]: #[20, 40]:
            for theta0 in [40]:
                exp_DeltaTheta(n=N, cohr_flag=cohr_flag, theta0=theta0, basedir=basedir)

    for cohr_flag in [False]:
        exp_SNR(cohr_flag=cohr_flag, basedir=basedir)
    
    # for cohr_flag in [False]:
    #     exp_M(cohr_flag=cohr_flag, basedir=basedir)
    # ---------------------------------------
    print(f'Total Running Time: {time.time() - t0_overall} sec.')
    return basedir
if __name__ == "__main__":
    basedir = run_all_exp()
    git_commit_and_push(commit_message=basedir)
