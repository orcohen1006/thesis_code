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
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_all_exp():
    np.random.seed(42)
    t0_overall = time.time()

    for cohr_flag in [False]:
        exp_N(cohr_flag)
        
    for cohr_flag in [False]:
        for N in [30]:
            exp_DeltaSNR(N, cohr_flag)

    for cohr_flag in [False]:
        for N in [20, 40]:
            exp_DeltaTheta(N, cohr_flag)

    for cohr_flag in [False]:
        exp_SNR(cohr_flag)
    
    for cohr_flag in [False]:
        exp_M(cohr_flag)
    print(f'Total Running Time: {time.time() - t0_overall} sec.')
    plt.show()

if __name__ == "__main__":
    run_all_exp()
    git_commit_and_push()
