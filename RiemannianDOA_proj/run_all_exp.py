import numpy as np
import time
import os
import matplotlib.pyplot as plt
from Exp_DeltaTheta import *
from Exp_DeltaSNR import *
from Exp_N import *
from commit_repo_git import git_commit_and_push


def run_all_exp():
    np.random.seed(42)
    Large_Scale_Flag = False
    t0_overall = time.time()

    for cohr_flag in [False]:
        for N in [30]:
            exp_DeltaSNR(N, cohr_flag, Large_Scale_Flag)

    for cohr_flag in [False]:
        for N in [20, 40]:
            exp_DeltaTheta(N, cohr_flag, Large_Scale_Flag)

    for cohr_flag in [False]:
        exp_N(cohr_flag, Large_Scale_Flag)

    print(f'Total Running Time: {time.time() - t0_overall} sec.')
    plt.show()

if __name__ == "__main__":
    run_all_exp()
    git_commit_and_push()
