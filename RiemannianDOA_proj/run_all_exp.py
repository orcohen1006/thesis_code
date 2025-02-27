import numpy as np
import time
import os
import matplotlib.pyplot as plt
from Exp_DeltaTheta import *

def run_all_exp():
    np.random.seed(42)
    Large_Scale_Flag = False
    t0_overall = time.time()

    # for cohr_flag in [False, True]:
    #     for N in [16, 120]:
    #         exp_SNR(N, cohr_flag, Large_Scale_Flag)

    for cohr_flag in [False, True]:
        for N in [16, 120]:
            exp_DeltaTheta(N, cohr_flag, Large_Scale_Flag)

    # for cohr_flag in [False, True]:
    #     exp_N(cohr_flag, Large_Scale_Flag)

    print(f'Total Running Time: {time.time() - t0_overall} sec.')
    plt.show()

if __name__ == "__main__":
    run_all_exp()
