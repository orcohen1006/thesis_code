import numpy as np
import time
import os
import matplotlib.pyplot as plt
from Exp_N import *
from Exp_M import *
from Exp_Rho import *
from Exp_SNR import *
from Exp_SNR_HUCA import *
from Exp_M import *
from Exp_OffGrid import *
import ToolsMC 
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
    doa=np.array([35.0, 43.0, 51.0])
    power_doa_db=np.array([0, 0, -5])
    # %%
    # path_resultsDir_M12 = exp_Msingle(basedir=basedir, power_doa_db=power_doa_db, doa=doa, basic=True, basic_M=12, current_M=12)
    # path_resultsDir_M120 = exp_Msingle(basedir=basedir, power_doa_db=power_doa_db, doa=doa, basic=False, basic_M=12, current_M=120)
    # ToolsMC.create_runtime_plot_M([path_resultsDir_M12, path_resultsDir_M120], [12, 120])
    # %%

    exp_SNR(basedir=basedir, doa=doa, power_doa_db=power_doa_db, N=50, M=12)

    exp_N(basedir=basedir, doa=doa, power_doa_db=power_doa_db, snr=-1.5)
        
    doa_for_exp_rho = np.array([35.0, 41.0])
    exp_rho(basedir=basedir, doa=doa_for_exp_rho, power_doa_db=np.array([0, 0]), N=50)


    exp_SNR_HUCA(basedir=basedir, doa=doa, power_doa_db=power_doa_db, N=50, M=14)


    doa_for_offgrid=np.array([35.0, 51.0])
    exp_OffGrid(basedir=basedir, doa=doa_for_offgrid, power_doa_db=np.array([0, 0]), N=50, M=12, snr=0)
    exp_OffGrid(basedir=basedir, doa=doa_for_offgrid, power_doa_db=np.array([0, 0]), N=50, M=12, snr=-3)
    
    # ---------------------------------------
    print(f'Total Running Time: {time.time() - t0_overall} sec.')
    return basedir




if __name__ == "__main__":
    basedir = run_all_exp()
    git_commit_and_push(commit_message=basedir)
