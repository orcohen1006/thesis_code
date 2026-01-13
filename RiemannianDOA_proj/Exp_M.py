# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import List, Optional
from utils import *
from ToolsMC import *

# %%

def exp_Msingle(cohr_flag: bool = False, power_doa_db: np.ndarray = np.array([0, 0, -5]), doa: np.ndarray = np.array([35.25, 43.25, 51.25]),
            basic = True, basic_M = 12, current_M = 12,
            basedir:str = ''):
    utils.globalParams = GlobalParms()  # reset global params to default values
    timestamp = datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    str_indp_cohr = 'cohr' if cohr_flag else 'indp'
    name_results_dir = f'Exp_Msingle_{timestamp}_{str_indp_cohr}_basicM_{basic_M}_currentM_{current_M}'
    name_results_dir = os.path.join(basedir, name_results_dir)
    path_results_dir = os.path.abspath(name_results_dir)
    print(f"Results will be saved in: {path_results_dir}")
    if not os.path.exists(path_results_dir):
        os.makedirs(path_results_dir)
    # %%
    num_mc = NUM_MC
    utils.globalParams.WANTED_ALGO_NAMES.discard("ESPRIT") # don't care for ESPRIT runtime



    resolution_factor = basic_M / current_M
    if (basic):
        assert current_M == basic_M, "In basic mode, current_M must equal basic_M"
    else:
        original_diff_doa = doa - doa[0]
        grid_minval_degrees = 40
        num_grid_points = get_doa_grid().shape[0]
        utils.globalParams.GRID_STEP_DEGREES = utils.globalParams.GRID_STEP_DEGREES * resolution_factor
        grid_maxval_degrees = grid_minval_degrees + utils.globalParams.GRID_STEP_DEGREES * (num_grid_points -1)
        utils.globalParams.GRID_MIN_MAX_VALS_DEGREES = (grid_minval_degrees, grid_maxval_degrees)
        doa_min = 45
        doa = doa_min + original_diff_doa*resolution_factor



    
    config = create_config(
        m=current_M, snr=0+convert_linear_to_db(resolution_factor), N=int(50 / resolution_factor), power_doa_db=power_doa_db, doa=doa
    )



    num_configs = 1
    config_list = []
    config_list.append(config)





    # %% Run the configurations
    results = RunDoaConfigsPBS(path_results_dir, config_list, num_mc)
    # %%
    results, algos_error_data = analyze_algo_errors(results)
    #
     
    
    
    # %%
    str_desc_name = os.path.basename(name_results_dir)

  
    plt.close()
    # %%
    experiment_configs_string_to_file(num_mc=num_mc, config_list=config_list, directory=path_results_dir)

    utils.globalParams = GlobalParms()  # reset global params to default values
    return path_results_dir

if __name__ == "__main__":
    # Example usage
    1/0


# %%
