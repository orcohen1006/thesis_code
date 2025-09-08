import os
from tqdm import tqdm
os.chdir('/home/or.cohen/thesis_code/RiemannianDOA_proj')
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib ipympl

from utils import *

f_AMV = lambda eigvals: np.sum((eigvals - 1)**2)
f_AIRM = lambda eigvals: np.sum(np.log(eigvals)**2)
f_JBLD = lambda eigvals: np.sum((np.log(0.5 * (eigvals + 1)) - 0.5 * np.log(eigvals)))

def normalize_vals(vals):
    q_high = np.percentile(vals, 5)
    q_low = np.percentile(vals, 0)
    # q_high = np.max(vals)
    # q_low = 0#np.min(vals)
    return (vals - q_low) / (q_high - q_low)

def create_param_vals(dtype, val_low, val_high, val_res, num_trials_overall):
    
    vals = np.arange(val_low, val_high, val_res)
    num_repeat = num_trials_overall // vals.size
    vals = np.repeat(vals, num_repeat, axis=0)
    # If there are any remaining trials, add them to the end
    remaining_trials = num_trials_overall - vals.size
    if remaining_trials > 0:
        vals = np.concatenate([vals, vals[:remaining_trials]])
    vals = vals.astype(dtype)
    return vals

config = create_config(
                m=12, snr=0, N=300, 
                power_doa_db=np.array([0,0]),
                doa=np.array([30 ,40]),
                cohr_flag=True,
                cohr_coeff=0.0,
                impulse_prob=0.0,
                impulse_factor=1.0
                )
power_doa = 10.0 ** (config["power_doa_db"] / 10.0)
A_true = get_steering_matrix(config["doa"], config["m"])
noise_power_db = np.max(config["power_doa_db"]) - config["snr"]
noise_power = 10.0 ** (noise_power_db / 10.0)
R = A_true @ np.diag(power_doa) @ A_true.conj().T + noise_power * np.eye(config["m"])

NUM_TRIALS = 1_000
np.random.seed(0)
list_tuples = [
    ("N", np.random.randint(config["m"], config["m"]*20, size=(NUM_TRIALS,1))),
    # ("N", create_param_vals(np.int, config["m"]*1, config["m"]*30,config["m"]/3, NUM_TRIALS)),
    # ("cohr_coeff", create_param_vals(np.float, 0, 1, 0.025, NUM_TRIALS)),
               ]
for i_parameter in range(len(list_tuples)):
    base_array = None
    if len(list_tuples[i_parameter]) == 2:
        parameter_name, parameter_vals = list_tuples[i_parameter]
    else:
        parameter_name, parameter_vals, base_array = list_tuples[i_parameter]
    vals_AMV = np.zeros(shape=(NUM_TRIALS,1))
    vals_AIRM = np.zeros(shape=(NUM_TRIALS,1))
    vals_JBLD = np.zeros(shape=(NUM_TRIALS,1))
    mat_eigvals = np.zeros(shape=(config["m"], NUM_TRIALS))
    # use tqdm for progress bar
    for i_trial in tqdm(range(NUM_TRIALS)):
        curr_config = config.copy()
        param_val = parameter_vals[i_trial].item()
        if base_array is not None:
            param_val += base_array
        curr_config[parameter_name] = param_val
        A_ = get_steering_matrix(curr_config["doa"], curr_config["m"])
        y_noisy = generate_signal(A_, curr_config["power_doa_db"], curr_config["N"], noise_power, cohr_flag=curr_config["cohr_flag"],
                                cohr_coeff=curr_config["cohr_coeff"], noncircular_coeff=curr_config["noncircular_coeff"],
                                impulse_prob=curr_config["impulse_prob"], impulse_factor=curr_config["impulse_factor"],
                                seed=i_trial)
        R_hat = y_noisy @ y_noisy.conj().T / curr_config["N"]
        eigvals = eigvals_of_Q(R_hat, R)
        vals_AMV[i_trial] = f_AMV(eigvals)
        vals_AIRM[i_trial] = f_AIRM(eigvals)
        vals_JBLD[i_trial] = f_JBLD(eigvals)

        mat_eigvals[:, i_trial] = eigvals


    alpha_transp = 0.3
    CONST_MUL_JBLD = 8
    fig1, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(parameter_vals, vals_AMV/vals_AIRM, alpha=alpha_transp, label=r'$\mathcal{D}^2_{\text{AMV}} / \mathcal{D}^2_{\text{AIRM}}$',color='red')
    ax.scatter(parameter_vals, CONST_MUL_JBLD*(vals_JBLD)/vals_AIRM, alpha=alpha_transp, label=r'$\mathcal{D}^2_{\text{JBLD}} / \mathcal{D}^2_{\text{AIRM}}$',color='blue')
    ax.set_xlabel(parameter_name)
    ax.set_ylabel(r'$\mathcal{D}^2 \,/ \,\mathcal{D}^2_{\text{AIRM}}$')
    ax.legend()
    plt.show()

    fig2, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=1, vmax=config["m"])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    for i_eig in range(config["m"]):
        color = sm.to_rgba(i_eig + 1)
        ax.scatter(parameter_vals, mat_eigvals[i_eig, :], color=color, alpha=alpha_transp)
    cbar = fig2.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.2)
    cbar.set_label('Eigenvalue Index', rotation=0, labelpad=5)
    ax.set_xlabel(parameter_name)
    ax.set_ylabel('Eigenvalues')
    plt.show()

    save_figure(fig1, "./", f"distances_relation_{parameter_name}")
    save_figure(fig2, "./", f"distances_relation_eigenvalues_{parameter_name}")