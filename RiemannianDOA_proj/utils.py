import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
import torch
import matplotlib.pyplot as plt
import pickle
import os

FILENAME_PBS_SCRIPT = "job_byOrCohen.pbs"
FILENAME_PBS_METADATA = "job_metadata.pkl"


EPS_REL_CHANGE = 1e-4

ALGONAME = "SERCOM"

def save_figure(fig: plt.Figure, path_results_dir: str, name: str):
    fig.savefig(os.path.join(path_results_dir, name +  '.png'), dpi=300)
    fig.savefig(os.path.join(path_results_dir, name +  '.pdf'), format="pdf", bbox_inches="tight")

    with open(os.path.join(path_results_dir, name +  '.pkl'), 'wb') as f:
        pickle.dump(fig, f)


def matrix_pinv_sqrtm(B_in):
    """Compute the inverse square root of a positive definite matrix B."""
    eigvals, eigvecs = np.linalg.eigh(B_in)
    eigvals_new = 1.0 / np.sqrt(np.clip(eigvals.real, a_min=1e-10, a_max=None))  # Avoid division by zero
    eigvals_new[eigvals.real < 1e-10] = 0
    Lam_new = np.diag(eigvals_new)
    B_out = eigvecs @ Lam_new @ eigvecs.conj().T
    return B_out
        
def eigvals_of_Q(R, R_hat):
    """
    Compute the eigenvalues of the matrix Q = R_hat^(-1/2) @ R @ R_hat^(-1/2).
    """
    pinv_sqrtm_R_hat = matrix_pinv_sqrtm(R_hat)
    Q = pinv_sqrtm_R_hat @ (R) @ pinv_sqrtm_R_hat
    eigvals = np.linalg.eigvalsh(Q).real
    return eigvals
def eigvals_of_Q_given_result(result):
    config = result['config']
    power_doa = convert_db_to_linear(config["power_doa_db"])
    A_true = get_steering_matrix(config["doa"], config["m"])
    noise_power = convert_db_to_linear(np.max(config["power_doa_db"]) - config["snr"])
    R = A_true @ np.diag(power_doa) @ A_true.conj().T + noise_power * np.eye(config["m"])
    return eigvals_of_Q(R, result['R_hat'])

def convert_db_to_linear(power_doa_db):
    """
    Convert power in dB to linear scale.

    :param power_doa_db: Power in dB
    :return: Power in linear scale
    """
    return 10.0 ** (power_doa_db / 10.0)
def convert_linear_to_db(power_doa):
    """
    Convert power in linear scale to dB.

    :param power_doa: Power in linear scale
    :return: Power in dB
    """
    return 10.0 * np.log10(power_doa)

def thresholded_l0_norm(p_vec, threshold=None):
    if threshold is None:
        threshold = 0.01 * np.max(p_vec)
    return np.sum(p_vec > threshold)

def compute_list_HPBW(p_vec, grid_doa, peak_indices):
    hpbw_values = []
    for peak_index in peak_indices:
        half_power_value = p_vec[peak_index] / 2
        # Find the left and right indices of the half-power points
        left_candidates = np.where((p_vec[:peak_index] <= half_power_value))[0]
        left_index = left_candidates[-1] if left_candidates.size > 0 else None
        # Find right side
        right_candidates = np.where((p_vec[peak_index+1:] <= half_power_value))[0]
        right_index = (peak_index + 1 + right_candidates[0]) if right_candidates.size > 0 else None
        if left_index is None or right_index is None:
            continue
        # Linear interpolation on the left
        x0, y0 = grid_doa[left_index], p_vec[left_index]
        x1, y1 = grid_doa[left_index + 1], p_vec[left_index + 1]
        left_theta = x0 + (half_power_value - y0) / (y1 - y0) if y1 != y0 else x0

        # Linear interpolation on the right
        x0, y0 = grid_doa[right_index - 1], p_vec[right_index - 1]
        x1, y1 = grid_doa[right_index], p_vec[right_index]
        right_theta = x0 + (half_power_value - y0) / (y1 - y0) if y1 != y0 else x1

        hpbw_values.append(right_theta - left_theta)
        
    return hpbw_values

def OLD_parabolic_peak_interpolation(p_vec, grid_doa, peak_index):
    """
    Perform parabolic interpolation to find the peak value.

    :param p_vec: Power vector
    :param grid_doa: DOA grid
    :param peak_index: Index of the peak
    :return: Interpolated peak DOA and power
    """
    if peak_index == 0 or peak_index == len(p_vec) - 1:
        # If the peak is at the boundary, we cannot interpolate
        return grid_doa[peak_index], p_vec[peak_index]
    
    x0, y0 = grid_doa[peak_index - 1], p_vec[peak_index - 1]
    x1, y1 = grid_doa[peak_index], p_vec[peak_index]
    x2, y2 = grid_doa[peak_index + 1], p_vec[peak_index + 1]
    
    denom = (x2 - x0) * (x1 - x0) * (x2 - x1)
    if denom < 1e-10:
        return grid_doa[peak_index], p_vec[peak_index]
    
    a = (y2 - y0) / denom
    b = (y1 - y0) / (x1 - x0) - a * (x1 + x0)
    c = y0
    if a < 1e-10:
        return grid_doa[peak_index], p_vec[peak_index]
    peak_x = -b / (2 * a)
    peak_y = a * peak_x**2 + b * peak_x + c
    
    return peak_x, peak_y


def parabolic_peak_interpolation(p_vec, grid_doa, peak_index):
    """
    Perform parabolic interpolation to refine the peak position.

    :param p_vec: Power vector (1D numpy array)
    :param grid_doa: DOA grid (same shape as p_vec)
    :param peak_index: Index of the peak (integer)
    :return: interpolated_doa (float), interpolated_power (float)
    """
    # Ensure inputs are valid
    if not (0 < peak_index < len(p_vec) - 1):
        # Cannot interpolate at the edge; return grid value
        return grid_doa[peak_index], p_vec[peak_index]

    # Neighboring values
    p1, p2, p3 = p_vec[peak_index - 1], p_vec[peak_index], p_vec[peak_index + 1]
    x1, x2, x3 = grid_doa[peak_index - 1], grid_doa[peak_index], grid_doa[peak_index + 1]

    # Fit a parabola: y = a*x^2 + b*x + c
    # Use vertex formula: x_vertex = x2 - 0.5 * (p3 - p1) / (p3 - 2*p2 + p1)
    denom = p3 - 2 * p2 + p1
    if denom == 0:
        # Prevent division by zero: return grid peak
        return x2, p2

    delta = 0.5 * (p1 - p3) / denom  # offset from x2 (grid_doa[peak_index])
    # Clamp delta to avoid going out of bounds (optional)
    delta = np.clip(delta, -1.0, 1.0)

    interpolated_doa = x2 + delta * (x3 - x2)  # assumes uniform grid
    # Estimate interpolated power (optional)
    interpolated_power = p2 - 0.25 * (p1 - p3) * delta

    return interpolated_doa, interpolated_power

def naive_peak_interpolation(p_vec, grid_doa, peak_index):
    if peak_index == 0 or peak_index == len(p_vec) - 1:
        # If the peak is at the boundary, we cannot interpolate
        return grid_doa[peak_index], p_vec[peak_index]
    x0, y0 = grid_doa[peak_index - 1], p_vec[peak_index - 1]
    x1, y1 = grid_doa[peak_index], p_vec[peak_index]
    x2, y2 = grid_doa[peak_index + 1], p_vec[peak_index + 1]
    
    w0 = y0 / (y0 + y1 + y2)
    w1 = y1 / (y0 + y1 + y2)
    w2 = y2 / (y0 + y1 + y2)

    peak_x = w0 * x0 + w1 * x1 + w2 * x2
    peak_y = y1

    return peak_x, peak_y

def estimate_doa_calc_errors(p_vec, grid_doa, true_doas, true_powers,
                                threshold_theta_detect = 2,
                                allowed_peak_height_relative_to_max=0.01):
    
    dummy_estimated_doa = 0.0
    dummy_estimated_power = convert_db_to_linear(-10)

    if isinstance(p_vec, torch.Tensor):
        p_vec = p_vec.numpy()
    num_sources = len(true_doas)
    # Find peaks in descending order
    threshold_peak_height = max(allowed_peak_height_relative_to_max * np.max(p_vec), 0)
    peak_indices, _ = find_peaks(p_vec, height=threshold_peak_height)
    num_detected_doas = len(peak_indices)
    peak_indices = peak_indices[np.argsort(p_vec[peak_indices])[::-1]] # Sort the inidices by peaks values in descending order
    
    all_detected_doas = grid_doa[peak_indices]
    all_detected_powers = p_vec[peak_indices]
        
    # all_detected_doas = []
    # all_detected_powers = []
    # for i_detected_doa in range(num_detected_doas):
    #     detected_doa, detected_power = parabolic_peak_interpolation(p_vec, grid_doa, peak_indices[i_detected_doa])
    #     all_detected_doas.append(detected_doa)
    #     all_detected_powers.append(detected_power)
    # all_detected_doas = np.array(all_detected_doas)
    # all_detected_powers = np.array(all_detected_powers)
    
    

    if num_detected_doas >= num_sources:
        selected_detected_doas = all_detected_doas[:num_sources]
        selected_detected_powers = all_detected_powers[:num_sources]
    else:
        pad_size = num_sources - num_detected_doas
        selected_detected_doas = np.concatenate([
            all_detected_doas,
            np.full(pad_size, dummy_estimated_doa)
        ])
        selected_detected_powers = np.concatenate([
            all_detected_powers,
            np.full(pad_size, dummy_estimated_power)
        ])


    cost_matrix = np.abs(true_doas[:, np.newaxis] - selected_detected_doas[np.newaxis, :])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    selected_doa_error = true_doas[row_ind] - selected_detected_doas[col_ind]
    selected_power_error = true_powers[row_ind] - selected_detected_powers[col_ind]
        

    succ_match_detected_doa = np.zeros((num_detected_doas,), dtype=bool)
    succ_match_true_doa = np.zeros((num_sources,), dtype=bool)
    for true_idx, detected_idx in zip(row_ind, col_ind):
        if detected_idx < num_detected_doas:  # Only valid detected indices (exclude dummy)
            err = abs(true_doas[true_idx] - selected_detected_doas[detected_idx])
            if err < threshold_theta_detect:
                succ_match_detected_doa[detected_idx] = True
                succ_match_true_doa[true_idx] = True
    

    mean_HPBW = compute_list_HPBW(p_vec, grid_doa, peak_indices)    

    return num_detected_doas, all_detected_doas, all_detected_powers, selected_doa_error, selected_power_error, \
            succ_match_detected_doa, succ_match_true_doa, mean_HPBW


def display_power_spectrum(config, list_p_vec, epsilon_power=None, algo_list=None, ax=None):
    """
    Display the power spectrum of the DOA estimation.

    :param config: Configuration dictionary
    :param list_p_vec: List of power vectors for different algorithms
    """
    import matplotlib.pyplot as plt

    power_doa_db = config["power_doa_db"]

    doa = config["doa"]


    doa_scan = get_doa_grid()

    if algo_list is None:
        algo_list = get_algo_dict_list()

    if epsilon_power is None:
        epsilon_power = 10.0 ** (-20 / 10.0)
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()

    list_plt = []
    for i_algo, algo_name in enumerate(algo_list.keys()):
        label = f"{ALGONAME}({algo_name})" if (algo_name == "AIRM" or algo_name == "JBLD") else algo_name
        est = list_p_vec[i_algo]
        # check if est is a tuple (for ESPRIT)
        if isinstance(est, tuple):
            doa_est_degrees = est
            doa_to_display = np.array(doa_est_degrees)
            powers_to_display = 0*doa_to_display
            pltobj, = ax.plot(doa_to_display, powers_to_display, label=label, **algo_list[algo_name])
        else:
            spectrum = est
            spectrum[spectrum < epsilon_power] = epsilon_power
            spectrum = convert_linear_to_db(spectrum)
            pltobj, = ax.plot(doa_scan, spectrum, label=label, **algo_list[algo_name])
        list_plt.append(pltobj)
    
    plt_doa, = ax.plot(doa, power_doa_db, 'x', color='black', label='DOA')
    # list_plt.append(plt_doa)
    
    lgd = ax.legend(handles=list_plt)
    for text in lgd.get_texts():
        if "JBLD" in text.get_text():
            text.set_fontweight("bold")
    ax.set_xlabel(r"$\theta$ (degrees)", fontsize=12)
    ax.set_ylabel(r"$\mathrm{Power}$ (dB)", fontsize=12)
    
    # plt.title('Directions Power Spectrum Estimation')
    return ax

# def detect_DOAs(p_vec, grid_doa, doa):
#     num_sources = len(doa)
#     # Find peaks in descending order
#     peak_indices, _ = find_peaks(p_vec)
#     peak_values = p_vec[peak_indices]
#     sorted_indices = np.argsort(-peak_values)  # Sort in descending order
#     peak_indices = peak_indices[sorted_indices]
#     peak_indices = np.atleast_1d(peak_indices)
#     if (not isinstance(peak_indices, np.ndarray)) or len(peak_indices) < num_sources:
#         # Not all peaks detected
#         print("Not all peaks detected")
#         detection_status = 0
#         doa_error = np.nan
#         detected_powers = np.nan
#         return detected_powers, doa_error, detection_status

#     # Check whether the detection is correct
#     detected_doas = grid_doa[peak_indices[:num_sources]]

#     # Sort detected DOAs in ascending order
#     sorted_indices = np.argsort(detected_doas)
#     detected_doas = detected_doas[sorted_indices]
#     doa_error = detected_doas - doa

#     detection_status = 1  # detection successful
#     # The powers from large value to small value
#     detected_powers = p_vec[peak_indices[:num_sources]]
#     # Sort the power according to the DOA
#     detected_powers = detected_powers[sorted_indices]

#     return detected_powers, doa_error, detection_status
    
def model_order_selection(R, N):

    eigs = np.linalg.eigvalsh(R)[::-1]  # Sort eigenvalues in descending order
    M = len(eigs)
    aic = np.zeros(M)
    mdl = np.zeros(M)
    for k in range(M):
        num = M - k
        geo = np.product(eigs[k:])**(1/num)
        arith = np.mean(eigs[k:])
        plunge = num * np.log(arith / geo)
        aic[k] = 2 * N * plunge + 2 * k * (2*M - k)
        mdl[k] = N * plunge + 0.5 * k * (2*M - k) * np.log(N)
    return np.argmin(aic), np.argmin(mdl)

def generate_signal(A_true, power_doa_db, t_samples, noise_power, cohr_flag=False, cohr_coeff = 1.0, noncircular_coeff = 0.0, 
                    impulse_prob=0.0, impulse_factor=1.0,
                    seed=None):
    if seed is not None:
        np.random.seed(seed)

    m = A_true.shape[0]
    num_sources = len(power_doa_db)
    amplitude_doa = np.sqrt(10.0 ** (power_doa_db / 10.0))

    # Generate signal
    noise = np.sqrt(noise_power / 2) * (np.random.randn(m, t_samples) + 1j * np.random.randn(m, t_samples))
    num_impulse_snapshots = int(impulse_prob * t_samples)
    noise[:, :num_impulse_snapshots] *= impulse_factor

    if not cohr_flag:  # independent sources
        waveform = np.exp(1j * 2 * np.pi * np.random.rand(num_sources, t_samples))
        # waveform = np.sqrt(1 / 2) * (np.random.randn(num_sources, t_samples) + 1j * np.random.randn(num_sources, t_samples))
        waveform = waveform * np.tile(amplitude_doa, (t_samples, 1)).T
    else:  # coherent sources
        waveform = np.exp(1j * 2 * np.pi * np.random.rand(num_sources - 1, t_samples))
        waveform_last = np.exp(1j * 2 * np.pi * np.random.rand(1, t_samples))
        waveform_last = cohr_coeff * waveform[0, :] + np.sqrt(1- cohr_coeff**2)*waveform_last
        waveform = np.vstack([waveform, waveform_last])
        waveform = waveform * np.tile(amplitude_doa, (t_samples, 1)).T
    waveform = make_non_circular(waveform, kappa=noncircular_coeff)

    y_noisefree = A_true @ waveform  # ideal noiseless measurements
    y_noisy = y_noisefree + noise  # noisy measurements

    return y_noisy

def make_non_circular(s, kappa):
    """
    Takes a circular signal s (K x N), returns non-circular version.
    """
    real = np.real(s)
    imag = np.imag(s)
    new_imag = np.sqrt(1 - kappa**2) * imag + kappa * real
    return real + 1j * new_imag

def get_doa_grid():
    res = 0.5 # resolution in degrees
    doa_scan = np.arange(0, 180+res, res)  # doa grid
    return doa_scan
def get_steering_matrix(theta_degrees, m, calcGradient_wrt_radians=False):

    doa_rad = np.deg2rad(theta_degrees) # Convert to radians
    delta_vec = np.arange(m)    
    A = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa_rad)))
    # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    # A = A / np.sqrt(m)
    
    if calcGradient_wrt_radians:
        dA_dtheta_radians = -1j * np.pi * np.outer(delta_vec, np.sin(doa_rad)) * A  # shape (m, K)
        return A, dA_dtheta_radians
    return A

def get_algo_dict_list(flag_get_all=False):
    # return [("PER",'r-->'), ("SPICE",'m--p'), ("SAMV",'b-^'), ("AIRM",'g--s'), ("JBLD",'y--o')]
    # return [("SPICE",'m--p'), ("SAMV",'r--^'), ("AIRM",'g-s'), ("JBLD",'b--o')]
    linewidth = 2
    d = {
        "SPICE": {"linestyle": "--", "color": "#BBB800FF", "marker": "s", "markersize": 4, "linewidth": linewidth},
        "SAMV":  {"linestyle": "--", "color": "#E65908", "marker": "^", "markersize": 5.5, "linewidth": linewidth},
        "AIRM":  {"linestyle": "-", "color": "#0CBD56", "marker": "o", "linewidth": linewidth},
        "JBLD":  {"linestyle": "-", "color": "#2B27FF", "marker": "o", "markerfacecolor": "none", "markersize": 8, "linewidth": linewidth},
    }
    if flag_get_all:
        d = {
                "PER": {"linestyle": ":", "color": "y", "marker": "^"},
                "LE_ss": {"linestyle": "-.", "color": "m", "marker": "s", "markersize": 4, "linewidth": linewidth},
                "MVDR": {"linestyle": "--", "color": "c", "marker": "o", "markersize": 6},
                "ESPRIT": {"linestyle": "", "color": "r", "marker": "o", "markersize": 8},
                **d}

    return d


def create_config(m, snr, N, power_doa_db, doa, cohr_flag=False, cohr_coeff=1.0, noncircular_coeff=0.0, 
                  impulse_prob=0.0, impulse_factor=1.0):
    """
    Create a configuration dictionary to hold parameters for simulations.
    """
    return {
        "m": m,
        "snr": snr,
        "N": N,
        "power_doa_db": power_doa_db,
        "doa": doa,
        "cohr_flag": cohr_flag,
        "cohr_coeff": cohr_coeff,
        "noncircular_coeff": noncircular_coeff,
        "impulse_prob": impulse_prob,
        "impulse_factor": impulse_factor,
    }

def experiment_configs_string_to_file(num_mc, config_list, directory="", filename="configurations_output.txt"):
    import os
    strs = []
    strs.append(f"Number of Monte Carlo runs: {num_mc}\n")
    for i_config in range(len(config_list)):
        strs.append(f"-------------- Config {i_config}:\n{config_list[i_config]}\n")
    configs_output = "\n".join(strs)
    with open(os.path.join(directory, filename), "w") as f:
        f.write(configs_output)