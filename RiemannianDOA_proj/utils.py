from unittest import case
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
import torch
import matplotlib.pyplot as plt
import pickle
import os

RUNNING_MPM = True

FILENAME_PBS_SCRIPT = "job_byOrCohen.pbs"
FILENAME_PBS_METADATA = "job_metadata.pkl"


EPS_REL_CHANGE = 1e-4
ALGONAME = "SERCOM"

class NormalizePowerType:
    NONE = 0
    MAX = 1
    DESIRED = 2
    PEAK_NEAR_DESIRED = 3
    PDF = 4

class SpectrumType:
    Bartlett = 0
    MVDR = 1
    MUSIC = 2

# =====================================================================
class GlobalParms: # trick to have a global variable that can be easily modified
    GRID_STEP_DEGREES = 0.5
    GRID_MIN_MAX_VALS_DEGREES = (0, 180)
    WANTED_ALGO_NAMES = {"SPICE", "SAMV", "AIRM", "JBLD", "LE", "ESPRIT"}
    SENSOR_ARRAY_TYPE = "ULA"
    SPECTRUM_TYPE = SpectrumType.MVDR
    SPECTRUM_NORMALIZATION = NormalizePowerType.NONE
    DELTA_FOR_DIAG_LOADING = 1e-4
globalParams = GlobalParms()
# =====================================================================

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
    num_sources = len(true_doas)

    if isinstance(p_vec, tuple):
        num_detected_doas = len(p_vec)
        all_detected_doas = np.array(p_vec)
        all_detected_powers = np.full((num_detected_doas,), np.nan)
        mean_HPBW = np.full((num_detected_doas,), np.nan)
    else:    
        if isinstance(p_vec, torch.Tensor):
            p_vec = p_vec.numpy()
        # Find peaks in descending order
        threshold_peak_height = max(allowed_peak_height_relative_to_max * np.max(p_vec), 0)
        peak_indices, _ = find_peaks(p_vec, height=threshold_peak_height)
        num_detected_doas = len(peak_indices)
        peak_indices = peak_indices[np.argsort(p_vec[peak_indices])[::-1]] # Sort the inidices by peaks values in descending order
        
        all_detected_doas = grid_doa[peak_indices]
        all_detected_powers = p_vec[peak_indices]
            
        mean_HPBW = compute_list_HPBW(p_vec, grid_doa, peak_indices)    

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
    


    return num_detected_doas, all_detected_doas, all_detected_powers, selected_doa_error, selected_power_error, \
            succ_match_detected_doa, succ_match_true_doa, mean_HPBW



def CreateSpectrum(G_hat, A, spectrum_type, normalize_type):
    if spectrum_type == SpectrumType.Bartlett:
        p_vec = np.sum(A.conj() * (G_hat @ A), axis=0).real
    elif spectrum_type == SpectrumType.MVDR:
        G_hat_inv_A = np.linalg.solve(G_hat + globalParams.DELTA_FOR_DIAG_LOADING * np.eye(A.shape[0]), A) 
        p_vec = 1 / np.sum(A.conj() * G_hat_inv_A, axis=0).real
    elif spectrum_type == SpectrumType.MUSIC:
        # Compute the noise subspace from the eigen-decomposition of G_hat
        eigvals, eigvecs = np.linalg.eigh(G_hat)
        # sort eigenvalues in descending order and eigenvectors accordingly
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        num_sources = np.argmax(eigvals[:-1] / eigvals[1:]) + 1 # Scree plot (largest eigenvalue ratio)
        # num_sources = 2; print(f"Using fixed num_sources={num_sources} for MUSIC spectrum")
        noise_subspace = eigvecs[:, num_sources:] # Take the eigenvectors corresponding to the smallest eigenvalues

        p_vec = 1.0 / np.sum(np.abs(noise_subspace.conj().T @ A)**2, axis=0).real
    else:
        raise ValueError(f"Unknown spectrum type: {spectrum_type}")
    

    # Normalization
    if normalize_type == NormalizePowerType.NONE:
        pass        
    elif normalize_type == NormalizePowerType.MAX:
        p_vec = p_vec / np.max(p_vec)
    elif normalize_type == NormalizePowerType.PDF:
        p_vec = p_vec / np.sum(p_vec)
    else:
        raise ValueError(f"illegal normalize type: {normalize_type}")


    return p_vec

def display_power_spectrum(config, list_p_vec, epsilon_power=None, algo_list=None, ax=None, normalize_power=NormalizePowerType.NONE,
                           do_legend=False, do_colorbar=True, algos_to_leave_out = []):
    """
    Display the power spectrum of the DOA estimation.

    :param config: Configuration dictionary
    :param list_p_vec: List of power vectors for different algorithms
    """
    import matplotlib.pyplot as plt

    power_doa_db = config["power_doa_db"]

    doa = config["doa"]


    grid_doa = get_doa_grid()

    if algo_list is None:
        algo_list = get_algo_dict_list()

    if epsilon_power is None:
        epsilon_power = 10.0 ** (-20 / 10.0)
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})        

    
    list_plt = []
    for i_algo, algo_name in enumerate(algo_list.keys()):
        if algo_name in algos_to_leave_out:
            continue
        label = f"{ALGONAME}({algo_name})" if (algo_name == "AIRM" or algo_name == "JBLD" or algo_name == "LE") else algo_name
        est = list_p_vec[i_algo]
        # check if est is a tuple (for ESPRIT)
        if isinstance(est, tuple):
            doa_est_degrees = est
            doa_to_display = np.array(doa_est_degrees)
            powers_to_display = 0*doa_to_display
            d = algo_list[algo_name].copy()
            d["linestyle"] = "none"
            pltobj, = ax.plot(doa_to_display, powers_to_display, label=label, **d)
        else:
            spectrum = est
            spectrum[spectrum < epsilon_power] = epsilon_power
            if normalize_power == NormalizePowerType.MAX:
                spectrum = spectrum / np.max(spectrum)
            elif normalize_power == NormalizePowerType.DESIRED:
                grid_index_desired_doa = np.argmin(np.abs(grid_doa - config["doa"][-1]))
                spectrum = spectrum / spectrum[grid_index_desired_doa]
            elif normalize_power == NormalizePowerType.PEAK_NEAR_DESIRED:
                grid_index_desired_doa = np.argmin(np.abs(grid_doa - config["doa"][-1]))
                # Find max in the neighborhood of the desired DOA
                neighborhood_indices = np.where(np.abs(grid_doa - config["doa"][-1]) <= 3.0)[0]
                local_peak = np.max(spectrum[neighborhood_indices])
                spectrum = spectrum / local_peak
            elif normalize_power == NormalizePowerType.PDF:
                spectrum = spectrum / np.sum(spectrum)
            spectrum = convert_linear_to_db(spectrum)

            curr_dict = {**algo_list[algo_name], "marker": "none", "alpha": 1, "linestyle": "-"}
            pltobj, = ax.plot(grid_doa, spectrum, label=label, **curr_dict)
            # pltobj, = ax.plot(grid_doa*np.pi/180, spectrum, label=label, **algo_list[algo_name])
            
        list_plt.append(pltobj)
    
    # plt_doa, = ax.plot(doa, power_doa_db, 'x', color='black', label='DOA')
    # list_plt.append(plt_doa)
    
    # ax.set_thetamin(0)
    # ax.set_thetamax(180)

    if do_colorbar:
        cbar = create_colorbar(algo_list, ax)
        
    for desired in config["doa"]:
        ax.axvline(x=desired, color='k', linestyle='-', linewidth=2)
    for interf in config["doa_interf"]:
        ax.axvline(x=interf, color='k', linestyle=':', linewidth=2)

    if do_legend:
        lgd = ax.legend(handles=list_plt)
        for text in lgd.get_texts():
            if "JBLD" in text.get_text():
                text.set_fontweight("bold")

    ax.set_xlabel(r"$\theta$ (degrees)", fontsize=12)
    ax.set_ylabel(r"$\mathrm{Power}$ (dB)", fontsize=12)
    
    # plt.title('Directions Power Spectrum Estimation')
    return ax

def extract_q_from_algo_name(algo_name):
    return float(algo_name.split('=')[1])

def create_colorbar(algo_list, ax):
    colormap = get_colormap()
    q_vals = np.array([extract_q_from_algo_name(algo_name) for algo_name in algo_list.keys() if algo_name.startswith("CMPM_q=")])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=q_vals.min(), vmax=q_vals.max()))
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', location="top",pad=0.05)
    cbar.set_label('$q$ value', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    return cbar


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


def generate_signal_with_interference(
        A_desired, power_desired_db,
        A_interf, power_interf_db, 
        N, noise_power,
        L, interference_segments_ind_mat,
        seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    M = A_desired.shape[0]
    K_desired = A_desired.shape[1]
    K_interf = A_interf.shape[1]
    amplitude_desired = np.sqrt(10.0 ** (power_desired_db / 10.0))
    amplitude_interf = np.sqrt(10.0 ** (power_interf_db / 10.0))
    # Generate signal
    noise = np.sqrt(noise_power / 2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
    waveform_desired = np.exp(1j * 2 * np.pi * np.random.rand(K_desired, N)) * np.tile(amplitude_desired, (N, 1)).T
    waveform_interf = np.exp(1j * 2 * np.pi * np.random.rand(K_interf, N)) * np.tile(amplitude_interf, (N, 1)).T

    # Zero out interference in segments where there is no interference
    for i_interf in range(K_interf):
        for l in range(L):
            start_ind, end_ind = get_segment_start_end_indices(N, L, l)
            if interference_segments_ind_mat[i_interf, l] == False:
                waveform_interf[i_interf, start_ind:end_ind] *= 0
    
    Y = A_desired @ waveform_desired + A_interf @ waveform_interf + noise
    return Y
    
def get_segment_start_end_indices(N, L, l):
    segment_len = np.ceil(N / L)
    start_ind = int(l * segment_len)
    end_ind = int(min((l + 1) * segment_len, N))
    return start_ind, end_ind





def make_non_circular(s, kappa):
    """
    Takes a circular signal s (K x N), returns non-circular version.
    """
    real = np.real(s)
    imag = np.imag(s)
    new_imag = np.sqrt(1 - kappa**2) * imag + kappa * real
    return real + 1j * new_imag

def get_doa_grid():
    step_deg = globalParams.GRID_STEP_DEGREES
    minval_deg, maxval_deg = globalParams.GRID_MIN_MAX_VALS_DEGREES
    doa_grid = np.arange(minval_deg, maxval_deg+step_deg, step_deg)
    return doa_grid

def get_steering_matrix(theta_degrees, M, calcGradient_wrt_radians=False):
    # write match-case for different array types:
    match globalParams.SENSOR_ARRAY_TYPE:
        case "ULA":
            return get_steering_matrix_ula(theta_degrees, M, calcGradient_wrt_radians)
        case "HALF_UCA":
            return get_steering_matrix_half_uca(theta_degrees, M, calcGradient_wrt_radians)
        case _:
            raise ValueError(f"Unknown SENSOR_ARRAY_TYPE: {globalParams.SENSOR_ARRAY_TYPE}")


def get_steering_matrix_ula(theta_degrees, M, calcGradient_wrt_radians=False):

    doa_rad = np.deg2rad(theta_degrees) # Convert to radians
    delta_vec = np.arange(M)    
    A = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa_rad)))
    
    if RUNNING_MPM:
        A = A / np.sqrt(M)
    
    # print("ULA:")
    # print(A[:,0])

    if calcGradient_wrt_radians:
        dA_dtheta_radians = -1j * np.pi * np.outer(delta_vec, np.sin(doa_rad)) * A  # shape (m, K)
        return A, dA_dtheta_radians
    return A

import numpy as np

def get_steering_matrix_half_uca(theta_degrees, M, calcGradient_wrt_radians=False):
    theta_degrees = np.asarray(theta_degrees)
    doa_rad = np.deg2rad(theta_degrees)        # (K,)
    K = doa_rad.size

    # choose R so arc spacing ≈ λ/2
    # Arc length = π R, (M-1) segments, each ≈ λ/2:
    # (M-1)*(λ/2) ≈ π R  =>  R/λ ≈ (M-1)/(2π)
    radius_over_lambda = (M - 1) / (2.0 * np.pi)

    # Sensor angles: half circle from -π/2 to +π/2
    sensor_angles = -0.5 * np.pi + np.pi * np.arange(M) / (M - 1)  # (M,)

    # kR = 2π (R/λ)
    kR = 2.0 * np.pi * radius_over_lambda

    # phase[m,k] = kR * cos(theta_k - phi_m)
    phase = kR * np.cos(doa_rad[None, :] - sensor_angles[:, None])  # (M, K)

    A = np.exp(1j * phase)  # (M, K)

    # print("HALF_UCA:")
    # print(A[:,0])
    
    if calcGradient_wrt_radians:
        # d/dθ cos(θ - φ) = -sin(θ - φ)
        dphase_dtheta = -kR * np.sin(doa_rad[None, :] - sensor_angles[:, None])  # (M, K)
        # a(θ) = exp(j phase(θ)), so:
        # dA/dθ = j * A * dphase_dtheta
        dA_dtheta_radians = 1j * A * dphase_dtheta  # (M, K)
        return A, dA_dtheta_radians

    return A


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

def estimate_num_sources(eigvals, Nsnap):
    """
    eigvals : eigenvalues (length M)
    Nsnap   : number of snapshots

    Returns:
        {"screeplot": ..., "MDL": ..., "AIC": ...}
    """

    eigvals = np.asarray(eigvals, dtype=float)
    M = len(eigvals)
    # sort in descending order:
    eigvals = np.sort(eigvals)[::-1]

    # Scree plot (largest eigenvalue ratio)
    scree = np.argmax(eigvals[:-1] / eigvals[1:]) + 1

    mdl = np.empty(M)
    aic = np.empty(M)

    for k in range(M):
        m = M - k
        noise_eigs = eigvals[k:]

        am = noise_eigs.mean()
        gm = np.exp(np.mean(np.log(noise_eigs)))

        ll = Nsnap * m * np.log(am / gm)

        mdl[k] = ll + 0.5 * k * (2 * M - k) * np.log(Nsnap)
        aic[k] = 2 * ll + 2 * k * (2 * M - k)
    # don't allow 0 sources to be selected
    mdl[0] = np.inf
    aic[0] = np.inf
    return {
        "screeplot": int(scree),
        "MDL": int(np.argmin(mdl)),
        "AIC": int(np.argmin(aic)),
    }

def get_colormap():
    # return plt.cm.vanimo
    return plt.cm.managua

def define_all_algo_dict_list():
    if RUNNING_MPM:
        q_vals = np.arange(-1, 1.01, 0.25)
        colormap = get_colormap()
        keys = [f"CMPM_q={q}" for q in q_vals]
        # linewidth = 1.5
        # d = {key: {"linestyle": "-", "color": colormap(i / (len(q_vals)-1)), "marker": "o", "markersize": 4, "linewidth": linewidth} 
        linewidth = 2.5
        d = {key: {"linestyle": "-", "color": colormap(i / (len(q_vals)-1)), "linewidth": linewidth, "alpha": 0.70,
                   "marker": "none", "markerfacecolor": "none", "markersize": 6} 
             for i, key in enumerate(keys)}
        #
        # d["CMPM_q=0.0"]["linewidth"] = linewidth
        d["CMPM_q=0.0"]["linestyle"] = "--"
        d["CMPM_q=0.0"]["marker"] = "o"
        d["CMPM_q=0.0"]["alpha"] = 1
        

        # d["CMPM_q=1.0"]["linewidth"] = linewidth
        d["CMPM_q=1.0"]["linestyle"] = "--"
        d["CMPM_q=1.0"]["marker"] = "o"
        d["CMPM_q=1.0"]["alpha"] = 1

        # add other algorithms with fixed styles
        d.update({
            "MinSpectrum": {"linestyle": ":", "color": "#CF0505", "linewidth": linewidth, 
                   "marker": "o", "markerfacecolor": "none", "markersize": 5},
        })
        
        d.update({
            "qstar": {"linestyle": "--", "color": "#05B305", "linewidth": linewidth/2, 
                   "marker": "*", "markerfacecolor": "none", "markersize": 5},
        })
        # d.update({
        #     "ProjectOutInterf": {"linestyle": "--", "color": "#FF0000", "linewidth": linewidth, 
        #            "marker": "o", "markerfacecolor": "none", "markersize": 5},
        # })
        
        # d.update({
        #     "OptimalCMPM": {"linestyle": "--", "color": "#FFA4F7", "linewidth": linewidth, 
        #            "marker": "o", "markerfacecolor": "none", "markersize": 5},
        # })

        # d.update({
        #     "OptimalNI": {"linestyle": "--", "color": "#FF00EA", "linewidth": linewidth, 
        #            "marker": "o", "markerfacecolor": "none", "markersize": 5},
        # })
    else:
        linewidth = 2
        d = {
            "SPICE": {"linestyle": "--", "color": "#BBB800FF", "marker": "s", "markersize": 4, "linewidth": linewidth},
            "SAMV":  {"linestyle": "--", "color": "#E65908", "marker": "^", "markersize": 5.5, "linewidth": linewidth},
            "AIRM":  {"linestyle": "-", "color": "#0CBD56", "marker": "o", "linewidth": linewidth},
            "LE": {"linestyle": "-.", "color": "m", "marker": "s", "markersize": 4, "linewidth": linewidth},
            "JBLD":  {"linestyle": "-", "color": "#2B27FF", "marker": "o", "markerfacecolor": "none", "markersize": 8, "linewidth": linewidth},
            "PER": {"linestyle": ":", "color": "y", "marker": "^"},
            # "LE_ss": {"linestyle": "-.", "color": "m", "marker": "s", "markersize": 4, "linewidth": linewidth},
            "MVDR": {"linestyle": "--", "color": "c", "marker": "o", "markersize": 6},
            "ESPRIT": {"linestyle": "--", "color": "black", "marker": "o", "markerfacecolor": "none", "markersize": 6},
            }

    return d

def get_segements_dict_list(L):
    import matplotlib as mpl
    colormap = mpl.colormaps['Accent']
    seg_ids = np.arange(1, L+1)
    keys = [f"segment {seg_id}" for seg_id in seg_ids]
    linewidth = 2.5
    d = {key: {"linestyle": "-.", "color": colormap(i), "linewidth": linewidth} 
            for i, key in enumerate(keys)}
    return d

def get_algo_dict_list():
    # all_algo_list = define_all_algo_dict_list()
    # wanted_algo_names = globalParams.WANTED_ALGO_NAMES
    # algo_list = {k: v for k, v in all_algo_list.items() if k in wanted_algo_names}
    # return algo_list
    return define_all_algo_dict_list()

def get_specific_inorder_algo_list(specific_algo_names):
    all_algo_list = define_all_algo_dict_list()
    algo_list = {k: all_algo_list[k] for k in specific_algo_names if k in all_algo_list}

    return algo_list

# def create_config(m, snr, N, power_doa_db, doa, cohr_flag=False, cohr_coeff=1.0, noncircular_coeff=0.0, 
#                   impulse_prob=0.0, impulse_factor=1.0):
#     return {
#         "m": m,
#         "snr": snr,
#         "N": N,
#         "power_doa_db": power_doa_db,
#         "doa": doa,
#         "cohr_flag": cohr_flag,
#         "cohr_coeff": cohr_coeff,
#         "noncircular_coeff": noncircular_coeff,
#         "impulse_prob": impulse_prob,
#         "impulse_factor": impulse_factor,
#     }
def create_config(m, snr, N, power_doa_db, doa, power_doa_interf_db, doa_interf, L):
    return {
        "m": m,
        "snr": snr,
        "N": N,
        "power_doa_db": power_doa_db,
        "doa": doa,
        "power_doa_interf_db": power_doa_interf_db,
        "doa_interf": doa_interf,
        "L": L,
        "cohr_flag": False,
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



def get_G_tensor(Y, L):
    M, N = Y.shape
    G_tensor = np.zeros((L,M,M), dtype=complex)
    for l in range(L):
        start, end = get_segment_start_end_indices(N, L, l)
        W = end - start
        Yl = Y[:, start:end]
        G = (Yl @ Yl.conj().T) / W
        G_tensor[l,:,:] = (G + G.conj().T) * 0.5
    return G_tensor


def calc_nismse(curr_p_vec, p_vec_ni, grid_index, half_window_num_grid_points):
    # Extract the power values around the desired DOA index
    start_index = max(0, grid_index - half_window_num_grid_points)
    end_index = min(len(curr_p_vec), grid_index + half_window_num_grid_points)
    
    curr_window = curr_p_vec[start_index:end_index+1]
    ni_window = p_vec_ni[start_index:end_index+1]

    # Normalize the power values
    # curr_window = curr_window / curr_p_vec[grid_index_desired_doa]
    # ni_window = ni_window / p_vec_ni[grid_index_desired_doa]

    # Calculate the MSE in this window
    nismse = np.mean((curr_window - ni_window) ** 2)
    
    
    return nismse



def esprit(R_hat, num_sources):
    M = R_hat.shape[0]
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(R_hat)
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    
    Us = eigvecs[:, :num_sources]     # (M x num_sources)

    # 2) Selection matrices for shift invariance
    #    J1 picks sensors 0..M-2, J2 picks sensors 1..M-1
    J1 = np.eye(M - 1, M, k=0)  # (M-1 x M)
    J2 = np.eye(M - 1, M, k=1)  # (M-1 x M)

    Us1 = J1 @ Us               # (M-1 x num_sources)
    Us2 = J2 @ Us               # (M-1 x num_sources)

    # 3) Solve Us2 ≈ Us1 * Psi  (least-squares)
    Psi = np.linalg.pinv(Us1) @ Us2   # (num_sources x num_sources)

    # 4) Eigen-decomposition of Psi
    eigvals, eigvecs = np.linalg.eig(Psi)

    # 5) Map eigenvalues to DOAs using your steering convention:
    #    eigvals ≈ exp(1j * pi * cos(theta))
    phi = np.angle(eigvals)          # in (-pi, pi]
    cos_theta = phi / np.pi
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety

    doa_deg = np.arccos(cos_theta) * 180.0 / np.pi  # in [0, 180]

    # sort by angle
    sort_idx = np.argsort(doa_deg)
    doa_deg = doa_deg[sort_idx]

    return tuple(doa_deg)
