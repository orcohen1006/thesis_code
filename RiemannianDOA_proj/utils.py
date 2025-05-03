import numpy as np
from scipy.signal import find_peaks
import torch

FILENAME_PBS_SCRIPT = "job_byOrCohen.pbs"
FILENAME_PBS_METADATA = "job_metadata.pkl"

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

def estimate_doa_calc_errors(p_vec, grid_doa, doa, power):
    if isinstance(p_vec, torch.Tensor):
        p_vec = p_vec.numpy()
    num_sources = len(doa)
    # Find peaks in descending order
    peak_indices, _ = find_peaks(p_vec)
    peak_values = p_vec[peak_indices]
    sorted_indices = np.argsort(-peak_values)  # Sort in descending order
    peak_indices = peak_indices[sorted_indices]
    peak_indices = np.atleast_1d(peak_indices)
    if (not isinstance(peak_indices, np.ndarray)) or len(peak_indices) < num_sources:
        # print("Not all peaks detected")
        detection_status = 0
        detected_doas = np.full((num_sources,), np.nan)
        detected_powers = np.full((num_sources,), np.nan)
        doa_error = np.full((num_sources,), np.nan)
        power_error = np.full((num_sources,), np.nan)
    else:
        detection_status = 1  # detection successful

        detected_doas = grid_doa[peak_indices[:num_sources]]
        detected_powers = p_vec[peak_indices[:num_sources]]

        # Sort doas ascending order (and powers accordingly)
        sorted_indices = np.argsort(detected_doas)
        detected_doas = detected_doas[sorted_indices]
        detected_powers = detected_powers[sorted_indices]

        sorted_indices = np.argsort(doa)
        doa = doa[sorted_indices]
        power = power[sorted_indices]

        doa_error = detected_doas - doa
        power_error = detected_powers - power

        # Sort the power according to the DOA
        
    return detection_status, detected_doas, detected_powers, doa_error, power_error
    

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
    

def generate_signal(A_true, power_doa_db, t_samples, noise_power, cohr_flag=False, seed=None):
    if seed is not None:
        np.random.seed(seed)

    m = A_true.shape[0]
    num_sources = len(power_doa_db)
    amplitude_doa = np.sqrt(10.0 ** (power_doa_db / 10.0))

    # Generate signal
    noise = np.sqrt(noise_power / 2) * (np.random.randn(m, t_samples) + 1j * np.random.randn(m, t_samples))

    if not cohr_flag:  # independent sources
        waveform = np.exp(1j * 2 * np.pi * np.random.rand(num_sources, t_samples))
        waveform = waveform * np.tile(amplitude_doa, (t_samples, 1)).T
    else:  # coherent sources
        waveform = np.exp(1j * 2 * np.pi * np.random.rand(num_sources - 1, t_samples))
        waveform = np.vstack([waveform, waveform[0, :]])
        waveform = waveform * np.tile(amplitude_doa, (t_samples, 1)).T

    y_noisefree = A_true @ waveform  # ideal noiseless measurements
    y_noisy = y_noisefree + noise  # noisy measurements

    return y_noisy

def get_doa_grid():
    doa_scan = np.arange(0, 180.5, 0.5)  # doa grid
    # doa_scan = np.arange(0, 181, 1)  # doa grid
    return doa_scan

def get_algo_dict_list(flag_also_use_PER=False):
    # return [("PER",'r-->'), ("SPICE",'m--p'), ("SAMV",'b-^'), ("AIRM",'g--s'), ("JBLD",'y--o')]
    # return [("SPICE",'m--p'), ("SAMV",'r--^'), ("AIRM",'g-s'), ("JBLD",'b--o')]
    d = {
        "SPICE": {"linestyle": ":", "color": "m", "marker": "p"},
        "SAMV":  {"linestyle": ":", "color": "r", "marker": "s"},
        "AIRM":  {"linestyle": "-", "color": "g", "marker": "o"},
        "JBLD":  {"linestyle": "--", "color": "b", "marker": "o", "markerfacecolor": "none", "markersize": 8},
    }
    if flag_also_use_PER:
        d = {"PER": {"linestyle": ":", "color": "y", "marker": "^"}, **d}
    return d


def create_config(m, snr, N, power_doa_db, doa, cohr_flag=False):
    """
    Create a configuration dictionary to hold parameters for simulations.

    :param m: Number of sensors
    :param snr: Signal-to-noise ratio
    :param N: Number of snapshots
    :param power_doa_db: Power of DOAs in dB
    :param doa: Directions of arrival (DOAs) in degrees
    :return: Configuration dictionary
    """
    return {
        "m": m,
        "snr": snr,
        "N": N,
        "power_doa_db": power_doa_db,
        "doa": doa,
        "cohr_flag": cohr_flag,
    }