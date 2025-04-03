import numpy as np
from scipy.signal import find_peaks

def detect_DOAs(p_vec, DOAscan, DOA):
    Numsources = len(DOA)
    # Find peaks in descending order
    peaks_indices, _ = find_peaks(p_vec)
    peak_values = p_vec[peaks_indices]
    sorted_idx = np.argsort(-peak_values)  # Sort in descending order
    peaks_indices = peaks_indices[sorted_idx]

    if (not isinstance(peaks_indices, np.ndarray)) or len(peaks_indices) < Numsources:
        # Not all peaks detected
        print("Not all peaks detected ")
        normal = 0
        Distance = np.nan
        Detected_powers = np.nan
        return Detected_powers, Distance, normal

    # Check whether the detection is right
    Detected_DOAs = DOAscan[peaks_indices[:Numsources]]

    # Sort detected DOAs in ascending order
    sorted_idx = np.argsort(Detected_DOAs)
    Detected_DOAs = Detected_DOAs[sorted_idx]
    Distance = Detected_DOAs - DOA

    normal = 1  # detection okay
    # The powers from large value to small value
    Detected_powers = p_vec[peaks_indices[:Numsources]]
    # Sort the power according to the DOA
    Detected_powers = Detected_powers[sorted_idx]

    return Detected_powers, Distance, normal


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