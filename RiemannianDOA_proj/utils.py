import numpy as np
from scipy.signal import find_peaks

def detect_DOAs(p_vec, DOAscan, DOA):
    Numsources = len(DOA)
    # Find peaks in descending order
    peaks_indices, _ = find_peaks(p_vec)
    peak_values = p_vec[peaks_indices]
    sorted_idx = np.argsort(-peak_values)  # Sort in descending order
    peaks_indices = peaks_indices[sorted_idx]

    if len(peaks_indices) < Numsources:
        # Not all peaks detected
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