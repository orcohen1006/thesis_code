import numpy as np
from scipy.signal import find_peaks

def fun_DASRes(Y, A, DAS_init, DOAscan, DOA):
    """
    DAS (Delay-And-Sum) estimator implementation.
    
    Parameters:
    Y: measured data, each col. is one snapshot
    A: steering vector matrix
    DAS_init: initial coefficients estimates by DAS
    DOAscan: grid
    DOA: truth (actual DOA angles)
    
    Returns:
    Detected_powers: powers of detected sources
    Distance: difference between detected and true DOAs
    p_vec: power spectrum vector
    normal: tag (1 if detection OK, 0 if failed)
    noisepower: estimated noise power (NaN for DAS)
    """
    noisepower = np.nan  # not able to give it for DAS
    M, thetaNum = A.shape
    t_samples = Y.shape[1]
    
    # Calculate modulus by DAS
    modulus_hat_das = np.sum(np.abs(A.conj().T @ Y / M), axis=1) / t_samples
    p_vec = np.abs(modulus_hat_das) ** 2
    
    Numsources = len(DOA)
    
    # Find peaks in descending order
    peaks, indices = find_peaks(p_vec)
    peak_values = p_vec[indices]
    sorted_idx = np.argsort(-peak_values)  # Sort in descending order
    indices = indices[sorted_idx]
    
    if len(indices) < Numsources:
        # Not all peaks detected
        normal = 0
        Distance = np.nan
        Detected_powers = np.nan
        return Detected_powers, Distance, p_vec, normal, noisepower
    
    # Check whether the detection is right
    Detected_DOAs = DOAscan[indices[:Numsources]]
    
    # Sort detected DOAs in ascending order
    sorted_idx = np.argsort(Detected_DOAs)
    Detected_DOAs = Detected_DOAs[sorted_idx]
    Distance = Detected_DOAs - DOA
    
    normal = 1  # detection okay
    # The powers from large value to small value
    Detected_powers = p_vec[indices[:Numsources]]
    # Sort the power according to the DOA
    Detected_powers = Detected_powers[sorted_idx]
    
    return Detected_powers, Distance, p_vec, normal, noisepower
