import numpy as np
from scipy.signal import find_peaks
from scipy import sparse

def fun_SAM3Res(Y, A, DAS_init, DOAscan, DOA, sigma_given=None):
    """
    SAM-3 (Sparse And Matched 3) estimator implementation.
    
    Parameters:
    Y: measured data, each col. is one snapshot
    A: steering vector matrix
    DAS_init: initial coefficients estimates by DAS
    DOAscan: grid
    DOA: truth (actual DOA angles)
    sigma_given: noise power if known (optional)
    
    Returns:
    Detected_powers: powers of detected sources
    Distance: difference between detected and true DOAs
    p_vec: power spectrum vector
    normal: tag (1 if detection OK, 0 if failed)
    noisepower: estimated noise power
    """
    flag_sigma_is_given = sigma_given is not None
    
    Numsources = len(DOA)
    threshold = 1e-4
    maxIter = 10000  # Original was 1e4;30
    
    M, thetaNum = A.shape
    t_samples = Y.shape[1]
    R_N = (Y @ Y.conj().T) / t_samples
    
    if flag_sigma_is_given:
        sigma = sigma_given
    else:
        sigma = np.mean(np.abs(Y.flatten()) ** 2)
    
    p_vec_Old = np.abs(DAS_init) ** 2
    
    for iterIdx in range(maxIter):
        # Create sparse diagonal matrix with p_vec_Old
        R = A @ sparse.diags(p_vec_Old) @ A.conj().T + sigma * np.eye(M)
        
        Rinv = np.linalg.inv(R)
        Rinv_A = Rinv @ A
        diag_A_Rinv_A = np.sum(A.conj() * Rinv_A, axis=0)
        tmp = np.sum(A.conj() * (Rinv @ R_N @ Rinv @ A), axis=0)
        p_vec = p_vec_Old * (tmp / diag_A_Rinv_A)
        
        if not flag_sigma_is_given:
            sigma = np.real(np.trace(Rinv @ Rinv @ R_N)) / np.real(np.trace(Rinv @ Rinv))
        
        p_diffs_ratio = np.linalg.norm(p_vec_Old - p_vec) / np.linalg.norm(p_vec_Old)
        if p_diffs_ratio < threshold:
            break
        
        p_vec_Old = p_vec
    
    p_vec = np.real(p_vec)
    
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
        noisepower = sigma
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
    
    noisepower = sigma
    
    return Detected_powers, Distance, p_vec, normal, noisepower
