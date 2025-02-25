import numpy as np
from scipy.signal import find_peaks
from scipy import linalg

def fun_SPICEplusRes(Y, A, DAS_init, DOAscan, DOA, sigma_given=None):
    """
    SPICE+ (SParse Iterative Covariance-based Estimation) implementation.
    
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
    p: power spectrum vector
    normal: tag (1 if detection OK, 0 if failed)
    noisepower: estimated noise power
    """
    flag_sigma_is_given = sigma_given is not None
    Numsources = len(DOA)
    
    maxIter = 10000  # Original was 1e4;25
    EPS_NORM_CHANGE = 1e-4
    
    M, N = A.shape
    t_samples = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / t_samples
    
    # Initialize power vector
    p = np.abs(DAS_init) ** 2
    S = np.sort(p)
    
    # Estimate initial sigma
    sigmainit = np.mean(S[:M])
    
    # Make R_hat Toeplitz using sample mean (commented out in original code)
    # R_hat_row = np.zeros(M)
    # for setvalue in range(M):
    #     R_hat_row[setvalue] = np.mean(np.diag(R_hat, setvalue))
    # R_hat = linalg.toeplitz(R_hat_row)
    
    invR_hat_A = np.linalg.solve(R_hat, A)
    
    # Compute weight using conventional way
    weight_first = np.real(np.sum(A.conj() * invR_hat_A, axis=0)) / M
    
    weight = np.concatenate([weight_first, np.diag(np.linalg.inv(R_hat)).conj() / M])
    gamma = np.mean(np.diag(np.linalg.inv(R_hat)))
    
    if flag_sigma_is_given:
        sigma = sigma_given
    else:
        sigma = sigmainit  # using SPICE+, with equal sigma constraint
    
    for jj in range(maxIter):
        p_prev = p.copy()
        
        # Prepare
        P = np.diag(p)
        R = A @ P @ A.conj().T + sigma * np.eye(M)
        R_hat_sqrt = linalg.sqrtm(R_hat)
        Rinv_R_hat_sqrt = np.linalg.solve(R, R_hat_sqrt)
        
        # Compute rho
        rho = 0
        
        am_Rinv_R_hat_sqrt = np.zeros((N, M), dtype=complex)
        norm_am_Rinv_R_hat_sqrt = np.zeros(N)
        
        for idx in range(N):
            am_Rinv_R_hat_sqrt[idx, :] = A[:, idx].conj().T @ Rinv_R_hat_sqrt
            norm_am_Rinv_R_hat_sqrt[idx] = np.sqrt(np.sum(np.diag(
                am_Rinv_R_hat_sqrt[idx, :].reshape(-1, 1) @ 
                am_Rinv_R_hat_sqrt[idx, :].conj().reshape(1, -1)
            )))
            rho = rho + np.sqrt(weight[idx]) * p[idx] * norm_am_Rinv_R_hat_sqrt[idx]
        
        # Keep the ||F for future use
        norm_Rinv_Rhatsqrt = np.sqrt(np.sum(np.diag(Rinv_R_hat_sqrt.conj().T @ Rinv_R_hat_sqrt)))
        rho = rho + np.sqrt(gamma) * sigma * norm_Rinv_Rhatsqrt
        
        # Compute sigma
        if not flag_sigma_is_given:
            sigma = sigma * norm_Rinv_Rhatsqrt / (np.sqrt(gamma) * rho)
        
        # Compute p
        for pidx in range(N):
            p[pidx] = p[pidx] * norm_am_Rinv_R_hat_sqrt[pidx] / (rho * np.sqrt(weight[pidx]))
        
        p = np.abs(p)
        
        measured_change_norm = np.linalg.norm(p - p_prev) / np.linalg.norm(p)
        if measured_change_norm < EPS_NORM_CHANGE:
            break
    
    # Find peaks in descending order
    peaks, indices = find_peaks(p)
    peak_values = p[indices]
    sorted_idx = np.argsort(-peak_values)  # Sort in descending order
    indices = indices[sorted_idx]
    
    if len(indices) < Numsources:
        # Not all peaks detected
        normal = 0
        Distance = np.nan
        Detected_powers = np.nan
        noisepower = sigma
        return Detected_powers, Distance, p, normal, noisepower
    
    # Check whether the detection is right
    Detected_DOAs = DOAscan[indices[:Numsources]]
    
    # Sort detected DOAs in ascending order
    sorted_idx = np.argsort(Detected_DOAs)
    Detected_DOAs = Detected_DOAs[sorted_idx]
    Distance = Detected_DOAs - DOA
    
    normal = 1  # detection okay
    # The powers from large value to small value
    Detected_powers = p[indices[:Numsources]]
    # Sort the power according to the DOA
    Detected_powers = Detected_powers[sorted_idx]
    
    noisepower = sigma
    
    return Detected_powers, Distance, p, normal, noisepower
