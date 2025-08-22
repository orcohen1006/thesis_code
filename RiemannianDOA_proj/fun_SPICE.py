import numpy as np
from utils import *
from scipy import linalg
from time import time
from utils import EPS_REL_CHANGE


def fun_SPICE(Y, A, DAS_init, DOAscan, DOA, sigma_given=None):
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
    # t0 = time()
    flag_sigma_is_given = sigma_given is not None
    Numsources = len(DOA)
    
    maxIter = 10000
    
    M, N = A.shape
    t_samples = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / t_samples
    
    # Initialize power vector
    p = np.abs(DAS_init) ** 2
    S = np.sort(p)
    
    # Estimate initial sigma
    sigmainit = np.mean(S[:M])

    invR_hat_A = np.linalg.solve(R_hat, A)
    
    # Compute weight using conventional way
    weight_first = np.real(np.sum(A.conj() * invR_hat_A, axis=0)) / M
    
    weight = np.real(np.concatenate([weight_first, np.diag(np.linalg.inv(R_hat)).conj() / M]))
    gamma = np.real(np.mean(np.diag(np.linalg.inv(R_hat))))
    
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

        am_Rinv_R_hat_sqrt = A.conj().T @ Rinv_R_hat_sqrt
        norm_am_Rinv_R_hat_sqrt = np.linalg.norm(am_Rinv_R_hat_sqrt, axis=1)
        rho = np.sum(np.sqrt(weight[:N].T) * p * norm_am_Rinv_R_hat_sqrt)

        # Keep the ||F for future use
        norm_Rinv_Rhatsqrt = np.sqrt(np.sum(np.diag(Rinv_R_hat_sqrt.conj().T @ Rinv_R_hat_sqrt)))
        rho = rho + np.sqrt(gamma) * sigma * norm_Rinv_Rhatsqrt
        
        # Compute sigma
        if not flag_sigma_is_given:
            sigma = sigma * norm_Rinv_Rhatsqrt / (np.sqrt(gamma) * rho)
        
        # Compute p
        # for pidx in range(N):
        #     p[pidx] = p[pidx] * norm_am_Rinv_R_hat_sqrt[pidx] / (rho * np.sqrt(weight[pidx]))
        p = p * norm_am_Rinv_R_hat_sqrt / (rho * np.sqrt(weight[:N]))

        p = np.abs(p)
        
        measured_change_norm = np.linalg.norm(p - p_prev) / (1e-5 + np.linalg.norm(p_prev))
        if measured_change_norm < EPS_REL_CHANGE:
            break
    # print(f"spice: #iters= {jj}, time= {time() - t0} [sec]")
    # Detected_powers, Distance, normal = detect_DOAs(p, DOAscan, DOA)

    noisepower = sigma
    
    return p, jj, noisepower
