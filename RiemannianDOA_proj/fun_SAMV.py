import numpy as np
from utils import *
from scipy import sparse
from time import time
from utils import EPS_REL_CHANGE

def fun_SAMV(Y, A, DAS_init, DOAscan, DOA, sigma_given=None):
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
    # t0 = time()
    flag_sigma_is_given = sigma_given is not None
    
    Numsources = len(DOA)
    maxIter = 10000
    
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
        
        p_diffs_ratio = np.linalg.norm(p_vec_Old - p_vec) / (1e-5 + np.linalg.norm(p_vec_Old))
        if p_diffs_ratio < EPS_REL_CHANGE:
            break
        
        p_vec_Old = p_vec.copy()

    p_vec = np.real(p_vec)

    # print(f"samv: #iters= {iterIdx}, time= {time() - t0} [sec]")
    # Detected_powers, Distance, normal = detect_DOAs(p_vec, DOAscan, DOA)

    noisepower = sigma

    return p_vec, iterIdx, noisepower
