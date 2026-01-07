import numpy as np
from utils import *
from scipy import sparse
from time import time
from utils import EPS_REL_CHANGE
import scipy.linalg as linalg
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
    maxIter = 5000
    
    M, thetaNum = A.shape
    t_samples = Y.shape[1]
    R_N = (Y @ Y.conj().T) / t_samples
    
    sigma = sigma_given
    
    p_vec_Old = np.abs(DAS_init) ** 2
    
    anytmp = False
    maxmax = 0.0

    for iterIdx in range(maxIter):
        print(f"SAMV iteration {iterIdx}/{maxIter}", end='\r')

        R = (A * p_vec_Old[np.newaxis, :]) @ A.conj().T + sigma * np.eye(M)

        # --------- Old and slower: ------------------------
        # Rinv = np.linalg.inv(R)
        # Rinv_A = Rinv @ A
        # tmp = np.sum(A.conj() * (Rinv @ R_N @ Rinv @ A), axis=0)
        # --------------------------------------------------------------

        # --------- opt2: ------------------------
        # Rinv = np.linalg.solve(R, np.eye(M))
        # Rinv_A = Rinv @ A
        # tmp = np.sum(A.conj() * (Rinv @ R_N @ Rinv_A), axis=0)
        # --------------------------------------------------------------

        # --------- opt3: ------------------------
        # Rinv = np.linalg.solve(R, np.eye(M))
        # Rinv_A = Rinv @ A
        # tmp = np.sum(Rinv_A.conj() * (R_N @ Rinv_A), axis=0).real
        # tmp = np.maximum(tmp, 0)
        # --------------------------------------------------------------


        # --------- check: ------------------------
        # Rinv = np.linalg.solve(R, np.eye(M))
        # Rinv_A = Rinv @ A
        # tmp2 = np.sum(A.conj() * (Rinv @ R_N @ Rinv_A), axis=0).real
        # tmp3 = np.sum(Rinv_A.conj() * (R_N @ Rinv_A), axis=0).real
        # max_abs_diff_tmp = np.max(np.abs(tmp2 - tmp3))
        # if maxmax < max_abs_diff_tmp:
        #     i_max = iterIdx
        #     maxmax = max_abs_diff_tmp 
        # anytmp = anytmp or np.any(tmp3<0)
        # print(f"maxmax: {maxmax}, at iter {i_max} any negative tmp3: {anytmp}")
        # tmp = tmp2
        # --------------------------------------------------------------

        # --------- More efficient calculation: ------------------------
        Rinv_A = np.linalg.solve(R, A)
        tmp = np.sum(Rinv_A.conj() * (R_N @ Rinv_A), axis=0)      
        # --------------------------------------------------------------
        
        
        diag_A_Rinv_A = np.sum(A.conj() * Rinv_A, axis=0)
        p_vec = (p_vec_Old * (tmp / diag_A_Rinv_A)).real
        
        
        p_diffs_ratio = np.linalg.norm(p_vec_Old - p_vec) / (1e-5 + np.linalg.norm(p_vec_Old))
        if p_diffs_ratio < EPS_REL_CHANGE:
            break
        
        p_vec_Old = p_vec.copy()

    p_vec = np.real(p_vec)

    noisepower = sigma

    return p_vec, iterIdx, noisepower
