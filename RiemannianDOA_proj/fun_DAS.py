import numpy as np
from utils import *

def fun_DAS(Y, A, DAS_init, DOAscan, DOA):
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
    # Detected_powers, Distance, normal = detect_DOAs(p_vec, DOAscan, DOA)

    num_iters = 0

    return p_vec, num_iters, noisepower
