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



def fun_PER(Y, A, noisepower):

    M, D = A.shape
    scaler = np.linalg.norm(A[:,0])  # assuming all steering vectors have same norm
    A = A / scaler

    sigma2 = noisepower
    M, D = A.shape
    N = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / N
    p_vec = np.diag(A.conj().T @ R_hat @ A).real 

    p_vec = np.maximum(p_vec - noisepower, 0)
    p_vec = p_vec / (scaler**2)
    num_iters = 0

    return p_vec, num_iters, noisepower

def fun_MVDR(Y, A, DAS_init, DOAscan, DOA, noisepower):

    
    M, D = A.shape
    N = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / N
    invR_hat_A = np.linalg.solve(R_hat, A)
    tmp = np.sum(A.conj() * invR_hat_A, axis=0).real
    p_vec = np.maximum(1/tmp - noisepower, 0)
    num_iters = 0

    return p_vec, num_iters, noisepower
