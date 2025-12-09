import numpy as np
from utils import *
from scipy import linalg
from time import time
from utils import EPS_REL_CHANGE


def fun_LE_ss(Y, A, DAS_init, DOAscan, DOA, noisepower):

    M, D = A.shape
    scaler = np.linalg.norm(A[:,0])  # assuming all steering vectors have same norm
    A = A / scaler

    sigma2 = noisepower

    N = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / N
    # normalize by noise power and set sigma_n^2 to 1
    R_hat /= sigma2

    logm_R_hat = linalg.logm(R_hat)    
    p = np.sum(A.conj() * (logm_R_hat @ A), axis=0).real
    p = np.exp(p) - 1

    
    p = p * sigma2
    p = p / (scaler**2)
    
    return p, 0, noisepower
