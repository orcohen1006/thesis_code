import numpy as np
from utils import *
from scipy import linalg
from time import time
from utils import EPS_REL_CHANGE


def fun_LE_ss(Y, A, DAS_init, DOAscan, DOA, sigma2=None):
    
    M, D = A.shape
    N = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / N 
    # A = A / np.sqrt(M)
    logm_R_hat = linalg.logm(R_hat)    
    p = np.sum(A.conj() * (logm_R_hat @ A), axis=0).real
    p = np.exp(p) - sigma2
    
    return p, 0, sigma2
