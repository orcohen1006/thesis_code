import numpy as np
from utils import *
from scipy import sparse
from time import time

def fun_JBLD(Y, A, DAS_init, DOAscan, DOA, noise_power):
  
    Numsources = len(DOA)
    threshold = 1e-4
    MAX_ITERATIONS = 10000
    
    M, D = A.shape
    t_samples = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / t_samples
    p_vec_prev = np.abs(DAS_init) ** 2
    # p_vec_prev = DAS_init

    for iter in range(MAX_ITERATIONS):
        R = A @ sparse.diags(p_vec_prev) @ A.conj().T + noise_power * np.eye(M)
        S = R + R_hat
        invR = np.linalg.pinv(R)
        invS = np.linalg.pinv(S)

        invS_A = np.linalg.solve(S, A)
        invR_A = np.linalg.solve(R, A)

        z = np.maximum(0, np.real(np.sum(np.conj(A) * (invS_A), axis=0)))
        w = np.maximum(0, np.real(np.sum(np.conj(A) * (invR_A), axis=0)))
        gamma = z / (1 - z*p_vec_prev)
        delta = w / (1 - w*p_vec_prev)
        # !!!!!!!!!!!!!!!!!!!!!!!!!
        # gamma_tmp = gamma.copy()
        # delta_tmp = delta.copy()
        # for d in range(D):
        #     B_d = R - p_vec_prev[d] * (A[:, d] @ A[:, d].conj().T)
        #     gamma_tmp[d] = A[:, d].conj().T @ (np.linalg.pinv(B_d + R_hat) @ A[:, d])
        #     delta_tmp[d] = A[:, d].conj().T @ (np.linalg.pinv(B_d) @ A[:, d])
        # !!!!!!!!!!!!!!!!!!!!!!!!!

        p_vec = np.maximum(0, (delta - 2*gamma)/(gamma*delta))
        p_diffs_ratio = np.linalg.norm(p_vec_prev - p_vec) / np.linalg.norm(p_vec)
        if p_diffs_ratio < threshold:
            break
        p_vec_prev = p_vec.copy()

    p_vec = np.real(p_vec)

    return p_vec, iter, noise_power
