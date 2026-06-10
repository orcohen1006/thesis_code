import numpy as np
from utils import *
from mpm import mpm
from time import time

USE_MVDR = False
DELTA_FOR_DIAG_LOADING = 1e-3

def fun_CMPM(Y, A, L, q, noise_power):

    eps_eigval = 1e-10
    eps_q = 1e-10
    M, N = Y.shape
    G_tensor = get_G_tensor(Y, L)

    '''                             OLD
    # ===================================
    # ===================================
    # ===================================
    if np.abs(q - 0) < eps_q: # q == 0
        # Log-Euclidean mean: exp( mean(log(G_l)) )
        S = np.zeros((M, M), dtype=Y.dtype)
        for l in range(L):
            evals, evecs = np.linalg.eigh(G_tensor[l,:,:])
            evals = np.maximum(evals.real, eps_eigval)
            loge = np.log(evals)
            logG = (evecs * loge[None, :]) @ evecs.conj().T
            S += logG
        S /= L
        evalsS, evecsS = np.linalg.eigh((S + S.conj().T) * 0.5)
        G_hat = (evecsS * np.exp(evalsS.real)[None, :]) @ evecsS.conj().T
    elif np.abs(q - 1) < eps_q: # q == 1
        G_hat = np.mean(G_tensor, axis=0)
    else:
        # ( mean(G_l^q) )^(1/q)
        Q = np.zeros((M, M), dtype=Y.dtype)
        for l in range(L):
            evals, evecs = np.linalg.eigh(G_tensor[l,:,:])
            evals = np.maximum(evals.real, eps_eigval)
            evals_q = evals ** q
            Gq = (evecs * evals_q[None, :]) @ evecs.conj().T
            Q += Gq
        Q /= L
        Q = (Q + Q.conj().T) * 0.5

        evalsQ, evecsQ = np.linalg.eigh(Q)
        evalsQ = np.maximum(evalsQ.real, eps_eigval)
        evals_1q = evalsQ ** (1.0 / q)
        G_hat = (evecsQ * evals_1q[None, :]) @ evecsQ.conj().T

    #
    G_hat = (G_hat + G_hat.conj().T) * 0.5
    # ===================================
    # ===================================
    # ===================================
    '''
    t0 = time()
    G_hat = mpm(G_tensor, q, delta=DELTA_FOR_DIAG_LOADING)
    dt = time() - t0
    # print(f"time={dt}[sec]")

    scaler = np.linalg.norm(A[:,0])  # assuming all steering vectors have same norm
    A = A / scaler
    
    if USE_MVDR:
        G_hat_inv_A = np.linalg.solve(G_hat, A)
        p_vec = 1 / np.sum(A.conj() * G_hat_inv_A, axis=0).real
    else: # use Bartlett
        p_vec = np.sum(A.conj() * (G_hat @ A), axis=0).real

    # p_vec = np.maximum(p_vec - noise_power, 0)
    p_vec = p_vec / (scaler**2)

    eigsG = np.linalg.eigvalsh(G_hat)
    return p_vec, 0, 0, eigsG


def fun_MinSpectrum(Y, A, L, q, noise_power):
    G_tensor = get_G_tensor(Y, L)
    scaler = np.linalg.norm(A[:,0])  # assuming all steering vectors have same norm
    A = A / scaler
    p_vec = np.inf * np.ones(A.shape[1])
    for l in range(L):
        if USE_MVDR:
            G_hat_l_inv_A = np.linalg.solve(G_tensor[l,:,:] + DELTA_FOR_DIAG_LOADING * np.eye(A.shape[0]), A)
            p_vec_l = 1 / np.sum(A.conj() * G_hat_l_inv_A, axis=0).real
        else: # use Bartlett            
            p_vec_l = np.sum(A.conj() * (G_tensor[l,:,:] @ A), axis=0).real
        # element-wise minimum between p_vec and p_vec_l
        p_vec = np.minimum(p_vec, p_vec_l)
    p_vec = p_vec / (scaler**2)
    return p_vec, 0, 0, None