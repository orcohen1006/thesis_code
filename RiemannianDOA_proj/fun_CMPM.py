import numpy as np
from utils import *


def fun_CMPM(Y, A, L, q, noise_power):

    eps_eigval = 1e-10
    eps_q = 1e-10
    M, N = Y.shape
    G_tensor = get_G_tensor(Y, L)

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
    scaler = np.linalg.norm(A[:,0])  # assuming all steering vectors have same norm
    A = A / scaler
    p_vec = np.sum(A.conj() * (G_hat @ A), axis=0).real
    # p_vec = np.maximum(p_vec - noise_power, 0)
    p_vec = p_vec / (scaler**2)

    eigsG = np.linalg.eigvalsh(G_hat)
    return p_vec, 0, 0, eigsG
