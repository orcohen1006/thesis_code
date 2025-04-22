import numpy as np
from scipy import linalg
from utils import *
from time import time
from OptimizeRiemannianLoss import optimize_adam_AIRM, optimize_adam_JBLD


def fun_Riemannian(Y, A, DAS_init, DOAscan, DOA, noise_power, loss_name="AIRM"):
    t_samples = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / t_samples
    sigma2_n = noise_power


    if loss_name == "AIRM":
        p,num_iters, _ = optimize_adam_AIRM(A, R_hat, sigma2_n, DAS_init, _max_iter=int(5e3), _lr=1e-1)
    elif loss_name == "JBLD":
        p,num_iters, _ = optimize_adam_JBLD(A, R_hat, sigma2_n, DAS_init, _max_iter=int(5e3), _lr=1e-1)

    # Detected_powers, Distance, normal = detect_DOAs(p, DOAscan, DOA)

    return p, num_iters, sigma2_n
