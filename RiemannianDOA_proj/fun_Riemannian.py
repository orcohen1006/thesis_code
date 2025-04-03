import numpy as np
from scipy import linalg
from utils import *
from time import time
from OptimizeRiemannianLoss import optimize_adam_AFFINV, optimize_adam_LD


def fun_Riemannian(Y, A, DAS_init, DOAscan, DOA, noise_power, loss_name="AFFINV"):
    t_samples = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / t_samples
    sigma2_n = noise_power


    if loss_name == "AIRM":
        p,_ = optimize_adam_AFFINV(A, R_hat, sigma2_n, DAS_init, _max_iter=int(5e3), _lr=1e-1)
    elif loss_name == "JBLD":
        p,_ = optimize_adam_LD(A, R_hat, sigma2_n, DAS_init, _max_iter=int(5e3), _lr=1e-1)

    Detected_powers, Distance, normal = detect_DOAs(p, DOAscan, DOA)


    noisepower = sigma2_n
    return Detected_powers, Distance, p, normal, noisepower
