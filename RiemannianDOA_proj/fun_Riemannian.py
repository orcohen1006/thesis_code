import numpy as np
from scipy import linalg
from utils import *
from time import time
from OptimizeRiemannianLoss import optimize_adam_AIRM, optimize_adam_JBLD, optimize_adam_LE
from fun_JBLD_advanced import *

def fun_Riemannian(Y, A, DAS_init, DOAscan, DOA, noise_power, loss_name="AIRM"):
    t_samples = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / t_samples
    # normalize by noise power and set sigma_n^2 to 1
    R_hat /= noise_power
    sigma2_n = 1.0

    do_store_convergence = globalParams.DO_STORE_CONVERGENCE

    if loss_name == "AIRM":
        p,num_iters, tuple_convergence = optimize_adam_AIRM(A, R_hat, sigma2_n, DAS_init, _max_iter=int(5e3), _lr=1e-2, do_store_convergence=do_store_convergence)
    elif loss_name == "JBLD":
        p,num_iters, tuple_convergence = optimize_adam_cholesky_JBLD(A, R_hat, sigma2_n, DAS_init, _max_iter=int(5e3), _lr=1e-2, do_store_convergence=do_store_convergence)
    elif loss_name == "LE":
        p,num_iters, tuple_convergence = optimize_adam_LE(A, R_hat, sigma2_n, DAS_init, _max_iter=int(5e3), _lr=1e-2, do_store_convergence=do_store_convergence)
    else:
        raise ValueError("loss_name not recognized in fun_Riemannian")
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    # rescale p 
    p *= noise_power

    return p, num_iters, tuple_convergence
