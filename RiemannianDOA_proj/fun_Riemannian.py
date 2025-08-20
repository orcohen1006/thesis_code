import numpy as np
from scipy import linalg
from utils import *
from time import time
from OptimizeRiemannianLoss import optimize_adam_AIRM, optimize_adam_JBLD
from fun_JBLD import optimize_fixedpoint_JBLD, optimize_cg_JBLD
from fun_JBLD_advanced import optimize_JBLD_cccp

def fun_Riemannian(Y, A, DAS_init, DOAscan, DOA, noise_power, loss_name="AIRM"):
    t_samples = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / t_samples
    sigma2_n = noise_power


    if loss_name == "AIRM":
        p,num_iters, _ = optimize_adam_AIRM(A, R_hat, sigma2_n, DAS_init, _max_iter=int(2e3), _lr=1e-2)
    elif loss_name == "JBLD":
        p,num_iters, _ = optimize_adam_JBLD(A, R_hat, sigma2_n, DAS_init, _max_iter=int(2e3), _lr=1e-2)
        # p,num_iters, _ = optimize_JBLD_cccp(A, R_hat, sigma2_n, DAS_init, _max_iter=int(5e3), _lr=1e-2)

    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    return p, num_iters, sigma2_n
