# %%
import numpy as np
from time import time
import matplotlib.pyplot as plt
# %matplotlib ipympl

from RunSingleMCIteration import run_single_mc_iteration
from utils import *
import os
import pickle


# %%
import utils
import ToolsMC
import importlib
importlib.reload(utils)
importlib.reload(ToolsMC)
from utils import *
from ToolsMC import *
# %%
config = create_config(
                m=100, snr=0, N=50, 
                power_doa_db=np.array([0, 0, -5]),
                doa=np.array([35.25, 43.25, 51.25]), 
                cohr_flag=False,
                )
power_doa = convert_db_to_linear(config["power_doa_db"])
A_true = get_steering_matrix(config["doa"], config["m"])
noise_power = convert_db_to_linear(np.max(config["power_doa_db"]) - config["snr"])
# R = A_true @ np.diag(power_doa) @ A_true.conj().T + noise_power * np.eye(config["m"])

y_noisy = generate_signal(A_true, config["power_doa_db"], config["N"], noise_power, cohr_flag=config["cohr_flag"],
                              cohr_coeff=config["cohr_coeff"], noncircular_coeff=config["noncircular_coeff"],
                              seed=0)
R_hat = (y_noisy @ y_noisy.conj().T) / config["N"]

doa_scan = get_doa_grid()
A = get_steering_matrix(doa_scan, config["m"])


from OptimizeRiemannianLoss import *

def loss_AFFINV_original(p, A, pinv_sqrtm_R_hat, sigma2):
    M, D = A.shape
    P_diag = torch.diag(p).to(TORCH_DTYPE)  # Diagonal matrix from p

    # Compute Q
    Q = pinv_sqrtm_R_hat @ (A @ P_diag @ A.conj().T + sigma2 * torch.eye(M, dtype=TORCH_DTYPE)) @ pinv_sqrtm_R_hat

    # Eigenvalues of Q
    eigvals = torch.linalg.eigvalsh(Q).real  # Ensure real part (Q should be Hermitian)
    # print(eigvals)
    # Compute loss
    loss = torch.sum(torch.log(eigvals) ** 2)
    return loss

def loss_AFFINV_adapted(p, A, R_hat, sigma2):
    M, D = A.shape
    P_diag = torch.diag(p).to(TORCH_DTYPE)  # Diagonal matrix from p

    pinv_sqrtm_R_hat = matrix_pinv_sqrtm(R_hat)  # Compute R_hat^(-1/2)
    # Compute Q
    Q = pinv_sqrtm_R_hat @ (A @ P_diag @ A.conj().T + sigma2 * torch.eye(M, dtype=TORCH_DTYPE)) @ pinv_sqrtm_R_hat

    # Eigenvalues of Q
    eigvals = torch.linalg.eigvalsh(Q).real  # Ensure real part (Q should be Hermitian)
    # print(eigvals)
    # Compute loss
    loss = torch.sum(torch.log(eigvals) ** 2)
    return loss

def loss_LD_original(p, A, R_hat, sigma2):
    M, D = A.shape
    P_diag = torch.diag(p).to(TORCH_DTYPE)  # Diagonal matrix from p
    R = (A @ P_diag @ A.conj().T + sigma2 * torch.eye(M, dtype=TORCH_DTYPE))
    loss = torch.logdet(0.5*(R_hat + R)) -0.5*torch.logdet(R) # -0.5*torch.logdet(R_hat) is a constant and can be ignored
    loss = loss.real
    return loss



_p_init = np.sum(np.abs(A.conj().T @ (y_noisy / config["m"])), axis=1) / config["N"]
pinv_sqrtm_R_hat = matrix_pinv_sqrtm(torch.as_tensor(R_hat, dtype=TORCH_DTYPE))  # Compute R_hat^(-1/2)
p = torch.as_tensor(_p_init, dtype=torch.float).clone().detach().requires_grad_(True)  # Use provided initialization
A = torch.as_tensor(A, dtype=TORCH_DTYPE)
R_hat = torch.as_tensor(R_hat, dtype=TORCH_DTYPE)

optimizer = torch.optim.Adam([p], lr=1e-1)

# loss1 = loss_AFFINV(p, A, pinv_sqrtm_R_hat, noise_power)
# loss2 = loss_LD(p, A, R_hat, noise_power)

import time

num_runs = 1000  # number of repetitions for averaging

# Warm-up
loss = loss_AFFINV_original(p, A, pinv_sqrtm_R_hat, noise_power)
loss.backward()
p.grad = None

# --- Timing loss_AFFINV ---
t0 = time.perf_counter()
for _ in range(num_runs):
    p.grad = None  # clear gradients without touching optimizer
    loss1 = loss_AFFINV_adapted(p, A, pinv_sqrtm_R_hat, noise_power)
    loss1.backward()
t1 = time.perf_counter()

time_affinv = (t1 - t0) / num_runs

# --- Timing loss_LD ---
t0 = time.perf_counter()
for _ in range(num_runs):
    p.grad = None
    loss2 = loss_LD_original(p, A, R_hat, noise_power)
    loss2.backward()
t1 = time.perf_counter()

time_ld = (t1 - t0) / num_runs

print(f"loss_AFFINV grad time: {time_affinv*1e3:.3f} ms")
print(f"loss_LD     grad time: {time_ld*1e3:.3f} ms")








