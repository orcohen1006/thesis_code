# %%
import torch
import numpy as np
from time import time
# %%
TORCH_DTYPE = torch.complex64
from utils import EPS_REL_CHANGE

def matrix_logm(B_in):
    dim = B_in.shape[-1]
    min_allowed_eig_diff = 1e-2
    ramp = torch.arange(dim, dtype=torch.float32) * min_allowed_eig_diff
    perturbation = torch.diag(ramp).to(TORCH_DTYPE)
    B_perturbed = B_in + perturbation

    eigvals, eigvecs = torch.linalg.eigh(B_perturbed)
    # with torch.no_grad():
    #     min_eig_diff = torch.min(torch.abs(eigvals[:,None] - eigvals[None,:]).masked_fill((torch.eye(len(eigvals), dtype=bool)), float('inf'))).item()
    #     print(f"min_eig_diff: {min_eig_diff}, min_eig: {torch.min(eigvals).item()}")
    eigvals_new = torch.log(torch.clamp(eigvals.real, min=1e-10))
    Lam_new = torch.diag(eigvals_new).type(TORCH_DTYPE)
    B_out = eigvecs @ Lam_new @ eigvecs.conj().T
    return B_out


def matrix_pinv_sqrtm(B_in):
    """Compute the inverse square root of a positive definite matrix B."""
    eigvals, eigvecs = torch.linalg.eigh(B_in)
    eigvals_new = 1.0 / torch.sqrt(torch.clamp(eigvals.real, min=1e-10))
    eigvals_new[eigvals.real < 1e-10] = 0
    Lam_new = torch.diag(eigvals_new).type(TORCH_DTYPE)
    B_out = eigvecs @ Lam_new @ eigvecs.conj().T
    return B_out




def loss_AFFINV(p, A, pinv_sqrtm_R_hat, sigma2):
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

def loss_LE(p, A, logm_R_hat, sigma2):
    M, D = A.shape
    P_diag = torch.diag(p).to(TORCH_DTYPE)  # Diagonal matrix from p

    logm_R = matrix_logm((A @ P_diag @ A.conj().T + sigma2 * torch.eye(M, dtype=TORCH_DTYPE)))
    diff_matrix = logm_R - logm_R_hat
    loss = torch.linalg.matrix_norm(diff_matrix, ord='fro') ** 2
    # loss = torch.sum(diff_matrix * diff_matrix).real
    return loss


def loss_LD(p, A, R_hat, sigma2):
    M, D = A.shape
    P_diag = torch.diag(p).to(TORCH_DTYPE)  # Diagonal matrix from p
    R = (A @ P_diag @ A.conj().T + sigma2 * torch.eye(M, dtype=TORCH_DTYPE))
    loss = torch.logdet(0.5*(R_hat + R)) -0.5*torch.logdet(R) # -0.5*torch.logdet(R_hat) is a constant and can be ignored
    loss = loss.real
    return loss

def optimize_adam_AIRM(_A, _R_hat, _sigma2, _p_init, _max_iter=100, _lr=0.01, do_store_history = False, do_verbose = False):
    # t0 = time()
    pinv_sqrtm_R_hat = matrix_pinv_sqrtm(torch.as_tensor(_R_hat, dtype=TORCH_DTYPE))  # Compute R_hat^(-1/2)
    p = torch.as_tensor(_p_init, dtype=torch.float).clone().detach().requires_grad_(True)  # Use provided initialization
    A = torch.as_tensor(_A, dtype=TORCH_DTYPE)

    p_prev = p.clone().detach()
    optimizer = torch.optim.Adam([p], lr=_lr)
    loss_history = []
    rel_change_history = []

    for step in range(_max_iter):
        print(f"AIRM iteration {step}/{_max_iter}", end='\r')
        optimizer.zero_grad()
        loss = loss_AFFINV(p, A, pinv_sqrtm_R_hat, _sigma2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            p.clamp_(min=0)  # Keep p non-negative if needed

        with torch.no_grad():
            rel_change = (torch.norm(p - p_prev) / (1e-5 + torch.norm(p_prev))).item()
            if do_store_history:
                rel_change_history.append(rel_change)
                loss_history.append(loss.item())

            p_prev = p.clone().detach()

        if do_verbose and step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item()}")
        if rel_change < EPS_REL_CHANGE:
            break
    # print(f"affinv: #iters= {step}, time= {time() - t0} [sec]")
    return p.detach(), step, (loss_history, rel_change_history)



def optimize_adam_LE(_A, _R_hat, _sigma2, _p_init, _max_iter=100, _lr=0.01, do_store_history = False, do_verbose = False):
    # t0 = time()
    logm_R_hat = matrix_logm(torch.as_tensor(_R_hat, dtype=TORCH_DTYPE)) # Compute log(R_hat)
    p = torch.as_tensor(_p_init, dtype=torch.float).clone().detach().requires_grad_(True)  # Use provided initialization
    A = torch.as_tensor(_A, dtype=TORCH_DTYPE)

    M = A.shape[0]
    p_prev = p.clone().detach()
    optimizer = torch.optim.Adam([p], lr=_lr)
    loss_history = []
    rel_change_history = []

    for step in range(_max_iter):
        print(f"LE iteration {step}/{_max_iter}", end='\r')
        optimizer.zero_grad()
        loss = loss_LE(p, A, logm_R_hat, _sigma2)
        # with torch.autograd.detect_anomaly():
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            p.clamp_(min=0)  # Keep p non-negative if needed

        # if not torch.isfinite(p).all():
        #     print(f"Parameter NaN/Inf detected")

        with torch.no_grad():
            rel_change = (torch.norm(p - p_prev) / (1e-5 + torch.norm(p_prev))).item()
            if do_store_history:
                rel_change_history.append(rel_change)
                loss_history.append(loss.item())

            p_prev = p.clone().detach()

        if do_verbose and step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item()}")
        if rel_change < EPS_REL_CHANGE:
            break
    # print(f"le: #iters= {step}, time= {time() - t0} [sec]")
    return p.detach(), step, (loss_history, rel_change_history)

def optimize_adam_JBLD(_A, _R_hat, _sigma2, _p_init, _max_iter=100, _lr=0.01, do_store_history = False, do_verbose = False):
    # t0 = time()

    R_hat = torch.as_tensor(_R_hat, dtype=TORCH_DTYPE)
    p = torch.as_tensor(_p_init, dtype=torch.float).clone().detach().requires_grad_(True)  # Use provided initialization
    A = torch.as_tensor(_A, dtype=TORCH_DTYPE)

    p_prev = p.clone().detach()
    optimizer = torch.optim.Adam([p], lr=_lr)
    loss_history = []
    rel_change_history = []

    for step in range(_max_iter):
        optimizer.zero_grad()
        loss = loss_LD(p, A, R_hat, _sigma2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            p.clamp_(min=0)  # Keep p non-negative if needed

        with torch.no_grad():
            rel_change = (torch.norm(p - p_prev) / (1e-5 + torch.norm(p_prev))).item()
            if do_store_history:
                rel_change_history.append(rel_change)
                loss_history.append(loss.item())

            p_prev = p.clone().detach()
        if do_verbose and step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item()}")
        if rel_change < EPS_REL_CHANGE:
            break
    # print(f"ld: #iters= {step}, time= {time() - t0} [sec]")
    return p.detach(), step, (loss_history, rel_change_history)
