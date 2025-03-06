import torch
import numpy as np
from time import time

EPS_REL_CHANGE = 1e-4

# correction_matrix = None
# def get_fixed_noise(size, noise_scale=1e-7):
#     global correction_matrix
#     if correction_matrix is None:
#         noise = (torch.randn(size, size) + 1j*torch.randn(size, size))/np.sqrt(2.0)
#         correction_matrix = noise_scale * (noise + noise.T)
#     return correction_matrix


# import scipy.linalg
# def adjoint(A, E, f): # https://stackoverflow.com/questions/73288332/is-there-a-way-to-compute-the-matrix-logarithm-of-a-pytorch-tensor
#     A_H = A.T.conj().to(E.dtype)
#     n = A.size(0)
#     M = torch.zeros(2*n, 2*n, dtype=E.dtype, device=E.device)
#     M[:n, :n] = A_H
#     M[n:, n:] = A_H
#     M[:n, n:] = E
#     return f(M)[:n, n:].to(A.dtype)
#
# def logm_scipy(A):
#     return torch.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)
#
# class Logm(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, A):
#         assert A.ndim == 2 and A.size(0) == A.size(1)  # Square matrix
#         assert A.dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128)
#         ctx.save_for_backward(A)
#         return logm_scipy(A)
#
#     @staticmethod
#     def backward(ctx, G):
#         A, = ctx.saved_tensors
#         return adjoint(A, G, logm_scipy)
#
# matrix_logm = Logm.apply

def matrix_logm(B_in):
    # B_in += get_fixed_noise(size=B_in.shape[0], noise_scale=1e-7)
    eigvals, eigvecs = torch.linalg.eigh(B_in)
    # print(torch.min(torch.abs(eigvals[:,None] - eigvals[None,:]).masked_fill((torch.eye(len(eigvals), dtype=bool)), float('inf'))))
    eigvals_new = torch.log(torch.clamp(eigvals.real, min=1e-10))
    Lam_new = torch.diag(eigvals_new).type(torch.complex64)
    B_out = eigvecs @ Lam_new @ eigvecs.conj().T
    return B_out


def matrix_pinv_sqrtm(B_in):
    """Compute the inverse square root of a positive definite matrix B."""
    eigvals, eigvecs = torch.linalg.eigh(B_in)
    eigvals_new = 1.0 / torch.sqrt(torch.clamp(eigvals.real, min=1e-10))
    eigvals_new[eigvals.real < 1e-10] = 0
    Lam_new = torch.diag(eigvals_new).type(torch.complex64)
    B_out = eigvecs @ Lam_new @ eigvecs.conj().T
    return B_out




def loss_AFFINV(p, A, pinv_sqrtm_R_hat, sigma2):
    M, D = A.shape
    P_diag = torch.diag(p).to(torch.complex64)  # Diagonal matrix from p

    # Compute Q
    Q = pinv_sqrtm_R_hat @ (A @ P_diag @ A.conj().T + sigma2 * torch.eye(M, dtype=torch.complex64)) @ pinv_sqrtm_R_hat

    # Eigenvalues of Q
    eigvals = torch.linalg.eigvalsh(Q).real  # Ensure real part (Q should be Hermitian)
    # print(eigvals)
    # Compute loss
    loss = torch.sum(torch.log(eigvals) ** 2)
    return loss

def loss_LE(p, A, logm_R_hat, sigma2):
    M, D = A.shape
    P_diag = torch.diag(p).to(torch.complex64)  # Diagonal matrix from p

    logm_R = matrix_logm((A @ P_diag @ A.conj().T + sigma2 * torch.eye(M, dtype=torch.complex64)))
    diff_matrix = logm_R - logm_R_hat
    # loss = torch.linalg.matrix_norm(diff_matrix, ord='fro') ** 2
    loss = torch.sum(diff_matrix * diff_matrix).real
    return loss


def loss_LD(p, A, R_hat, sigma2):
    M, D = A.shape
    P_diag = torch.diag(p).to(torch.complex64)  # Diagonal matrix from p
    R = (A @ P_diag @ A.conj().T + sigma2 * torch.eye(M, dtype=torch.complex64))
    loss = torch.logdet(0.5*(R_hat + R)) -0.5*torch.logdet(R_hat @ R)
    loss = loss.real
    return loss

def optimize_adam_AFFINV(_A, _R_hat, _sigma2, _p_init, _max_iter=100, _lr=0.01, do_store_history = False, do_verbose = False):
    t0 = time()
    pinv_sqrtm_R_hat = matrix_pinv_sqrtm(torch.as_tensor(_R_hat, dtype=torch.complex64))  # Compute R_hat^(-1/2)
    p = torch.as_tensor(_p_init, dtype=torch.float).clone().detach().requires_grad_(True)  # Use provided initialization
    A = torch.as_tensor(_A, dtype=torch.complex64)

    p_prev = p.clone().detach()
    optimizer = torch.optim.Adam([p], lr=_lr)
    loss_history = []
    rel_change_history = []

    for step in range(_max_iter):
        optimizer.zero_grad()
        loss = loss_AFFINV(p, A, pinv_sqrtm_R_hat, _sigma2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            p.clamp_(min=0)  # Keep p non-negative if needed

        with torch.no_grad():
            p_norm = torch.norm(p)
            rel_change = 0
            if p_norm > 0:
                rel_change = (torch.norm(p - p_prev) / p_norm).item()
            if do_store_history:
                rel_change_history.append(rel_change)
                loss_history.append(loss.item())

            p_prev = p.clone().detach()

        if do_verbose and step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item()}")
        if rel_change < EPS_REL_CHANGE:
            break
    print(f"affinv: #iters= {step}, time= {time() - t0} [sec]")
    return p.detach(), (loss_history, rel_change_history)



def optimize_adam_LE(_A, _R_hat, _sigma2, _p_init, _max_iter=100, _lr=0.01, do_store_history = False, do_verbose = False):
    t0 = time()
    logm_R_hat = matrix_logm(torch.as_tensor(_R_hat, dtype=torch.complex64)) # Compute log(R_hat)
    p = torch.as_tensor(_p_init, dtype=torch.float).clone().detach().requires_grad_(True)  # Use provided initialization
    A = torch.as_tensor(_A, dtype=torch.complex64)

    M = A.shape[0]
    tmp_matrix = torch.randn(M, M) + torch.randn(M, M)*1j
    correction_matrix = (tmp_matrix @ tmp_matrix.conj().T) * 1e-7

    p_prev = p.clone().detach()
    optimizer = torch.optim.Adam([p], lr=_lr)
    loss_history = []
    rel_change_history = []

    for step in range(_max_iter):
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
            p_norm = torch.norm(p)
            rel_change = 0
            if p_norm > 0:
                rel_change = (torch.norm(p - p_prev) / p_norm).item()
            if do_store_history:
                rel_change_history.append(rel_change)
                loss_history.append(loss.item())

            p_prev = p.clone().detach()

        if do_verbose and step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item()}")
        if rel_change < EPS_REL_CHANGE:
            break
    print(f"le: #iters= {step}, time= {time() - t0} [sec]")
    return p.detach(), (loss_history, rel_change_history)

def optimize_adam_LD(_A, _R_hat, _sigma2, _p_init, _max_iter=100, _lr=0.01, do_store_history = False, do_verbose = False):
    t0 = time()

    R_hat = torch.as_tensor(_R_hat, dtype=torch.complex64)
    p = torch.as_tensor(_p_init, dtype=torch.float).clone().detach().requires_grad_(True)  # Use provided initialization
    A = torch.as_tensor(_A, dtype=torch.complex64)

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
            p_norm = torch.norm(p)
            rel_change = 0
            if p_norm > 0:
                rel_change = (torch.norm(p - p_prev) / p_norm).item()
            if do_store_history:
                rel_change_history.append(rel_change)
                loss_history.append(loss.item())

            p_prev = p.clone().detach()
        if do_verbose and step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item()}")
        if rel_change < EPS_REL_CHANGE:
            break
    print(f"ld: #iters= {step}, time= {time() - t0} [sec]")
    return p.detach(), (loss_history, rel_change_history)
