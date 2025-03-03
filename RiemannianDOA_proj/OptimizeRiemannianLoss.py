import torch
import numpy as np

def matrix_pinv_sqrtm(B_in):
    """Compute the inverse square root of a positive definite matrix B."""
    eigvals, eigvecs = torch.linalg.eigh(B_in)
    eigvals_new = 1.0 / torch.sqrt(torch.clamp(eigvals.real, min=1e-10))
    eigvals_new[eigvals.real < 1e-10] = 0
    Lam_new = torch.diag(eigvals_new).type(torch.complex64)
    B_out = eigvecs @ Lam_new @ eigvecs.conj().T
    return B_out

def matrix_logm(B_in):
    eigvals, eigvecs = torch.linalg.eigh(B_in)
    eigvals = torch.clamp(eigvals.real, min=1e-10)
    Lam_new = torch.diag(torch.log(eigvals)).type(torch.complex64)
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

    loss = torch.linalg.matrix_norm(logm_R - logm_R_hat, ord='fro') ** 2
    return loss


def optimize_lbfgs_AFFINV(_A, _R_hat, _sigma2, _p_init, _max_iter=100):
    pinv_sqrtm_R_hat = matrix_pinv_sqrtm(torch.as_tensor(_R_hat, dtype=torch.complex64))  # Compute R_hat^(-1/2)
    p = torch.as_tensor(_p_init,dtype=torch.float).clone().detach().requires_grad_(True)  # Use provided initialization
    A = torch.as_tensor(_A,dtype=torch.complex64)
    optimizer = torch.optim.LBFGS([p], max_iter=_max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = loss_AFFINV(p, A, pinv_sqrtm_R_hat, _sigma2)
        loss.backward()
        return loss

    optimizer.step(closure)

    return p.detach().numpy()


def optimize_adam_AFFINV(_A, _R_hat, _sigma2, _p_init, _max_iter=100, _lr=0.01, do_store_history = False, do_verbose = False):
    pinv_sqrtm_R_hat = matrix_pinv_sqrtm(torch.as_tensor(_R_hat, dtype=torch.complex64))  # Compute R_hat^(-1/2)
    p = torch.as_tensor(_p_init, dtype=torch.float).clone().detach().requires_grad_(True)  # Use provided initialization
    A = torch.as_tensor(_A, dtype=torch.complex64)

    optimizer = torch.optim.Adam([p], lr=_lr)
    loss_history = []

    for step in range(_max_iter):
        optimizer.zero_grad()
        loss = loss_AFFINV(p, A, pinv_sqrtm_R_hat, _sigma2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            p.clamp_(min=0)  # Keep p non-negative if needed


        if do_store_history: # Store loss
            loss_history.append(loss.item())

        if do_verbose and step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item()}")

    return p.detach(), loss_history


# # Example usage
# M, D = 4, 3
# A = torch.randn(M, D, dtype=torch.complex64)
# B = torch.eye(M, dtype=torch.complex64)
# sigma2 = 0.1
# p_init = torch.rand(D)  # Smart initialization
#
# optimized_p = optimize_p_lbfgs(A, B, sigma2, p_init)
# print("Optimized p:", optimized_p)
