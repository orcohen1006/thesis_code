import torch
import numpy as np
# import scipy

def matrix_sqrt_inv(B):
    """Compute the inverse square root of a positive definite matrix B."""
    eigvals, eigvecs = torch.linalg.eigh(B)
    sqrt_inv = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)).type(torch.complex64) @ eigvecs.T
    return sqrt_inv


def loss_function(p, A, B_sqrt_inv, sigma2):
    """
    Compute the loss function:
    loss(p) = sum_{i=1}^M (log(lambda_i))^2
    where lambda_i are eigenvalues of Q.
    """
    M, D = A.shape
    P_diag = torch.diag(p).to(torch.complex64)  # Diagonal matrix from p

    # Compute Q
    Q = B_sqrt_inv @ (A @ P_diag @ A.conj().T + sigma2 * torch.eye(M, dtype=torch.complex64)) @ B_sqrt_inv

    # Eigenvalues of Q
    eigvals = torch.linalg.eigvalsh(Q).real  # Ensure real part (Q should be Hermitian)

    # Compute loss
    loss = torch.sum(torch.log(eigvals) ** 2)
    return loss


def optimize_p_lbfgs(A, B, sigma2, p_init, max_iter=100):
    """
    Optimize p using L-BFGS to minimize the loss function.

    Args:
        A (torch.Tensor): M x D complex matrix.
        B (torch.Tensor): M x M complex matrix.
        sigma2 (float): Scalar noise parameter.
        p_init (torch.Tensor): Initial D-dimensional real vector.
        max_iter (int): Number of optimization steps.

    Returns:
        p (torch.Tensor): Optimized D-dimensional real vector.
    """
    B_sqrt_inv = matrix_sqrt_inv(B)  # Compute B^(-1/2)
    p = p_init.clone().detach().requires_grad_(True)  # Use provided initialization

    optimizer = torch.optim.LBFGS([p], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = loss_function(p, A, B_sqrt_inv, sigma2)
        loss.backward()
        return loss

    optimizer.step(closure)

    return p.detach()


# Example usage
M, D = 4, 3
A = torch.randn(M, D, dtype=torch.complex64)
B = torch.eye(M, dtype=torch.complex64)
sigma2 = 0.1
p_init = torch.rand(D)  # Smart initialization

optimized_p = optimize_p_lbfgs(A, B, sigma2, p_init)
print("Optimized p:", optimized_p)
