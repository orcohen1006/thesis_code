import numpy as np
from numpy.linalg import inv, slogdet
from scipy.sparse.linalg import cg


def jbld_cost(R, R_hat):
    sign1, logdet1 = slogdet(0.5 * (R + R_hat))
    sign2, logdet2_R = slogdet(R)
    sign3, logdet2_Rhat = slogdet(R_hat)
    return np.real(logdet1 - 0.5 * (logdet2_R + logdet2_Rhat))

def optimize_fixedpoint_JBLD(A, R_hat, sigma2, p_init, max_iter, do_store_history = False, do_verbose = False):
    """
    Minimize JBLD(R(p), R_hat) using fixed-point iterations:
        p_d^{new} = p_d * (0.5 * a_d^H R^{-1} a_d) / (a_d^H (R + R_hat)^{-1} a_d)
    """
    threshold = 1e-4
    eps_p = 1e-10
    M, D = A.shape
    p = np.maximum(p_init, eps_p)  # ensure positivity

    rel_change_history = []

    for it in range(max_iter):
        R = A @ np.diag(p) @ A.conj().T + sigma2 * np.eye(M)
        S = R + R_hat

        invR_A = np.linalg.solve(R, A)
        invS_A = np.linalg.solve(S, A)

        w = 0.5*np.maximum(0, np.real(np.sum(np.conj(A) * (invR_A), axis=0)))
        z = np.maximum(0, np.real(np.sum(np.conj(A) * (invS_A), axis=0)))
        
        nominator = w
        denominator = z
        denominator = np.maximum(denominator, 1e-20)
        # Fixed-point update
        p_new = p * nominator / denominator

        # Convergence check
        rel_change = np.linalg.norm(p_new - p) / np.linalg.norm(p)
        if do_verbose:
            print(f"Iter {it}: rel_change = {rel_change:.2e}")
        if do_store_history:
                rel_change_history.append(rel_change)
        if rel_change < threshold:
            break

        p = np.maximum(p_new, 0)  # enforce non-negativity

    return p,it,(None, rel_change_history)

def optimize_mm_JBLD(
    A, R_hat, sigma2, p_init, max_iter,
    cg_tol=1e-6,
    cg_maxiter=10,
    use_linesearch=False,
    do_store_history = False, do_verbose = False
):
    threshold = 1e-4
    eps_p = 1e-10
    M, D = A.shape
    p = np.maximum(p_init, eps_p)  # ensure positivity

    rel_change_history = []

    for it in range(max_iter):
        p_prev = p.copy() 
        
        R = A @ np.diag(p) @ A.conj().T + sigma2 * np.eye(M)
        S = R + R_hat
        invR_A = np.linalg.solve(R, A)
        invS_A = np.linalg.solve(S, A)

        # Gradient terms
        grad_f = np.maximum(0, np.real(np.sum(np.conj(A) * (invS_A), axis=0)))
        grad_g = -0.5*np.maximum(0, np.real(np.sum(np.conj(A) * (invR_A), axis=0)))
        # J(p | p_prev) = grad_f(p_prev)^T p + g(p), is the majorization function and is convex.
        # grad_J(p_prev) = grad_f(p_prev) + grad_g(p_prev)
        grad_J = grad_f + grad_g
        rhs = -grad_J  # Right-hand side for CG: (-1)*gradient

        # Precompute full Hessian: H = |A^H R^{-1} A|^2
        G = A.conj().T @ invR_A
        H = np.abs(G) ** 2

        # Solve H @ delta_p = rhs using CG
        delta_p, info = cg(H, rhs, tol=cg_tol, maxiter=cg_maxiter)
        if info != 0:
            print(f"[warning] CG did not fully converge at iter {it} (info = {info})")

        # Line search (optional)
        if use_linesearch:
            alpha = 1.0
            beta = 0.5
            c = 1e-4
            cost_old = jbld_cost(R, R_hat)

            while True:
                p_new = np.maximum(p - alpha * delta_p, 0)
                R_new =  A @ np.diag(p_new) @ A.conj().T + sigma2 * np.eye(M)
                cost_new = jbld_cost(R_new, R_hat)

                if cost_new <= cost_old - c * alpha * np.dot(delta_p, delta_p):
                    break

                alpha *= beta
                if alpha < 1e-6:
                    p_new = p  # Reject update
                    break

            p = p_new
        else:
            # Basic projected update
            p = np.maximum(p - delta_p, 0)

        # Convergence check
        rel_change = np.linalg.norm(p_prev - p) / np.linalg.norm(p_prev)
        if do_verbose:
            print(f"Iter {it}: rel_change = {rel_change:.2e}")
        if do_store_history:
                rel_change_history.append(rel_change)
        if rel_change < threshold:
            break

    return p
