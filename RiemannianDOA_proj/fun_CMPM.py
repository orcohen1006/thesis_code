import numpy as np
from utils import *
from mpm import *
from time import time

def fun_CMPM(Y, A, L, q, noise_power):


    G_tensor = get_G_tensor(Y, L)
    t0 = time()
    G_hat = mpm(G_tensor, q, delta=globalParams.DELTA_FOR_DIAG_LOADING)
    dt = time() - t0
    # print(f"time={dt}[sec]")

    scaler = np.linalg.norm(A[:,0])  # assuming all steering vectors have same norm
    A = A / scaler
    
    p_vec = CreateSpectrum(G_hat, A, globalParams.SPECTRUM_TYPE, globalParams.SPECTRUM_NORMALIZATION)

    p_vec = p_vec / (scaler**2)

    # p_vec = esprit(G_hat, 2)

    eigsG = np.linalg.eigvalsh(G_hat)
    return p_vec, 0, 0, eigsG


def fun_CMPM_qstar(Y, A, L):

    G_tensor = get_G_tensor(Y, L)

    M, N = Y.shape
    W = N / L
    G_bar = karcher_mean(G_tensor,  epsilon=1e-4, max_iter= 10, delta=globalParams.DELTA_FOR_DIAG_LOADING)
    dispersion = np.mean([riemann_dist2(G_bar, G_tensor[l,:,:]) for l in range(L)])
    normalized_dispersion = dispersion * (W / M**2)
    normalized_dispersion = np.sqrt(normalized_dispersion)
    normalized_dispersion_min = 1.0
    normalized_dispersion_max = 1.05
    normalized_dispersion_clipped = np.clip(normalized_dispersion, normalized_dispersion_min, normalized_dispersion_max)
    dtilde = (normalized_dispersion_clipped - normalized_dispersion_min) / (normalized_dispersion_max - normalized_dispersion_min)
    # qstar = 1 - 2*dtilde
    qstar = 1 - 2/(1 + np.exp(-10*(dtilde - 0.5)))

    t0 = time()
    G_hat = mpm(G_tensor, qstar, delta=globalParams.DELTA_FOR_DIAG_LOADING)
    dt = time() - t0
    # print(f"time={dt}[sec]")

    scaler = np.linalg.norm(A[:,0])  # assuming all steering vectors have same norm
    A = A / scaler
    
    p_vec = CreateSpectrum(G_hat, A, globalParams.SPECTRUM_TYPE, globalParams.SPECTRUM_NORMALIZATION)

    p_vec = p_vec / (scaler**2)

    # p_vec = esprit(G_hat, 2)

    return p_vec, qstar, 0, None


def fun_OptimalCMPM(Y, A, L, q_vals, noise_power):

    eps_eigval = 1e-10
    eps_q = 1e-10
    M, N = Y.shape
    G_tensor = get_G_tensor(Y, L)

    C_sum = np.zeros((M, M), dtype=complex)
    for q in q_vals:
        G_hat = mpm(G_tensor, q, delta=globalParams.DELTA_FOR_DIAG_LOADING)
        eigvals, eigvecs = np.linalg.eigh(G_hat)
        # sort eigenvalues in descending order and eigenvectors accordingly
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        num_sources = np.argmax(eigvals[:-1] / eigvals[1:]) + 1 # Scree plot (largest eigenvalue ratio)
        # num_sources = 2
        source_subspace = eigvecs[:, :num_sources]
        C_sum += source_subspace @ source_subspace.conj().T
    G_hat = C_sum / len(q_vals)



    scaler = np.linalg.norm(A[:,0])  # assuming all steering vectors have same norm
    A = A / scaler
    
    p_vec = CreateSpectrum(G_hat, A, globalParams.SPECTRUM_TYPE, globalParams.SPECTRUM_NORMALIZATION)

    # p_vec = np.maximum(p_vec - noise_power, 0)
    p_vec = p_vec / (scaler**2)

    eigsG = np.linalg.eigvalsh(G_hat)
    return p_vec, 0, 0, eigsG




def fun_MinSpectrum(Y, A, L, q, noise_power):
    G_tensor = get_G_tensor(Y, L)
    scaler = np.linalg.norm(A[:,0])  # assuming all steering vectors have same norm
    A = A / scaler
    p_vec = np.inf * np.ones(A.shape[1])
    for l in range(L):
        p_vec_l = CreateSpectrum(G_tensor[l,:,:] , A, globalParams.SPECTRUM_TYPE, globalParams.SPECTRUM_NORMALIZATION)
        # element-wise minimum between p_vec and p_vec_l
        p_vec = np.minimum(p_vec, p_vec_l)
    p_vec = p_vec / (scaler**2)
    return p_vec, 0, 0, None





def fun_ProjectOutInterf(Y, A, L, q, noise_power):
    G_tensor = get_G_tensor(Y, L)
    scaler = np.linalg.norm(A[:,0])  # assuming all steering vectors have same norm
    A = A / scaler

    G_hat = covariance_of_ProjectOutInterf(G_tensor)
    p_vec = CreateSpectrum(G_hat, A, globalParams.SPECTRUM_TYPE, globalParams.SPECTRUM_NORMALIZATION)

    p_vec = p_vec / (scaler**2)

    # p_vec = esprit(G_hat, 2)

    return p_vec, 0, 0, None

def covariance_of_ProjectOutInterf(
    G_tensor: np.ndarray,
    n_interf_per_segment: int | list[int] = 1,
) -> np.ndarray:
    """
    Interference subspace projection baseline covariance estimator.

    For each segment covariance, estimates the interference subspace from
    its dominant eigenvectors, projects it out, then returns the arithmetic
    mean of the projected (cleaned) covariances.

    This is a natural practitioner heuristic: it uses all segments and
    directly targets per-segment interference, but requires knowing (or
    estimating) the number of interferers per segment. It makes no use of
    Riemannian geometry and provides no theoretical monotonicity guarantees.

    Parameters
    ----------
    G_tensor : ndarray, shape (L, M, M)
        Segment sample covariance matrices. Must be HPD.
    n_interf_per_segment : int or list of int
        Number of interference sources (dominant eigenvectors to remove)
        per segment. If int, the same value is used for all segments.
        If list, must have length L.

    Returns
    -------
    C_baseline : ndarray, shape (M, M)
        Arithmetic mean of the interference-projected segment covariances.
        Hermitian and positive semi-definite (positive definite if
        n_interf_per_segment < M for all segments).

    Notes
    -----
    The projection for segment ell is:
        P_ell = I - U_ell @ U_ell^H
    where U_ell in C^{M x k_ell} holds the k_ell dominant eigenvectors
    of R_hat_ell. The cleaned covariance is:
        R_clean_ell = P_ell @ R_hat_ell @ P_ell^H
    The output is (1/L) * sum_ell R_clean_ell.

    Assumption cost vs. MPM: this method requires knowing n_interf_per_segment,
    whereas MPM requires only q. For a fair experimental comparison, either
    use oracle knowledge of interference count (upper-bounding this baseline's
    performance) or estimate it via an MDL/AIC model-order selector.
    """
    L, M, M2 = G_tensor.shape
    assert M == M2, "Segment covariances must be square."

    # Resolve per-segment interference counts
    if isinstance(n_interf_per_segment, int):
        k_list = [n_interf_per_segment] * L
    else:
        k_list = list(n_interf_per_segment)
    assert len(k_list) == L, "n_interf_per_segment list length must equal L."
    assert all(0 <= k < M for k in k_list), (
        "Each n_interf_per_segment must be in [0, M)."
    )

    C_sum = np.zeros((M, M), dtype=complex)

    for ell in range(L):
        R = G_tensor[ell]  # (M, M), HPD
        k = k_list[ell]

        if k == 0:
            # No projection for this segment
            C_sum += R
            continue

        # Eigendecomposition — use eigh for guaranteed real eigenvalues
        # on Hermitian input; eigenvalues returned in ascending order.
        eigenvalues, eigenvectors = np.linalg.eigh(R)

        # Dominant k eigenvectors = interference subspace estimate
        # eigh returns ascending order, so dominant are the last k columns.
        U_interf = eigenvectors[:, -k:]  # (M, k)

        # Orthogonal projector onto the interference subspace complement
        P_orth = np.eye(M, dtype=complex) - U_interf @ U_interf.conj().T  # (M, M)

        # Project out interference subspace
        R_clean = P_orth @ R @ P_orth.conj().T  # (M, M)

        C_sum += R_clean

    C_baseline = C_sum / L
    return C_baseline