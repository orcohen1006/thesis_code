import numpy as np

# ------------------------- Helpers (module scope) -------------------------

_EIG_FLOOR_DEFAULT = 1e-14

def herm(A: np.ndarray) -> np.ndarray:
    """Hermitian symmetrization."""
    return (A + A.conj().T) * 0.5

def eigh_clip(A: np.ndarray, eig_floor: float = _EIG_FLOOR_DEFAULT):
    """Eigen-decomposition of Hermitian matrix with eigenvalue floor."""
    w, V = np.linalg.eigh(herm(A))
    w = np.maximum(w.real, eig_floor)
    return w, V

def powm_herm(A: np.ndarray, p: float, eig_floor: float = _EIG_FLOOR_DEFAULT) -> np.ndarray:
    """Hermitian PD matrix power A^p via eigen-decomposition."""
    w, V = eigh_clip(A, eig_floor=eig_floor)
    wp = w ** p
    return (V * wp) @ V.conj().T

def logm_herm(A: np.ndarray, eig_floor: float = _EIG_FLOOR_DEFAULT) -> np.ndarray:
    """Hermitian PD matrix log via eigen-decomposition."""
    w, V = eigh_clip(A, eig_floor=eig_floor)
    lw = np.log(w)
    return (V * lw) @ V.conj().T

def expm_herm(A: np.ndarray) -> np.ndarray:
    """Hermitian matrix exp via eigen-decomposition (returns HPD)."""
    w, V = np.linalg.eigh(herm(A))
    ew = np.exp(w.real)
    return (V * ew) @ V.conj().T

def inv_herm_pd(A: np.ndarray) -> np.ndarray:
    """Inverse of HPD matrix via linear solve (more stable than explicit inv)."""
    I = np.eye(A.shape[0], dtype=A.dtype)
    return np.linalg.solve(A, I)

def sqrt_invsqrt_from_eigh(w: np.ndarray, V: np.ndarray):
    """Given eigendecomp with w>0, return X^{1/2} and X^{-1/2}."""
    sqrtw = np.sqrt(w)
    invsqrtw = 1.0 / sqrtw
    X_sqrt = (V * sqrtw) @ V.conj().T
    X_invsqrt = (V * invsqrtw) @ V.conj().T
    return X_sqrt, X_invsqrt


# ------------------------- Karcher mean (q=0) -------------------------

def karcher_mean(G_tensor: np.ndarray,
                 epsilon: float = 1e-6,
                 max_iter: int = 200,
                 delta: float = 0.0,
                 eig_floor: float = _EIG_FLOOR_DEFAULT) -> np.ndarray:
    """
    AIRM (Karcher) mean iteration:
      X_{k+1} = X^{1/2} exp( (1/L) sum log(X^{-1/2} C_l X^{-1/2}) ) X^{1/2}
    with Log-Euclidean init: exp( (1/L) sum log(C_l) ).
    """
    if G_tensor.ndim != 3 or G_tensor.shape[1] != G_tensor.shape[2]:
        raise ValueError("G_tensor must have shape (L, M, M).")
    L, M, _ = G_tensor.shape
    dtype = G_tensor.dtype
    I = np.eye(M, dtype=dtype)

    # C_l = R_l + delta I
    C = np.empty_like(G_tensor, dtype=dtype)
    for ell in range(L):
        C[ell] = herm(G_tensor[ell] + delta * I)

    # Log-Euclidean initialization: X0 = exp( mean log(C_l) )
    mean_logC = np.zeros((M, M), dtype=dtype)
    for ell in range(L):
        mean_logC += logm_herm(C[ell], eig_floor=eig_floor)
    mean_logC /= float(L)
    X = herm(expm_herm(mean_logC))

    sqrtM = np.sqrt(float(M))

    for _k in range(max_iter):
        # X^(1/2) and X^(-1/2) from one eigendecomp
        wX, VX = eigh_clip(X, eig_floor=eig_floor)
        X_sqrt, X_invsqrt = sqrt_invsqrt_from_eigh(wX, VX)

        # Delta = (1/L) sum log( X^{-1/2} C_l X^{-1/2} )
        Delta = np.zeros((M, M), dtype=dtype)
        for ell in range(L):
            Y = herm(X_invsqrt @ C[ell] @ X_invsqrt)
            Delta += logm_herm(Y, eig_floor=eig_floor)
        Delta /= float(L)
        Delta = herm(Delta)

        # Stopping based on ||Delta||_F / sqrt(M)
        rel_change = np.linalg.norm(Delta, ord='fro') / sqrtM
        # print(f"q={0},iter={_k}: rel_change = {rel_change}")
        if  rel_change <= epsilon:
            break

        X = herm(X_sqrt @ expm_herm(Delta) @ X_sqrt)

    return X


# ------------------------- Matrix power mean (q != 0) -------------------------

def mpm(G_tensor: np.ndarray,
        q: float,
        epsilon: float = 1e-4,
        max_iter: int = 10,
        delta: float = 1e-3,
        q0_thresh: float = 1e-10,
        eig_floor: float = _EIG_FLOOR_DEFAULT) -> np.ndarray:
    """
    Matrix Power Mean (MPM) aggregate for HPD matrices.

    Args:
        G_tensor: ndarray (L, M, M), HPD matrices.
        q: power-mean parameter in [-1,1]. If abs(q) < q0_thresh -> Karcher mean.
        epsilon: stopping tolerance.
        max_iter: maximum iterations.
        delta: diagonal loading (primarily needed for q<0 or ill-conditioning).
        q0_thresh: threshold for treating q as 0 (Karcher mean).
        eig_floor: eigenvalue floor used in matrix functions for numerical stability.

    Returns:
        Gamma_q: ndarray (M, M), matrix power mean (Karcher mean if |q| small).
    """
    if G_tensor.ndim != 3 or G_tensor.shape[1] != G_tensor.shape[2]:
        raise ValueError("G_tensor must have shape (L, M, M).")
    L, M, _ = G_tensor.shape
    if L < 1:
        raise ValueError("L must be >= 1.")

    q = float(q)
    if abs(q) < q0_thresh:
        return karcher_mean(G_tensor, epsilon=epsilon, max_iter=max_iter, delta=delta, eig_floor=eig_floor)

    alpha = abs(q)
    dtype = G_tensor.dtype
    I = np.eye(M, dtype=dtype)

    C = np.empty_like(G_tensor, dtype=dtype)
    # Build C_ell using inverse-reduction for q<0
    if q < 0.0:
        # C_ell = (R_ell + delta I)^(-1)
        for ell in range(L):
            C[ell] = herm(inv_herm_pd(herm(G_tensor[ell] + delta * I)))
    else:
        # For q>0,  no diagonal loading.
        for ell in range(L):
            C[ell] = herm(G_tensor[ell])

    # Initialization: X0 = ( mean C_ell^alpha )^(1/alpha)
    mean_Ca = np.zeros((M, M), dtype=dtype)
    for ell in range(L):
        mean_Ca += powm_herm(C[ell], alpha, eig_floor=eig_floor)
    mean_Ca /= float(L)
    X = herm(powm_herm(mean_Ca, 1.0 / alpha, eig_floor=eig_floor))

    # Fixed-point iterations
    for _k in range(max_iter):
        # X^(1/2) and X^(-1/2) from one eigendecomp
        wX, VX = eigh_clip(X, eig_floor=eig_floor)
        X_sqrt, X_invsqrt = sqrt_invsqrt_from_eigh(wX, VX)

        # S = (1/L) sum ( X^{-1/2} C_ell X^{-1/2} )^alpha
        S = np.zeros((M, M), dtype=dtype)
        for ell in range(L):
            Y = herm(X_invsqrt @ C[ell] @ X_invsqrt)
            S += powm_herm(Y, alpha, eig_floor=eig_floor)
        S /= float(L)

        X_next = herm(X_sqrt @ S @ X_sqrt)

        denom = np.linalg.norm(X, ord='fro') + 1e-30
        rel_change = np.linalg.norm(X_next - X, ord='fro') / denom
        # print(f"q={q},iter={_k}: rel_change = {rel_change}")
        if  rel_change <= epsilon:
            X = X_next
            break
        X = X_next

    # inverse-reduction output
    if q < 0.0:
        return herm(inv_herm_pd(X))
    return X
