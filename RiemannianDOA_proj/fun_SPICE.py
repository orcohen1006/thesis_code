import numpy as np
from utils import *
from scipy import linalg
from time import time
from utils import EPS_REL_CHANGE


def fun_SPICE(Y, A, DAS_init, DOAscan, DOA, sigma_given=None):

    # t0 = time()

    sigma = sigma_given
    maxIter = 5000
    
    M, D = A.shape
    N = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / N
    
    # Initialize power vector
    p = np.abs(DAS_init) ** 2
    
    
    invR_hat_A = np.linalg.solve(R_hat, A)
    
    # Compute weight using conventional way
    weight_first = np.real(np.sum(A.conj() * invR_hat_A, axis=0)) / M
    weight_first = np.maximum(weight_first, 1e-30)
    weight = np.real(np.concatenate([weight_first, np.diag(np.linalg.inv(R_hat)).conj() / M]))
    gamma = np.real(np.mean(np.diag(np.linalg.inv(R_hat))))

    sqrt_weight = np.sqrt(weight[:D])
    sqrt_gamma = np.sqrt(gamma)
    R_hat_sqrt = linalg.sqrtm(R_hat)
    
    for jj in range(maxIter):
        print(f"SPICE iteration {jj}/{maxIter}", end='\r')
        p_prev = p.copy()
        
        # R = A @ np.diag(p) @ A.conj().T + sigma * np.eye(M)
        R = (A * p[np.newaxis, :]) @ A.conj().T + sigma * np.eye(M)

        Rinv_R_hat_sqrt = np.linalg.solve(R, R_hat_sqrt)
        am_Rinv_R_hat_sqrt = A.conj().T @ Rinv_R_hat_sqrt
        norm_am_Rinv_R_hat_sqrt = np.linalg.norm(am_Rinv_R_hat_sqrt, axis=1)
        # norm_Rinv_Rhatsqrt = np.sqrt(np.sum(np.sum(Rinv_R_hat_sqrt.conj() * Rinv_R_hat_sqrt, axis=0))) # more efficient calculation
        norm_Rinv_Rhatsqrt = np.linalg.norm(Rinv_R_hat_sqrt)


        ## OLD AND SLOWER:
        # Rinv = np.linalg.inv(R)
        # E_mat = Rinv @ R_hat @ Rinv.conj().T 
        # norm_Rinv_Rhatsqrt = np.sqrt(np.real(np.trace(E_mat)))
        # norm_am_Rinv_R_hat_sqrt = np.sqrt(np.diag(A.conj().T @ E_mat @ A))

        rho = np.sum(sqrt_weight.T * p * norm_am_Rinv_R_hat_sqrt)
        rho = rho + sqrt_gamma * sigma * norm_Rinv_Rhatsqrt
        
        # Compute p
        p = (p * norm_am_Rinv_R_hat_sqrt / (rho * sqrt_weight)).real

        
        measured_change_norm = np.linalg.norm(p - p_prev) / (1e-5 + np.linalg.norm(p_prev))
        if measured_change_norm < EPS_REL_CHANGE:
            break

    noisepower = sigma
    
    return p, jj, noisepower





# v2
def fun_SPICE_v2(Y, A, DAS_init, DOAscan, DOA, sigma_given=None):
    sigma2 = sigma_given
    max_iter = 5000
    
    M, D = A.shape
    N = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / N

    
    # 1. Pre-calculate the matrix square root of R_hat
    # Using eigh is often more stable for Hermitian positive semi-definite matrices
    evals, evecs = np.linalg.eigh(R_hat)
    evals = np.maximum(evals, 0) # Ensure no negative eigenvalues due to precision
    R_hat_half = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T
    
    # 2. Pre-calculate the norm of the steering vectors (denominator)
    a_norms = np.linalg.norm(A, axis=0) # Shape: (D,)
    
    # 3. Initialize p (using Periodogram/DAS estimate is a good starting point)
    # p = sum(abs(a^H R_hat a)) / ||a||^4
    p = np.real(np.diag(A.conj().T @ R_hat @ A)) / (a_norms**4)
    # p = np.abs(DAS_init) ** 2

    I_M = np.eye(M)
    
    for i in range(max_iter):
        print(f"SPICE iteration {i}/{max_iter}", end='\r')

        p_old = p.copy()
        
        # Construct modeled covariance R
        # R = A * Diag(p) * A^H + sigma2 * I
        R = (A * p) @ A.conj().T + sigma2 * I_M
        
        # Invert R (using solve for better numerical stability)
        # R_inv_R_hat_half = R^-1 * R_hat^1/2
        R_inv_R_hat_half = np.linalg.solve(R, R_hat_half)
        
        # Calculate Numerator: || a_k^H * R^-1 * R_hat^1/2 ||_2
        # We compute this for all k simultaneously
        temp = A.conj().T @ R_inv_R_hat_half # Shape: (D, M)
        numerator = np.linalg.norm(temp, axis=1) # Shape: (D,)
        # numerator = np.sum(np.abs(temp)**2, axis=1).real
        # Update power vector
        p = p_old * (numerator / a_norms)
        
        # Check convergence
        measured_change_norm = np.linalg.norm(p - p_old) / (1e-5 + np.linalg.norm(p_old))
        if measured_change_norm < EPS_REL_CHANGE:
            break
            
    return p, i, sigma2









def fun_SPICE_v3(Y, A, DAS_init, DOAscan, DOA, sigma_given=None):
    sigma2 = sigma_given
    max_iter = 5000
    
    M, D = A.shape
    N = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / N

    R_hat = np.asarray(R_hat)
    A = np.asarray(A)
    M, D = A.shape
    assert R_hat.shape == (M, M)

    # --- Precompute R_hat^{-1} and SPICE weights omega_k (constant) ---
    # Add a tiny ridge to stabilize inversion if needed.
    Rh = (R_hat + R_hat.conj().T) * 0.5 + 1e-12 * np.eye(M, dtype=R_hat.dtype)
    Rhat_inv = np.linalg.solve(Rh, np.eye(M, dtype=Rh.dtype))

    # omega_k = (a_k^H R_hat^{-1} a_k) / M  :contentReference[oaicite:4]{index=4}
    RhatInvA = Rhat_inv @ A
    omega = (np.sum(np.conj(A) * RhatInvA, axis=0).real) / M
    omega = np.maximum(omega, 1e-30)  # keep strictly positive for sqrt / division

    # gamma = sum noise-atom omegas; for uniform noise with I_M atoms:
    # omega_noise_m = (e_m^H R_hat^{-1} e_m)/M = diag(R_hat^{-1})/M, so gamma = tr(R_hat^{-1})/M. :contentReference[oaicite:5]{index=5}
    gamma = (np.trace(Rhat_inv).real) / M
    sqrt_omega = np.sqrt(omega)
    sqrt_gamma = np.sqrt(max(gamma, 0.0))

    # # --- Initialize p ---
    # if p0 is None:
    #     # Simple, robust-ish init: matched filter power proxy (nonnegative)
    #     # (You can swap this for DAS/Capon init if you prefer.)
    #     p = np.maximum((np.sum(np.conj(A) * (R_hat @ A), axis=0).real) / (np.sum(np.abs(A)**2, axis=0) + 1e-30), 0.0)
    #     p = p.astype(float, copy=False)
    # else:
    #     p = np.maximum(np.asarray(p0, dtype=float), 0.0)
    
    
    p = np.abs(DAS_init) ** 2

    # Enforce SPICE normalization: sum omega_k p_k + gamma*sigma2 = 1  :contentReference[oaicite:6]{index=6}
    denom = float(np.dot(omega, p) + gamma * sigma2)
    if denom <= 0:
        p = np.ones(D, dtype=float)
        denom = float(np.dot(omega, p) + gamma * sigma2)
    p *= (1.0 / denom)

    # --- Main loop ---
    I = np.eye(M, dtype=R_hat.dtype)
    for i in range(max_iter):
        p_prev = p.copy()

        # Build R(p) = A diag(p) A^H + sigma2 I
        # Efficient: A*(p) scales columns, then (A*p) @ A^H
        Ap = A * p[None, :]
        R = Ap @ A.conj().T + (sigma2 + 1e-12) * I
        R = (R + R.conj().T) * 0.5  # symmetrize

        # R^{-1}
        Rinv = np.linalg.solve(R, I)

        # Compute X = R^{-1} R_hat R^{-1}  (Hermitian PSD-ish)
        X = Rinv @ R_hat @ Rinv

        # For each k: num_k = sqrt( a_k^H X a_k )
        XA = X @ A
        akXak = np.sum(np.conj(A) * XA, axis=0).real
        akXak = np.maximum(akXak, 0.0)
        num = np.sqrt(akXak + 1e-30)

        # Frobenius term without sqrtm:
        # || R^{-1} R_hat^{1/2} ||_F^2 = tr( R_hat R^{-2} ) = tr(R_hat @ Rinv @ Rinv)
        fro2 = np.trace(R_hat @ (Rinv @ Rinv)).real
        fro = np.sqrt(max(fro2, 0.0) + 1e-30)

        # rho = sum_k sqrt(omega_k) p_k num_k + sqrt(gamma) * sigma2 * fro   :contentReference[oaicite:7]{index=7}
        rho = float(np.dot(sqrt_omega * p, num) + sqrt_gamma * sigma2 * fro)
        rho = max(rho, 1e-30)

        # Multiplicative update: p_k <- p_k * num_k / (sqrt(omega_k) * rho)  :contentReference[oaicite:8]{index=8}
        p *= (num / (sqrt_omega * rho))

        # Re-enforce constraint with fixed sigma2: sum omega_k p_k + gamma*sigma2 = 1  :contentReference[oaicite:9]{index=9}
        target = 1.0 - gamma * sigma2
        if target <= 0:
            # If gamma*sigma2 >= 1, the SPICE constraint is infeasible in this exact form.
            # Fall back to simple normalization of p (still keeps p finite).
            p /= max(np.dot(omega, p), 1e-30)
        else:
            scale = target / max(np.dot(omega, p), 1e-30)
            p *= scale

        # Convergence check (relative)
        measured_change_norm = np.linalg.norm(p - p_prev) / (1e-5 + np.linalg.norm(p_prev))
        if measured_change_norm < EPS_REL_CHANGE:
            break

    return p, i, sigma2



def fun_SPICE_chol(Y, A, DAS_init, DOAscan, DOA, sigma_given=None):

    # t0 = time()

    sigma = sigma_given
    maxIter = 5000
    
    M, D = A.shape
    N = Y.shape[1]
    R_hat = (Y @ Y.conj().T) / N
    
    # Initialize power vector
    p = np.abs(DAS_init) ** 2
    
    
    invR_hat_A = np.linalg.solve(R_hat, A)
    
    # Compute weight using conventional way
    weight_first = np.real(np.sum(A.conj() * invR_hat_A, axis=0)) / M
    weight_first = np.maximum(weight_first, 1e-30)
    weight = np.real(np.concatenate([weight_first, np.diag(np.linalg.inv(R_hat)).conj() / M]))
    gamma = np.real(np.mean(np.diag(np.linalg.inv(R_hat))))

    sqrt_weight = np.sqrt(weight[:D])
    sqrt_gamma = np.sqrt(gamma)
    R_hat_sqrt = linalg.sqrtm(R_hat)
    
    for jj in range(maxIter):
        print(f"SPICE iteration {jj}/{maxIter}", end='\r')
        p_prev = p.copy()
        
        R = (A * p[np.newaxis, :]) @ A.conj().T + sigma * np.eye(M)

        # 1. Use Cholesky Factorization for R
        # Since R is Positive Definite, Cholesky is 2x faster than linalg.solve
        # and much more stable than linalg.inv
        # lower=True is often faster due to memory layout
        L = linalg.cholesky(R, lower=True)

        # 2. Solve R * X = R_hat_sqrt efficiently using the Cholesky factor
        # This avoids the overhead of a full solve in every iteration
        X = linalg.cho_solve((L, True), R_hat_sqrt)

        # 3. Calculate norm_Rinv_Rhatsqrt (Frobenius norm)
        # This is much faster and memory-efficient than manual sum-squares
        norm_X = np.linalg.norm(X)

        # 4. Skip 'am_Rinv_R_hat_sqrt' intermediate matrix
        # Use the "Trace Trick" / Quadratic Form identity: 
        # The row-norm of (A.H @ X) is sqrt(diag(A.H @ X @ X.H @ A))
        # Precompute P_mat (MxM) which is smaller than am_X if A has many columns
        P_mat = X @ X.conj().T
        norm_am_sq = np.einsum('ji,jk,ki->i', A.conj(), P_mat, A).real
        norm_am = np.sqrt(norm_am_sq)

        # 5. Update rho and p (Vectorized)
        rho = np.dot(sqrt_weight.T * p, norm_am) + (sqrt_gamma * sigma * norm_X)
        p = (p * norm_am) / (rho * sqrt_weight)

        p = np.abs(p)
        
        measured_change_norm = np.linalg.norm(p - p_prev) / (1e-5 + np.linalg.norm(p_prev))
        if measured_change_norm < EPS_REL_CHANGE:
            break

    noisepower = sigma
    
    return p, jj, noisepower

