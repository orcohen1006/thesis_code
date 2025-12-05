import numpy as np
from utils import *

# def fun_ESPRIT(Y, A, DAS_init, DOAscan, DOA):

def fun_ESPRIT(Y, num_sources):
    """
    ESPRIT DOA estimator for a ULA with steering vectors:
        a_m(theta) = exp(1j * pi * m * cos(theta)),  m = 0,...,M-1

    Parameters
    ----------
    Y : ndarray of shape (M, N)
        Array data: M sensors, N snapshots (complex-valued).
    num_sources : int
        Number of sources (model order).

    Returns
    -------
    doa_rad : ndarray of shape (num_sources,)
        Estimated DOAs in radians, consistent with
        a(theta) = exp(1j * pi * m * cos(theta)).
    """
    M, N = Y.shape

    # 1) Signal subspace from SVD of Y
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    Us = U[:, :num_sources]     # (M x num_sources)

    # 2) Selection matrices for shift invariance
    #    J1 picks sensors 0..M-2, J2 picks sensors 1..M-1
    J1 = np.eye(M - 1, M, k=0)  # (M-1 x M)
    J2 = np.eye(M - 1, M, k=1)  # (M-1 x M)

    Us1 = J1 @ Us               # (M-1 x num_sources)
    Us2 = J2 @ Us               # (M-1 x num_sources)

    # 3) Solve Us2 ≈ Us1 * Psi  (least-squares)
    Psi = np.linalg.pinv(Us1) @ Us2   # (num_sources x num_sources)

    # 4) Eigen-decomposition of Psi
    eigvals, eigvecs = np.linalg.eig(Psi)

    # 5) Map eigenvalues to DOAs using your steering convention:
    #    eigvals ≈ exp(1j * pi * cos(theta))
    phi = np.angle(eigvals)          # in (-pi, pi]
    cos_theta = phi / np.pi
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety

    doa_deg = np.arccos(cos_theta) * 180.0 / np.pi  # in [0, 180]

    # sort by angle
    sort_idx = np.argsort(doa_deg)
    doa_deg = doa_deg[sort_idx]

    return tuple(doa_deg)

