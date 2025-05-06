import numpy as np
from utils import *

def cramer_rao_lower_bound(config):
    """
    Simple CRB (Cramer-Rao Bound)
    using the stochastic CRB definitions
    
    Parameters:
    config: Configuration dictionary containing:
        - snr: input SNR value
        - N: number of time samples
        - cohr_flag: coherent sources flag
        - power_doa_db: source powers in dB (optional)
        - doa: true DOA angles (optional)
    
    Returns:
    mse_CBRDOA: Mean Square Error Cramer-Rao Bound for DOA
    """
    m = config["m"]  # number of sensors
    t_samples = config["N"]
    
    PowerDOA = convert_db_to_linear(config["power_doa_db"])
    amplitudeDOA = np.sqrt(PowerDOA)
    
    # Complex Gaussian Noise
    SNR = config["snr"]  # input parameters
    noisePowerdB = np.max(config["power_doa_db"]) - SNR
    noisePower = convert_db_to_linear(noisePowerdB)

    delta_vec = np.arange(m)

    DOA = np.sort(config["doa"])  # must be in ascending order to work right
    source_no = len(DOA)  # # of sources
    
    # Real steering vector matrix
    A = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(DOA * np.pi / 180)))
    DA = (-1j * np.pi * np.outer(delta_vec, np.sin(DOA * np.pi / 180) * np.pi / 180)) * A
    
    PI_A = np.eye(m) - A @ np.linalg.inv(A.conj().T @ A) @ A.conj().T
    
    # Compute P, dependent on the cohr_flag
    if not config["cohr_flag"]:  # independent sources
        Pmat = np.diag(PowerDOA)
    else:
        rho = 1  # 100% coherence assumption
        # For simplicity, assuming only 2 sources
        Pmat = np.array([
            [PowerDOA[0], rho * amplitudeDOA[0] * amplitudeDOA[1]],
            [rho * amplitudeDOA[1] * amplitudeDOA[0], PowerDOA[1]]
        ])
    
    R = A @ Pmat @ A.conj().T + noisePower * np.eye(m)
    Inside = (DA.conj().T @ PI_A @ DA) * (Pmat @ A.conj().T @ np.linalg.inv(R) @ A @ Pmat).T
    CRB_matrix = np.linalg.inv(np.real(Inside)) * noisePower / (2 * t_samples)
    
    mse_CBRDOA = np.trace(CRB_matrix)
    
    return mse_CBRDOA
