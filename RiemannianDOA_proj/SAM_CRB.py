import numpy as np
from utils import create_config

def SAM_CRB(config):
    """
    Simple CRB (Cramer-Rao Bound) for the SAMV paper draft
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
    m = 12
    t_samples = config["N"]
    
    # Fixed Source powers
    if config["power_doa_db"] is None:
        config["power_doa_db"] = np.array([3, 4])  # in dB
    
    PowerDOA = 10 ** (config["power_doa_db"] / 10)
    amplitudeDOA = np.sqrt(PowerDOA)
    
    # Complex Gaussian Noise
    SNR = config["snr"]  # input parameters
    noisePowerdB = np.max(config["power_doa_db"]) - SNR
    noisePower = 10 ** (noisePowerdB / 10)

    delta_vec = np.arange(m)

    if config["doa"] is None:
        config["doa"] = np.array([35.11, 50.15])  # true DOA angles, off grid case
    
    DOA = np.sort(config["doa"])  # must be in ascending order to work right
    source_no = len(DOA)  # # of sources
    
    # Real steering vector matrix
    A = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(DOA * np.pi / 180)))
    DA = (-1j * np.pi * np.outer(delta_vec, np.sin(DOA * np.pi / 180) * np.pi / 180)) * A
    
    PI_A = np.eye(m) - A @ np.linalg.inv(A.conj().T @ A) @ A.conj().T
    
    # Compute P, dependent on the cohr_flag
    if not config.get("cohr_flag", False):  # independent sources
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
