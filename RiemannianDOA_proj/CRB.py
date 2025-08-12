import numpy as np
from utils import *

def cramer_rao_lower_bound(config):
    m = config["m"]
    N = config["N"]
    doa_deg = np.sort(config["doa"])
    K = len(doa_deg)
    p_vec = convert_db_to_linear(config["power_doa_db"])
    
    SNR = config["snr"]
    noise_power_db = np.max(config["power_doa_db"]) - SNR
    noise_power = convert_db_to_linear(noise_power_db)

    A, dA = get_steering_matrix(doa_deg, m, calcGradient_wrt_radians=True)

    A_H = A.T.conj()
    I = np.eye(m)
    if config["cohr_flag"]:
        assert(len(p_vec) == 2) # implemented only for 2 sources
        rho = config["cohr_coeff"]
        P = np.array([
            [p_vec[0], rho * np.sqrt(p_vec[0] * p_vec[1])],
            [rho*np.sqrt(p_vec[0] * p_vec[1]), p_vec[1]]
        ])
    else:
        P = np.diag(p_vec)               # K x K

    # Compute the covairance matrix: R = A P A^H + \sigma I
    R = A @ P @ A_H + noise_power * I
    # Compute the projection matrix: A (A^H A)^{-1} A^H
    P_A = A @ np.linalg.solve(A_H @ A, A_H)
    # Compute the H matrix.
    H = dA.T.conj() @ (I - P_A) @ dA
    # Compute the CRB
    CRB = (H * ((P @ (A_H @ np.linalg.solve(R, A)) @ P).T)).real
    CRB = np.linalg.inv(CRB) * (noise_power / N / 2)
    CRB *= (180 / np.pi) ** 2  # Convert to degrees squared
    diag_CRB = np.diag(CRB)
    return diag_CRB

def HARD_cramer_rao_lower_bound(config):
    if (config["cohr_flag"]):
        return np.nan  # Not implemented for coherent sources
    m = config["m"]
    N = config["N"]
    doa_deg = np.sort(config["doa"])
    K = len(doa_deg)
    p_vec = convert_db_to_linear(config["power_doa_db"])
    
    SNR = config["snr"]
    noise_power_db = np.max(config["power_doa_db"]) - SNR
    noise_power = convert_db_to_linear(noise_power_db)

    A, dA = get_steering_matrix(doa_deg, m, calcGradient_wrt_radians=True)

    P = np.diag(p_vec)               # K x K
    R = A @ P @ A.conj().T + noise_power * np.eye(m)  # m x m
    R_inv = np.linalg.inv(R)

    FIM = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        dai = dA[:, i:i+1]  # m x 1
        ai = A[:, i:i+1]    # m x 1
        dRi = p_vec[i] * (dai @ ai.conj().T + ai @ dai.conj().T)  # m x m

        for j in range(K):
            daj = dA[:, j:j+1]
            aj = A[:, j:j+1]
            dRj = p_vec[j] * (daj @ aj.conj().T + aj @ daj.conj().T)

            FIM[i, j] = N * np.real(np.trace(R_inv @ dRi @ R_inv @ dRj))

    CRB = np.linalg.inv(FIM)  # K x K
    CRB *= (180 / np.pi) ** 2  # Convert to degrees squared
    diag_CRB = np.diag(CRB)
    return diag_CRB

def OLD_cramer_rao_lower_bound(config):
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

    # # Normalize by the number of sources
    # K = len(config["doa"])
    # mse_CBRDOA /= K
    
    return mse_CBRDOA
