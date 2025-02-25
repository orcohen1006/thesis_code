import numpy as np

def SAM_CRB(SNR_in, t_sample_in, cohr_flag, PowerDOAdB=None, DOA=None):
    """
    Simple CRB (Cramer-Rao Bound) for the SAMV paper draft
    using the stochastic CRB definitions
    
    Parameters:
    SNR_in: input SNR value
    t_sample_in: number of time samples
    cohr_flag: coherent sources flag
    PowerDOAdB: source powers in dB (optional)
    DOA: true DOA angles (optional)
    
    Returns:
    mse_CBRDOA: Mean Square Error Cramer-Rao Bound for DOA
    """
    M = 12
    t_samples = t_sample_in
    
    # Fixed Source powers
    if PowerDOAdB is None:
        PowerDOAdB = np.array([3, 4])  # in dB
    
    PowerDOA = 10 ** (PowerDOAdB / 10)
    amplitudeDOA = np.sqrt(PowerDOA)
    
    # Complex Gaussian Noise
    SNR = SNR_in  # input parameters
    noisePowerdB = np.mean(PowerDOAdB) - SNR
    noisePower = 10 ** (noisePowerdB / 10)
    
    # Inter-element spacing of sensors
    Dist = np.ones(M - 1)
    DistTmp = np.concatenate(([0], np.cumsum(Dist)))  # locations of the M sensors
    
    if DOA is None:
        DOA = np.array([35.11, 50.15])  # true DOA angles, off grid case
    
    DOA = np.sort(DOA)  # must be in ascending order to work right
    source_no = len(DOA)  # # of sources
    
    # Real steering vector matrix
    A = np.exp(1j * np.pi * np.outer(DistTmp, np.cos(DOA * np.pi / 180)))
    DA = (-1j * np.pi * np.outer(DistTmp, np.sin(DOA * np.pi / 180) * np.pi / 180)) * A
    
    PI_A = np.eye(M) - A @ np.linalg.inv(A.conj().T @ A) @ A.conj().T
    
    # Compute P, dependent on the cohr_flag
    if not cohr_flag:  # independent sources
        Pmat = np.diag(PowerDOA)
    else:
        rho = 1  # 100% coherence assumption
        # For simplicity, assuming only 2 sources
        Pmat = np.array([
            [PowerDOA[0], rho * amplitudeDOA[0] * amplitudeDOA[1]],
            [rho * amplitudeDOA[1] * amplitudeDOA[0], PowerDOA[1]]
        ])
    
    R = A @ Pmat @ A.conj().T + noisePower * np.eye(M)
    Inside = (DA.conj().T @ PI_A @ DA) * (Pmat @ A.conj().T @ np.linalg.inv(R) @ A @ Pmat).T
    CRB_matrix = np.linalg.inv(np.real(Inside)) * noisePower / (2 * t_samples)
    
    mse_CBRDOA = np.trace(CRB_matrix)
    
    return mse_CBRDOA
