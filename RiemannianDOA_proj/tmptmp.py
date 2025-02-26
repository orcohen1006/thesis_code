import numpy as np
import time
import os


def ComputeAlgosStdErr(algo_list, NUM_MC, SNR, t_samples, M, cohr_flag, PowerDOAdB, DOA):
    num_algos = len(algo_list)
    Failed_total_times = np.zeros(num_algos)
    SE_history = np.zeros((num_algos, NUM_MC))
    num_sources = len(DOA)
    PowerDOA = 10 ** (PowerDOAdB / 10)
    amplitudeDOA = np.sqrt(PowerDOA)
    DOAscan = np.arange(0, 180.5, 0.5)
    DOA = np.sort(DOA)
    delta_vec = np.arange(M)
    A_true = np.exp(1j * np.pi * delta_vec[:, None] * np.cos(np.deg2rad(DOA)))
    A = np.exp(1j * np.pi * delta_vec[:, None] * np.cos(np.deg2rad(DOAscan)))
    noisePowerdB = np.mean(PowerDOAdB) - SNR
    noisePower = 10 ** (noisePowerdB / 10)

    for i_MC in range(NUM_MC):
        noise = np.sqrt(noisePower) * (np.random.randn(M, t_samples) + 1j * np.random.randn(M, t_samples)) / np.sqrt(2)

        if not cohr_flag:
            waveform = np.exp(1j * 2 * np.pi * np.random.rand(num_sources, t_samples)) * amplitudeDOA[:, None]
        else:
            waveform = np.exp(1j * 2 * np.pi * np.random.rand(num_sources - 1, t_samples))
            waveform = np.vstack((waveform, waveform[0, :])) * amplitudeDOA[:, None]

        y_noisefree = A_true @ waveform
        y_noisy = y_noisefree + noise
        modulus_hat_das = np.sum(np.abs(A.T @ y_noisy / M), axis=1) / t_samples

        for i_algo, algo in enumerate(algo_list):
            if algo == "PER":
                Detected_powers, Distance, _, normal, _ = fun_DASRes(y_noisy, A, modulus_hat_das, DOAscan, DOA)
            elif algo == "SAMV":
                Detected_powers, Distance, _, normal, _ = fun_SAM3Res(y_noisy, A, modulus_hat_das, DOAscan, DOA,
                                                                      noisePower)
            elif algo == "SPICE":
                Detected_powers, Distance, _, normal, _ = fun_SPICEplusRes(y_noisy, A, modulus_hat_das, DOAscan, DOA,
                                                                           noisePower)
            elif algo == "AFFINV":
                Detected_powers, Distance, _, normal, _ = fun_Affinv(y_noisy, A, modulus_hat_das, DOAscan, DOA,
                                                                     noisePower)
            else:
                raise ValueError("Algorithm not implemented")

            if not normal:
                SE_history[i_algo, i_MC] = np.inf
                Failed_total_times[i_algo] += 1
            else:
                SE_history[i_algo, i_MC] = np.sum(Distance ** 2)

    SE_history_sorted = np.sort(SE_history, axis=1)
    percent = 2
    actual_del_entries = int(np.ceil(NUM_MC * percent / 100))

    SE_mean_perAlgo = np.zeros(num_algos)
    Failing_rate_perAlgo = np.zeros(num_algos)
    for i_algo in range(num_algos):
        valid_entries = SE_history_sorted[i_algo, SE_history_sorted[i_algo, :] < np.inf]
        SE_mean_perAlgo[i_algo] = np.mean(valid_entries[actual_del_entries:])
        Failing_rate_perAlgo[i_algo] = Failed_total_times[i_algo] / NUM_MC

    CRB_val = SAM_CRB(SNR, t_samples, cohr_flag, PowerDOAdB, DOA)
    return SE_mean_perAlgo, Failing_rate_perAlgo, CRB_val

