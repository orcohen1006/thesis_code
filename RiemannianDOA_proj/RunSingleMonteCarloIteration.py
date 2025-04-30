def run_single_mc_iteration(
        i_mc: int,
        algo_list: List[str],
        config: dict,
        seed: int
):
 

    t0 = time()

    # Fixed Source powers
    num_sources = len(config["doa"])  # # of sources

    power_doa = 10.0 ** (config["power_doa_db"] / 10.0)

    doa_scan = get_doa_grid()

    config["doa"] = np.sort(config["doa"])

    delta_vec = np.arange(config["m"])
    A_true = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(config["doa"] * np.pi / 180)))
    A = np.exp(1j * np.pi * np.outer(delta_vec, np.cos(doa_scan * np.pi / 180)))

    noise_power_db = np.max(config["power_doa_db"]) - config["snr"]
    noise_power = 10.0 ** (noise_power_db / 10.0)



    num_algos = len(algo_list)
    num_sources = len(config["doa"])
    amplitude_doa = np.sqrt(10.0 ** (config["power_doa_db"] / 10.0))

    noise_power_db = np.max(config["power_doa_db"]) - config["snr"]
    noise_power = 10.0 ** (noise_power_db / 10.0)

    y_noisy = generate_signal(A_true, config["power_doa_db"], config["N"], noise_power, cohr_flag=False, seed=seed)

    modulus_hat_das = np.sum(np.abs(A.conj().T @ (y_noisy / config["m"])), axis=1) / config["N"]

    # Run on all algorithms
    sqr_err = [None] * num_algos
    power_se = [None] * num_algos
    p_vec_cell = [None] * num_algos
    runtime_list = [None] * num_algos
    num_iters_list = [None] * num_algos

    for i_algo in range(num_algos):
        t_algo_start = time()
        if algo_list[i_algo] == "PER":
            p_vec, num_iters, _ = fun_DAS(y_noisy, A, modulus_hat_das, doa_scan, config["doa"])
        elif algo_list[i_algo] == "SAMV":
            p_vec, num_iters, _ = fun_SAMV(y_noisy, A, modulus_hat_das, doa_scan, config["doa"], noise_power)
        elif algo_list[i_algo] == "SPICE":
            p_vec, num_iters, _ = fun_SPICE(y_noisy, A, modulus_hat_das, doa_scan, config["doa"], noise_power)
        elif algo_list[i_algo] == "AIRM":
            p_vec, num_iters, _ = fun_Riemannian(y_noisy, A, modulus_hat_das, doa_scan, config["doa"], noise_power, loss_name="AIRM")
        elif algo_list[i_algo] == "JBLD":
            p_vec, num_iters, _ = fun_Riemannian(y_noisy, A, modulus_hat_das, doa_scan, config["doa"], noise_power, loss_name="JBLD")
        else:
            raise ValueError("Algorithm not implemented")

        runtime_list[i_algo] = time() - t_algo_start
        print(f"{algo_list[i_algo]}: #iters= {num_iters}, runtime= {runtime_list[i_algo]} [sec]")
        num_iters_list[i_algo] = num_iters

        p_vec_cell[i_algo] = p_vec
        detected_powers, distance, normal = detect_DOAs(p_vec, doa_scan, config["doa"])

        if not normal:
            sqr_err[i_algo] = np.nan
            power_se[i_algo] = np.nan
        else:
            power_dif = detected_powers - 10.0 ** (config["power_doa_db"] / 10.0)
            distance = distance.astype(float)
            sqr_err[i_algo] = np.dot(distance, distance)
            power_se[i_algo] = np.dot(power_dif, power_dif)

    if False:
        plt.figure()
        plt.grid(True)
        plts = []

        for i_algo in range(num_algos):
            plt_line, = plt.plot(doa_scan, 10 * np.log10(p_vec_cell[i_algo]), '-o', label=algo_list[i_algo])
            plts.append(plt_line)

        plt_doa, = plt.plot(config["doa"], config["power_doa_db"], 'x', label='DOA')
        plts.append(plt_doa)
        plt.legend(handles=plts)
        plt.ylim([-15,15])
        plt.show()
    # Convert list to array for processing
    se_all_m = np.array([se if se is not None else np.nan for se in sqr_err])

    # Use the flag to track failures
    nan_flag_col_vec = np.isnan(se_all_m)

    dt = time() - t0
    print(f"i_MC = {i_mc + 1}, elapsed time: {dt:.2f} [sec]")

    return se_all_m, nan_flag_col_vec, p_vec_cell, runtime_list, num_iters_list

