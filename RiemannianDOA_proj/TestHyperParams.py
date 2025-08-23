# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from utils import *
from OptimizeRiemannianLoss import optimize_adam_AIRM, optimize_adam_LE, optimize_adam_JBLD
from fun_JBLD import *
from fun_JBLD_advanced import *
from itertools import product
import os
from functools import partial
# %%
def test_hyper_params(foldername, input) -> None:
    # Test parameters
    np.random.seed(42)
    m = 12
    MAX_ITERS = int(5e3)

    doa_scan = get_doa_grid()

    A = get_steering_matrix(doa_scan, m)

    lr_values = input['lr_values']
    optimize_func = input['optimize_func']

    # list_snr = [-5, 0, 5]
    # list_N = [25, 50]
    # list_seed = np.arange(0,3).tolist()

    list_snr = [-5, 0, 5]
    list_N = [12, 60]
    list_seed = np.arange(0,2).tolist()

    list_settings = list(product(list_snr,list_N, list_seed))

    # Results tracking
    results_curr_method = {
        'lr': [],
        'avg_final_loss': [],
        'std_final_loss': [],
        'avg_convergence_steps': [],
        'std_convergence_steps': [],
        'all_loss_histories': [],  # Will store a list of lists: [lr_idx][scenario_idx]
        'all_rel_change_histories': [],
    }

    # Test each learning rate
    for lr_idx, lr in enumerate(tqdm(lr_values, desc="Testing learning rates")):
        scenario_final_losses = []
        scenario_loss_histories = []
        scenario_rel_change_histories = []

        for idx_setting, setting in enumerate(tqdm(list_settings, desc=f"Testing scenarios for lr={lr}", leave=False)):
            snr, N, seed = setting
            # doa = np.sort(np.array([firstDOA, firstDOA+deltaDOA]))
            doa=np.array([35.25, 43.25, 51.25])
            power_doa_db=np.array([0, 0, -5])
            A_true = get_steering_matrix(doa, m)
            noise_power_db = np.max(power_doa_db) - snr
            noise_power = 10.0 ** (noise_power_db / 10.0)



            # Generate signal for this scenario
            Y = generate_signal(A_true, power_doa_db, N, noise_power, seed=seed)

            # Prepare DAS initial guess
            modulus_hat_das = np.sum(np.abs(A.conj().T @ (Y / m)), axis=1) / N


            # Prepare R_hat
            R_hat = Y @ Y.conj().T / N

            # Optimize with this learning rate
            p, _, tuple_history = optimize_func(A, R_hat, noise_power, modulus_hat_das,
                                                    _max_iter=MAX_ITERS, _lr=lr,
                                                    do_store_history=True)
            step_losses, rel_changes = tuple_history
            # Store results for this scenario
            scenario_final_losses.append(step_losses[-1] if step_losses else float('inf'))
            scenario_loss_histories.append(step_losses)
            scenario_rel_change_histories.append(rel_changes)

        # Store aggregated results for this learning rate
        results_curr_method['lr'].append(lr)
        results_curr_method['avg_final_loss'].append(np.mean(scenario_final_losses))
        results_curr_method['std_final_loss'].append(np.std(scenario_final_losses))
        results_curr_method['all_loss_histories'].append(scenario_loss_histories)
        results_curr_method['all_rel_change_histories'].append(scenario_rel_change_histories)

    # Plot results
    fig1, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle(input['name'], fontsize=16)
    # Plot 1: Average loss curves for all learning rates
    for i, lr in enumerate(results_curr_method['lr']):
        # Calculate mean and std of loss across scenarios at each iteration
        max_len = max(len(hist) for hist in results_curr_method['all_loss_histories'][i])
        q25_losses = np.zeros(max_len)
        q50_losses = np.zeros(max_len)
        q75_losses = np.zeros(max_len)

        for step in range(max_len):
            step_losses = [hist[step] if step < len(hist) else hist[-1]
                           for hist in results_curr_method['all_loss_histories'][i]]
            q25_losses[step] = np.quantile(step_losses, 0.25)
            q50_losses[step] = np.quantile(step_losses, 0.50)
            q75_losses[step] = np.quantile(step_losses, 0.75)

        # Plot mean loss with shaded std region
        x = np.arange(max_len)
        axs[0, 0].plot(x, q50_losses, label=f'lr={lr}')
        axs[0, 0].fill_between(x, q25_losses, q75_losses, alpha=0.2)

    axs[0, 0].set_title('Average Loss Curves Across Scenarios')
    axs[0, 0].set_xlabel('Iterations')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].set_yscale('log')

    # Plot 2: Early iterations (first 500)
    for i, lr in enumerate(results_curr_method['lr']):
        # Calculate early iterations statistics
        early_iter = 500
        max_len = min(early_iter, max(len(hist) for hist in results_curr_method['all_loss_histories'][i]))
        for step in range(max_len):
            step_losses = [hist[step] if step < len(hist) else hist[-1]
                            for hist in results_curr_method['all_loss_histories'][i]]
            q25_losses[step] = np.quantile(step_losses, 0.25)
            q50_losses[step] = np.quantile(step_losses, 0.50)
            q75_losses[step] = np.quantile(step_losses, 0.75)

        # Plot mean loss with shaded std region
        x = np.arange(max_len)
        axs[0, 1].plot(x, q50_losses[:max_len], label=f'lr={lr}')
        axs[0, 1].fill_between(x, q25_losses[:max_len], q75_losses[:max_len], alpha=0.2)

    axs[0, 1].set_title('Average Loss Curves (First 500 Iterations)')
    axs[0, 1].set_xlabel('Iterations')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].set_yscale('log')

    # Plot 3: Final loss vs learning rate (with error bars)
    axs[1, 0].errorbar(results_curr_method['lr'], results_curr_method['avg_final_loss'],
                       yerr=results_curr_method['std_final_loss'], fmt='o-')
    axs[1, 0].set_title('Average Final Loss vs Learning Rate')
    axs[1, 0].set_xlabel('Learning Rate')
    axs[1, 0].set_ylabel('Final Loss (mean ± std)')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')

    # Plot 4:
    for i, lr in enumerate(results_curr_method['lr']):
        # Average parameter changes across scenarios
        max_len = MAX_ITERS #min(1000, max(len(changes) for changes in results['all_rel_change_histories'][i]))
        q25_changes = np.zeros(max_len)
        q50_changes = np.zeros(max_len)
        q75_changes = np.zeros(max_len)

        for step in range(max_len):
            step_changes = [changes[step] if step < len(changes) else 0
                            for changes in results_curr_method['all_rel_change_histories'][i]]
            q25_changes[step] = np.quantile(step_changes, 0.25)
            q50_changes[step] = np.quantile(step_changes, 0.50)
            q75_changes[step] = np.quantile(step_changes, 0.75)

        # Plot mean parameter change with shaded std region
        x = np.arange(max_len)
        axs[1, 1].plot(x, q50_changes, label=f'lr={lr}')
        axs[1, 1].fill_between(x, q25_changes, q75_changes, alpha=0.2)

    axs[1, 1].set_title('Parameter Change During Optimization')
    axs[1, 1].set_xlabel('Iterations')
    axs[1, 1].set_ylabel('‖pᵢ - pᵢ₋₁‖/‖pᵢ‖')
    axs[1, 1].legend()
    axs[1, 1].set_yscale('log')

    plt.tight_layout()
    save_figure(fig1, foldername,  input['name'] + '_learning_rate_comparison')
    # plt.show()


    # Create robustness plot - how consistent is each learning rate across scenarios?
    fig2 = plt.figure(figsize=(10, 6))

    # For each LR, plot the distribution of final losses
    boxplot_data = []
    for i, lr in enumerate(results_curr_method['lr']):
        # Collect all final losses for this learning rate
        final_losses = [hist[-1] for hist in results_curr_method['all_loss_histories'][i]]
        boxplot_data.append(final_losses)

    plt.boxplot(boxplot_data, labels=[str(lr) for lr in results_curr_method['lr']])
    plt.title('Distribution of Final Losses Across Scenarios')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Loss')
    # plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_figure(fig2, foldername,  input['name'] + '_learning_rate_boxplots')
    # plt.show()

def compare_methods_and_hyperparams(foldername, list_dict_method) -> None:
    # Test parameters
    np.random.seed(42)
    m = 12
    MAX_ITERS = int(5e3)

    doa_scan = get_doa_grid()

    A = get_steering_matrix(doa_scan, m)

    list_snr = [-5, 0, 5]
    list_N = [12, 60]
    list_seed = np.arange(0,2).tolist()
    list_doa_settings = list(product(list_snr,list_N, list_seed))

    num_methods = len(list_dict_method)
    all_results = []
    for dict_method in list_dict_method:
        results_curr_method = run_aux(dict_method, list_doa_settings, A, m, MAX_ITERS)
        all_results.append(results_curr_method)
    
    num_rows = len(list_snr)
    num_cols = len(list_N) * len(list_seed)
    fig1, axes1 = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3),
        sharex=True, sharey=True
    )
    fig2, axes2 = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3),
        sharex=True, sharey=True
    )
    
    if num_rows == 1 or num_cols == 1:
        axes1 = np.array(axes1).reshape(num_rows, num_cols)
        axes2 = np.array(axes2).reshape(num_rows, num_cols)

    for i_doa_setting, (snr, N, seed) in enumerate(list_doa_settings):
        row_idx = list_snr.index(snr)
        col_idx = (list_N.index(N) * len(list_seed)) + list_seed.index(seed)

        min_loss_val = np.inf
        max_loss_val = -np.inf
        for result in all_results:
            for i_subresult in range(len(result['desc'])):
                loss_vals = result['all_loss_histories'][i_subresult][i_doa_setting]
                min_loss_val = np.min([np.min(loss_vals), min_loss_val])
                max_loss_val = np.max([np.max(loss_vals), max_loss_val])
        displayed_min_loss_val = 1e-10
        displayed_max_loss_val = 1      
        for result in all_results:
            num_subresults = len(result['desc'])
            for i_subresult in range(num_subresults):
                loss_vals = result['all_loss_histories'][i_subresult][i_doa_setting]
                loss_vals = (loss_vals - min_loss_val)/(displayed_max_loss_val - displayed_min_loss_val) + displayed_min_loss_val
                relchange_vals = result['all_rel_change_histories'][i_subresult][i_doa_setting]
                iters = np.arange(len(loss_vals))
                axes1[row_idx, col_idx].plot(iters, loss_vals, label=result['desc'][i_subresult])
                axes2[row_idx, col_idx].plot(iters, relchange_vals, label=result['desc'][i_subresult])

        axes1[row_idx, col_idx].set_title(f'SNR: {snr} dB, N: {N}, Seed: {seed}, min_loss={min_loss_val}',fontdict={"size":8})
        axes2[row_idx, col_idx].set_title(f'SNR: {snr} dB, N: {N}, Seed: {seed}',fontdict={"size":8})
        axes1[row_idx, col_idx].grid(True)
        axes2[row_idx, col_idx].grid(True)
        if (i_doa_setting == 0):
            axes1[row_idx, col_idx].legend()
            axes2[row_idx, col_idx].legend()
        axes1[row_idx, col_idx].set_yscale('log')
        axes2[row_idx, col_idx].set_yscale('log')
    fig1.tight_layout()
    fig2.tight_layout()
    save_figure(fig1, foldername,  'losses')
    save_figure(fig2, foldername,  'rel_changes')


    
def run_aux(dict_method, list_doa_settings, A, m, MAX_ITERS):
    lr_values = dict_method['lr_values']
    optimize_func = dict_method['optimize_func']
    results_curr_method = {
        'desc': [],
        'avg_final_loss': [],
        'std_final_loss': [],
        'avg_convergence_steps': [],
        'std_convergence_steps': [],
        'all_loss_histories': [],  # Will store a list of lists: [lr_idx][scenario_idx]
        'all_rel_change_histories': [],
    }

    # Test each learning rate
    print(f"-------------- {dict_method['name']} ----------------")
    for lr_idx, lr in enumerate(tqdm(lr_values, desc="Testing learning rates")):
        scenario_final_losses = []
        scenario_loss_histories = []
        scenario_rel_change_histories = []

        for idx_setting, setting in enumerate(tqdm(list_doa_settings, desc=f"Testing scenarios for lr={lr}", leave=False)):
            snr, N, seed = setting
            # doa = np.sort(np.array([firstDOA, firstDOA+deltaDOA]))
            doa=np.array([35.25, 43.25, 51.25])
            power_doa_db=np.array([0, 0, -5])
            A_true = get_steering_matrix(doa, m)
            noise_power_db = np.max(power_doa_db) - snr
            noise_power = 10.0 ** (noise_power_db / 10.0)



            # Generate signal for this scenario
            Y = generate_signal(A_true, power_doa_db, N, noise_power, seed=seed)

            # Prepare DAS initial guess
            modulus_hat_das = np.sum(np.abs(A.conj().T @ (Y / m)), axis=1) / N


            # Prepare R_hat
            R_hat = Y @ Y.conj().T / N

            # Optimize with this learning rate
            p, _, tuple_history = optimize_func(A, R_hat, noise_power, modulus_hat_das,
                                                    _max_iter=MAX_ITERS, _lr=lr,
                                                    do_store_history=True)
            step_losses, rel_changes = tuple_history
            # Store results for this scenario
            scenario_final_losses.append(step_losses[-1] if step_losses else float('inf'))
            scenario_loss_histories.append(step_losses)
            scenario_rel_change_histories.append(rel_changes)

        # Store aggregated results for this learning rate
        results_curr_method['desc'].append(f"{dict_method['name']}, lr={lr}")
        results_curr_method['avg_final_loss'].append(np.mean(scenario_final_losses))
        results_curr_method['std_final_loss'].append(np.std(scenario_final_losses))
        results_curr_method['all_loss_histories'].append(scenario_loss_histories)
        results_curr_method['all_rel_change_histories'].append(scenario_rel_change_histories)

    return results_curr_method

# if __name__ == "__main__":
#     foldername = "TestHyperparams_" + datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
#     os.makedirs(foldername, exist_ok=True)
#     # test_hyper_params(foldername, {'name':'AIRM','optimize_func':optimize_adam_AIRM, 'lr_values':[1e-2, 1e-1]})
#     # test_hyper_params(foldername, {'name':'JBLD','optimize_func':optimize_adam_JBLD, 'lr_values':[1e-2, 1e-1]})
    
#     # optimize_JBLD_cccp_adam_inner10 = partial(optimize_JBLD_cccp, inner_opt='adam', inner_iters=10)
#     # test_hyper_params(foldername, {'name':'JBLD_cccp_adam','optimize_func':optimize_JBLD_cccp_adam_inner10, 'lr_values':[1e-2, 1e-1]})

#     # optimize_JBLD_cccp_adam_inner25 = partial(optimize_JBLD_cccp, inner_opt='adam', inner_iters=25)
#     # test_hyper_params(foldername, {'name':'JBLD_cccp_adam_inner25','optimize_func':optimize_JBLD_cccp_adam_inner25, 'lr_values':[1e-2, 1e-1]})

#     optimize_JBLD_cccp_adam_inner5 = partial(optimize_JBLD_cccp, inner_opt='adam', inner_iters=5)
#     test_hyper_params(foldername, {'name':'JBLD_cccp_adam_inner5','optimize_func':optimize_JBLD_cccp_adam_inner5, 'lr_values':[1e-2]})

#     # optimize_JBLD_cccp_lbfgs_inner25 = partial(optimize_JBLD_cccp, inner_opt='lbfgs', inner_iters=25, line_search_fn_bfgs="strong_wolfe")
#     # test_hyper_params(foldername, {'name':'JBLD_cccp_lbfgs_inner25','optimize_func':optimize_JBLD_cccp_lbfgs_inner25, 'lr_values':[5e-1, 1]})


#     # optimize_JBLD_cccp_lbfgs_inner10 = partial(optimize_JBLD_cccp, inner_opt='lbfgs', inner_iters=10, line_search_fn_bfgs="strong_wolfe")
#     # test_hyper_params(foldername, {'name':'JBLD_cccp_lbfgs_inner10','optimize_func':optimize_JBLD_cccp_lbfgs_inner10, 'lr_values':[5e-1]})

#     plt.show()


if __name__ == "__main__":
    foldername = "zCompareMethodsHyperparams_" + datetime.now().strftime('y%Y-m%m-d%d_%H-%M-%S')
    os.makedirs(foldername, exist_ok=True)




    # optimize_JBLD_cccp_lbfgs_inner25 = partial(optimize_JBLD_cccp, inner_opt='lbfgs', inner_iters=25, line_search_fn_bfgs="strong_wolfe")
    # optimize_JBLD_cccp_adam_inner5 = partial(optimize_JBLD_cccp, inner_opt='adam', inner_iters=5)
    # optimize_JBLD_cccp_lbfgs_inner5 = partial(optimize_JBLD_cccp, inner_opt='lbfgs', inner_iters=5, line_search_fn_bfgs="strong_wolfe")


    # optimize_JBLD_cccp_repar_lbfgs_inner20 = partial(optimize_JBLD_cccp_with_reparametrization, inner_opt='lbfgs', inner_iters=20, line_search_fn_bfgs="strong_wolfe")
    # optimize_JBLD_cccp_repar_adam_inner5 = partial(optimize_JBLD_cccp_with_reparametrization, inner_opt='adam', inner_iters=5)


    compare_methods_and_hyperparams(foldername, 
                                    [
                                     {'name':'JBLD_scipy_lbfgsb_inner20','optimize_func':partial(optimize_JBLD_cccp, inner_opt='scipy_lbfgsb', inner_iters=20), 'lr_values':[1e-2]},
                                     {'name':'JBLD_scipy_lbfgsb_inner10','optimize_func':partial(optimize_JBLD_cccp, inner_opt='scipy_lbfgsb', inner_iters=10), 'lr_values':[1e-2]},
                                    #  {'name':'JBLD_scipy_lbfgsb_inner10','optimize_func':partial(optimize_JBLD_cccp, inner_opt='scipy_lbfgsb', inner_iters=10), 'lr_values':[1e-2]},
                                    #  {'name':'JBLD_scipy_lbfgsb_inner10_gtol1e-3','optimize_func':partial(optimize_JBLD_cccp, inner_opt='scipy_lbfgsb', inner_iters=10, gtol=1e-3), 'lr_values':[1e-2]},
                                    # {'name':'JBLD_scipy_lbfgsb_inner25','optimize_func':partial(optimize_JBLD_cccp, inner_opt='scipy_lbfgsb', inner_iters=25), 'lr_values':[1e-2]},
                                    # {'name':'JBLD_scipy_lbfgsb_inner25_gtol1e-3','optimize_func':partial(optimize_JBLD_cccp, inner_opt='scipy_lbfgsb', inner_iters=25, gtol=1e-3), 'lr_values':[1e-2]},
                                    # {'name':'JBLD_scipy_lbfgsb_inner50','optimize_func':partial(optimize_JBLD_cccp, inner_opt='scipy_lbfgsb', inner_iters=50), 'lr_values':[1e-2]},
                                    #  {'name':'JBLD_cccp_sgd_inner5','optimize_func':partial(optimize_JBLD_cccp, inner_opt='sgd', inner_iters=5), 'lr_values':[1e-2]},
                                    #  {'name':'JBLD_cccp_sgd_inner10','optimize_func':partial(optimize_JBLD_cccp, inner_opt='sgd', inner_iters=10), 'lr_values':[1e-2]},
                                    #  {'name':'JBLD_cccp_ls_sgd_inner20','optimize_func':partial(optimize_JBLD_cccp, inner_opt='ls_sgd', inner_iters=20), 'lr_values':[5e-2]},
                                     {'name':'JBLD_cccp_sgd_inner20','optimize_func':partial(optimize_JBLD_cccp, inner_opt='sgd', inner_iters=20), 'lr_values':[1e-1]},
                                    #  {'name':'JBLD_cccp_adam_inner5','optimize_func':partial(optimize_JBLD_cccp, inner_opt='adam', inner_iters=5), 'lr_values':[1e-2]},
                                     {'name':'JBLD_adam_cholesky','optimize_func':optimize_adam_cholesky_JBLD, 'lr_values':[1e-2]},
                                    #  {'name':'JBLD_BB','optimize_func':optimize_JBLD_BB, 'lr_values':[1]},
                                    #  {'name':'JBLD','optimize_func':optimize_adam_JBLD, 'lr_values':[1e-2]},
                                    #  {'name':'JBLD_cccp_lbfgs_inner5','optimize_func':optimize_JBLD_cccp_lbfgs_inner5, 'lr_values':[1e-2,1e-1, 1]},
                                    #  {'name':'optimize_JBLD_cccp_repar_lbfgs_inner20','optimize_func':optimize_JBLD_cccp_repar_lbfgs_inner20, 'lr_values':[5e-1, 1e0, 2e0]},
                                    #  {'name':'optimize_JBLD_cccp_repar_adam_inner5','optimize_func':optimize_JBLD_cccp_repar_adam_inner5, 'lr_values':[1e-1,5e-1]},
                                     ])