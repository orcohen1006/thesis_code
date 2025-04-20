import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from utils import *
from OptimizeRiemannianLoss import optimize_adam_AFFINV, optimize_adam_LE, optimize_adam_LD
from itertools import product


def test_hyper_params(input) -> None:
    # Test parameters
    np.random.seed(42)
    m = 12
    snr = 0
    N = 16
    MAX_ITERS = int(5e3)
    # Source powers in dB
    power_doa_db = np.array([3, 4])

    doa_scan = get_doa_grid()

    A = np.exp(1j * np.pi * np.outer(np.arange(m), np.cos(doa_scan * np.pi / 180)))

    firstDOA = 35.11

    lr_values = input['lr_values']
    optimize_func = input['optimize_func']

    # list_deltaDOA = [5,10]
    # list_snr = [-5, 0, 5]
    # list_N = [16, 120]
    # list_seed = np.arange(0,3).tolist()

    list_deltaDOA = [5]
    list_snr = [0]
    list_N = [16]
    list_seed = np.arange(0, 2).tolist()

    list_settings = list(product(list_deltaDOA, list_snr,list_N, list_seed))

    # Results tracking
    results = {
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

        # Run multiple scenarios for each learning rate
        for idx_setting, setting in enumerate(tqdm(list_settings, desc=f"Testing scenarios for lr={lr}", leave=False)):
            deltaDOA, snr, N, seed = setting
            doa = np.sort(np.array([firstDOA, firstDOA+deltaDOA]))
            A_true = np.exp(1j * np.pi * np.outer(np.arange(m), np.cos(doa * np.pi / 180)))
            noise_power_db = np.mean(power_doa_db) - snr
            noise_power = 10.0 ** (noise_power_db / 10.0)

            # Generate signal for this scenario
            Y = generate_signal(A_true, power_doa_db, N, noise_power, False, seed=seed)

            # Prepare DAS initial guess
            DAS_init = np.sum(np.abs(A.conj().T @ (Y / m)), axis=1) / N
            DAS_init = torch.tensor(DAS_init, dtype=torch.float)

            # Prepare R_hat
            R_hat = Y @ Y.conj().T / N

            # Optimize with this learning rate
            p, tuple_history = optimize_func(A, R_hat, noise_power, DAS_init,
                                                    _max_iter=MAX_ITERS, _lr=lr,
                                                    do_store_history=True)
            step_losses, rel_changes = tuple_history
            # Store results for this scenario
            scenario_final_losses.append(step_losses[-1] if step_losses else float('inf'))
            scenario_loss_histories.append(step_losses)
            scenario_rel_change_histories.append(rel_changes)

        # Store aggregated results for this learning rate
        results['lr'].append(lr)
        results['avg_final_loss'].append(np.mean(scenario_final_losses))
        results['std_final_loss'].append(np.std(scenario_final_losses))
        results['all_loss_histories'].append(scenario_loss_histories)
        results['all_rel_change_histories'].append(scenario_rel_change_histories)

    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(input['name'], fontsize=16)
    # Plot 1: Average loss curves for all learning rates
    for i, lr in enumerate(results['lr']):
        # Calculate mean and std of loss across scenarios at each iteration
        max_len = max(len(hist) for hist in results['all_loss_histories'][i])
        avg_losses = np.zeros(max_len)
        std_losses = np.zeros(max_len)

        for step in range(max_len):
            step_losses = [hist[step] if step < len(hist) else hist[-1]
                           for hist in results['all_loss_histories'][i]]
            avg_losses[step] = np.mean(step_losses)
            std_losses[step] = np.std(step_losses)

        # Plot mean loss with shaded std region
        x = np.arange(max_len)
        axs[0, 0].plot(x, avg_losses, label=f'lr={lr}')
        axs[0, 0].fill_between(x, avg_losses - std_losses, avg_losses + std_losses, alpha=0.2)

    axs[0, 0].set_title('Average Loss Curves Across Scenarios')
    axs[0, 0].set_xlabel('Iterations')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].set_yscale('log')

    # Plot 2: Early iterations (first 500)
    for i, lr in enumerate(results['lr']):
        # Calculate early iterations statistics
        early_iter = 500
        max_len = min(early_iter, max(len(hist) for hist in results['all_loss_histories'][i]))
        avg_losses = np.zeros(max_len)
        std_losses = np.zeros(max_len)

        for step in range(max_len):
            step_losses = [hist[step] if step < len(hist) else hist[-1]
                           for hist in results['all_loss_histories'][i]]
            avg_losses[step] = np.mean(step_losses)
            std_losses[step] = np.std(step_losses)

        # Plot mean loss with shaded std region
        x = np.arange(max_len)
        axs[0, 1].plot(x, avg_losses, label=f'lr={lr}')
        axs[0, 1].fill_between(x, avg_losses - std_losses, avg_losses + std_losses, alpha=0.2)

    axs[0, 1].set_title('Average Loss Curves (First 500 Iterations)')
    axs[0, 1].set_xlabel('Iterations')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].set_yscale('log')

    # Plot 3: Final loss vs learning rate (with error bars)
    axs[1, 0].errorbar(results['lr'], results['avg_final_loss'],
                       yerr=results['std_final_loss'], fmt='o-')
    axs[1, 0].set_title('Average Final Loss vs Learning Rate')
    axs[1, 0].set_xlabel('Learning Rate')
    axs[1, 0].set_ylabel('Final Loss (mean ± std)')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')

    # Plot 4:
    for i, lr in enumerate(results['lr']):
        # Average parameter changes across scenarios
        max_len = MAX_ITERS #min(1000, max(len(changes) for changes in results['all_rel_change_histories'][i]))
        avg_changes = np.zeros(max_len)
        std_changes = np.zeros(max_len)

        for step in range(max_len):
            step_changes = [changes[step] if step < len(changes) else 0
                            for changes in results['all_rel_change_histories'][i]]
            avg_changes[step] = np.mean(step_changes)
            std_changes[step] = np.std(step_changes)

        # Plot mean parameter change with shaded std region
        x = np.arange(max_len)
        axs[1, 1].plot(x, avg_changes, label=f'lr={lr}')
        axs[1, 1].fill_between(x, avg_changes - std_changes, avg_changes + std_changes, alpha=0.2)

    axs[1, 1].set_title('Parameter Change During Optimization')
    axs[1, 1].set_xlabel('Iterations')
    axs[1, 1].set_ylabel('‖pᵢ - pᵢ₋₁‖/‖pᵢ‖')
    axs[1, 1].legend()
    axs[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(input['name'] + '_learning_rate_comparison_multi_scenario.png')
    # plt.show()

    # Print recommendation
    # best_lr_idx = np.argmin(results['avg_final_loss'])
    # best_lr = results['lr'][best_lr_idx]
    # print(f"\nRecommended learning rate: {best_lr}")
    # print(
    #     f"Average lowest loss: {results['avg_final_loss'][best_lr_idx]:.6e} ± {results['std_final_loss'][best_lr_idx]:.6e}")
    # print(
    #     f"Average convergence in {results['avg_convergence_steps'][best_lr_idx]:.1f} ± {results['std_convergence_steps'][best_lr_idx]:.1f} iterations")

    # Create robustness plot - how consistent is each learning rate across scenarios?
    plt.figure(figsize=(10, 6))

    # For each LR, plot the distribution of final losses
    boxplot_data = []
    for i, lr in enumerate(results['lr']):
        # Collect all final losses for this learning rate
        final_losses = [hist[-1] for hist in results['all_loss_histories'][i]]
        boxplot_data.append(final_losses)

    plt.boxplot(boxplot_data, labels=[str(lr) for lr in results['lr']])
    plt.title('Distribution of Final Losses Across Scenarios')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Loss')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(input['name'] + '_learning_rate_robustness.png')
    # plt.show()


if __name__ == "__main__":
    test_hyper_params({'name':'AFFINV','optimize_func':optimize_adam_AFFINV,'lr_values':[1e-4, 1e-3, 1e-2]})
    test_hyper_params({'name':'LD','optimize_func':optimize_adam_LD,'lr_values':[1e-4, 1e-3, 1e-2]})
    # test_hyper_params({'name':'LE','optimize_func':optimize_adam_LE,'lr_values':[1e-3]})
    plt.show()
