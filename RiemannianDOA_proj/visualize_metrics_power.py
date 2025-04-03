from dataclasses import dataclass
from datetime import datetime
import dill
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable
from utils import *
import scipy

m = 12
snr = -5
N = 20
power_doa_db = np.array([0, -5])
sources_power = 10.0 ** (power_doa_db / 10.0)
doas_true = np.array([40, 45])
power_db_scan = np.linspace(start=power_doa_db.min() - 10, stop=power_doa_db.max() + 10, num=int(1e2))

A_true = np.exp(1j * np.pi * np.outer(np.arange(m), np.cos(doas_true * np.pi / 180)))
noise_power_db = np.max(power_doa_db) - snr
noise_power = 10.0 ** (noise_power_db / 10.0)
Y = generate_signal(A_true, power_doa_db, N, noise_power, False, seed=42)
R_hat = Y @ Y.conj().T / N
# sources_power = 10.0 ** (power_doa_db / 10.0)
# R_hat = A_true @ np.diag(sources_power) @ A_true.conj().T + noise_power * np.eye(m)

pinv_sqrtm_fun = lambda x: scipy.linalg.sqrtm(np.linalg.pinv(x))
pinv_sqrtm_R_hat = pinv_sqrtm_fun(R_hat)

fun_AIRM = lambda R: np.linalg.matrix_norm(scipy.linalg.logm(pinv_sqrtm_R_hat @ R @ pinv_sqrtm_R_hat)) ** 2
fun_JBLD = lambda R: np.linalg.slogdet(0.5*(R_hat + R))[1] -0.5*np.linalg.slogdet(R_hat@R)[1]
fun_SPICE = lambda R: np.linalg.matrix_norm(pinv_sqrtm_fun(R) @ (R - R_hat) @ pinv_sqrtm_R_hat ) ** 2
fun_MV = lambda R: np.linalg.matrix_norm(pinv_sqrtm_fun(R) @ (R - R_hat) @ pinv_sqrtm_R_hat) ** 2
fun_ML = lambda R: np.linalg.slogdet(R)[1] + np.trace(np.linalg.pinv(R) @ R_hat)

@dataclass
class Metric:
    name: str
    fun: Callable[[np.ndarray], np.ndarray]
    color: str
    resultMat: np.ndarray


def calculate_metrics():
    np.random.seed(42)
    resultMat = np.zeros(shape=(len(power_db_scan), len(power_db_scan)))
    metrics = [
        Metric("AIRM", fun_AIRM, "r", resultMat.copy()),
        Metric("JBLD", fun_JBLD, "g", resultMat.copy()),
        Metric("SPICE", fun_SPICE, "b", resultMat.copy()),
        Metric("AMV", fun_MV, "m", resultMat.copy()),
        # Metric("ML", fun_ML, "m", resultMat.copy()),
    ]
    for ii in range(0, len(power_db_scan)):
        for jj in range(0, len(power_db_scan)):
            print(f'({ii}, {jj})')
            doas = doas_true
            A = np.exp(1j * np.pi * np.outer(np.arange(m), np.cos(doas * np.pi / 180)))
            sources_power = 10.0 ** (np.array([power_db_scan[ii],power_db_scan[jj]]) / 10.0)
            R = A @ np.diag(sources_power) @ A.conj().T + noise_power * np.eye(m)
            for metric in metrics:
                metric.resultMat[ii, jj] = metric.fun(R)

    return metrics


def visualize_metrics(metrics):
    # ---- PLOTTING ----
    # Create figure with more space between subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True, constrained_layout=True)  # Increased figure size
    axes = axes.flatten()  # Flatten to easily iterate over axes

    # Loop over metrics and axes
    for ax, metric in zip(axes, metrics):
        X, Y = np.meshgrid(power_db_scan, power_db_scan)  # Create grid
        Z = metric.resultMat
        Z_display = Z.copy()

        # Create contour plot
        c = ax.contourf(X, Y, Z_display, cmap="coolwarm", levels=60, alpha=0.75)

        # Create contour lines
        # contour_lines = ax.contour(X, Y, Z, levels=5, colors='black', linewidths=2)
        percentile_levels = [np.percentile(Z, 1), np.percentile(Z, 25), np.percentile(Z, 50)]
        bold_contours = ax.contour(X, Y, Z_display, levels=percentile_levels, colors='black', linewidths=2)

        # Add red 'x' for true position
        ax.scatter(power_doa_db[1], power_doa_db[0], color='red', marker='x', s=30, edgecolor='black', linewidth=2, zorder=10)

        # Add colorbar
        cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)  # Slightly adjusted colorbar
        # cbar.set_label('Loss', fontsize=12)

        # Set plot titles, labels, and grid
        ax.set_title(metric.name, fontsize=16, pad=20)  # Added padding to title
        ax.set_xlabel(r"$power_1 [dB]$", fontsize=14)
        ax.set_ylabel(r"$power_2 [dB]$", fontsize=14)

        # Customize gridlines and ticks
        ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', length=6, width=2, direction='in', grid_color='gray', grid_alpha=0.7)
        ax.tick_params(axis='both', which='minor', length=3, width=1, direction='in', grid_color='gray', grid_alpha=0.5)
        ax.set_aspect('equal')
    # Adjust layout with more space

    # plt.tight_layout(w_pad=1, h_pad=1)  # Added padding between subplots
    # plt.tight_layout()



if __name__ == "__main__":
    timestamp = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    # ================
    metrics = calculate_metrics()
    # ================
    dir_name = 'visualize_metrics_power' + timestamp
    os.makedirs(dir_name)
    with open(os.path.join(dir_name, 'data.pkl'), 'wb') as f:
        dill.dump(metrics, f)
    # ================
    # # dir_name = 'visualize_metrics_power27-03-2025_19-04-08'
    # with open(os.path.join(dir_name, 'data.pkl'), 'rb') as f:
    #     metrics = dill.load(f)
    # ================
    visualize_metrics(metrics)
    plt.savefig(os.path.join(dir_name, 'metrics.png'), dpi=300)
    plt.show()
