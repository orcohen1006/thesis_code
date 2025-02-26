import numpy as np
from scipy import linalg
from utils import *


def adamupdate(p, grad, averageGrad, averageSqGrad, iter, learnRate, gradDecay, sqGradDecay):
    """
    Implement Adam optimizer update
    """
    if len(averageGrad) == 0:
        averageGrad = np.zeros_like(p)
        averageSqGrad = np.zeros_like(p)
    
    # Update biased first moment estimate
    averageGrad = gradDecay * averageGrad + (1 - gradDecay) * grad
    
    # Update biased second raw moment estimate
    averageSqGrad = sqGradDecay * averageSqGrad + (1 - sqGradDecay) * grad**2
    
    # Compute bias-corrected first moment estimate
    correctedGrad = averageGrad / (1 - gradDecay**iter)
    
    # Compute bias-corrected second raw moment estimate
    correctedSqGrad = averageSqGrad / (1 - sqGradDecay**iter)
    
    # Update parameters
    p = p - learnRate * correctedGrad / (np.sqrt(correctedSqGrad) + 1e-8)
    
    return p, averageGrad, averageSqGrad

def fun_Affinv_aux(sigma2_n, A, GammaTensor, DAS_init):
    """Auxiliary function for Affinv algorithm"""
    D = A.shape[1]
    M = A.shape[0]
    L = 1  # Assuming GammaTensor is 2D
    
    NUM_MAX_ITERS = 10000
    eta = 1e-3
    eps_pow = 1e-5
    nu = 0  # D**2 * 1e-8 in original
    beta = 0.9
    EPS_NORM_CHANGE = 1e-4
    
    p_init = DAS_init
    
    # Initialize logger (for debugging/tracking)
    logger = {
        'p_init': p_init,
        'all_grads': [],
        'all_ps': [],
        'loss_vec': []
    }
    
    p = p_init.copy()
    sum_grad_squared = 0
    iter_best_p = None
    loss_best_p = float('inf')
    iter_prev_best_p = None
    loss_prev_best_p = float('inf')
    averageGrad = []
    averageSqGrad = []
    
    all_grads = np.zeros((len(p), NUM_MAX_ITERS+1))
    all_ps = np.zeros((len(p), NUM_MAX_ITERS+1))
    loss_vec = np.zeros(NUM_MAX_ITERS+1)
    
    for iter in range(1, NUM_MAX_ITERS+2):
        R = A @ np.diag(p) @ A.conj().T + sigma2_n * np.eye(M)
        grad = 0
        prev_loss = 0
        
        # Assuming GammaTensor is just R_hat (not a 3D tensor)
        invsqrtm_R_l = np.linalg.pinv(linalg.sqrtm(GammaTensor))
        Q = invsqrtm_R_l @ R @ invsqrtm_R_l
        Q = (Q + Q.conj().T) / 2  # For numerical stability
        
        U, Lam, _ = np.linalg.svd(Q)
        lambdas = Lam
        
        prev_loss = np.sum(np.log(lambdas)**2)
        
        V = np.abs(U.conj().T @ invsqrtm_R_l @ A)**2
        b = 2 * np.log(lambdas) / lambdas
        grad = V.T @ b
        
        loss_vec[iter-1] = prev_loss
        
        if prev_loss < loss_best_p:
            iter_prev_best_p = iter_best_p
            loss_prev_best_p = loss_best_p
            
            iter_best_p = iter - 1
            loss_best_p = prev_loss
            
        if iter > NUM_MAX_ITERS:
            break
            
        if iter_prev_best_p is not None and iter_best_p is not None and iter_prev_best_p > 2 and iter_best_p > 2:
            measured_change_norm = np.linalg.norm(all_ps[:, iter_best_p] - all_ps[:, iter_prev_best_p]) / np.linalg.norm(all_ps[:, iter_prev_best_p])
            if measured_change_norm < EPS_NORM_CHANGE:
                break
                
        # Add regularization
        grad_regularization = nu * (p >= eps_pow)
        grad = grad + grad_regularization
        
        # Adam update
        learnRate = 1e-3
        gradDecay = 0.95
        sqGradDecay = 0.95
        p, averageGrad, averageSqGrad = adamupdate(p, grad, averageGrad, averageSqGrad, iter, learnRate, gradDecay, sqGradDecay)
        
        # Set small values to zero
        p[p < eps_pow] = 0
        
        # Store history
        all_grads[:, iter-1] = grad
        all_ps[:, iter-1] = p
    
    # Choose best iteration based on loss
    chosen_last_iter = iter_best_p if iter_best_p is not None else 0
    
    logger['all_grads'] = all_grads[:, :chosen_last_iter]
    logger['all_ps'] = all_ps[:, :chosen_last_iter]
    logger['loss_vec'] = loss_vec[:chosen_last_iter+1]
    
    if chosen_last_iter == 0:
        p = p_init
    else:
        p = all_ps[:, chosen_last_iter]
        
    return p, logger

def fun_Affinv(Y, A, DAS_init, DOAscan, DOA, noise_power):
    """
    Affine Invariant estimator implementation.
    
    Parameters:
    Y: measured data, each col. is one snapshot
    A: steering vector matrix
    DAS_init: initial coefficients estimates by DAS
    DOAscan: grid
    DOA: truth (actual DOA angles)
    noise_power: known noise power
    
    Returns:
    Detected_powers: powers of detected sources
    Distance: difference between detected and true DOAs
    p: power spectrum vector
    normal: tag (1 if detection OK, 0 if failed)
    noisepower: estimated noise power
    """
    t_samples = Y.shape[1]
    R_hat = Y @ Y.conj().T / t_samples
    sigma2_n = noise_power
    
    # Call auxiliary function for Affinv estimation
    p, _ = fun_Affinv_aux(sigma2_n, A, R_hat, DAS_init)
    Detected_powers, Distance, normal = detect_DOAs(p, DOAscan, DOA)

    
    noisepower = sigma2_n
    
    return Detected_powers, Distance, p, normal, noisepower
