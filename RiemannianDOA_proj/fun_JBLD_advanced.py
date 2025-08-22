import torch
from torch import Tensor
from typing import Callable, Dict, Optional, Tuple
from OptimizeRiemannianLoss import  TORCH_DTYPE
from utils import EPS_REL_CHANGE
###############################################################################
# Utilities
###############################################################################

def hermitian(A: Tensor) -> Tensor:
    return A.mT.conj()

@torch.no_grad()
def project_nonneg_(p: Tensor) -> Tensor:
    p.clamp_(min=0)
    return p


def build_R(A: Tensor, p: Tensor, sigma2: float) -> Tensor:
    """R(p) = A diag(p) A^H + sigma2 I
    Works with complex or real A. p is real, shape (K,).
    """
    # Scale columns by p: A @ diag(p) == A * p (broadcast along rows)
    As = A * p.unsqueeze(0)  # (M,K)
    R = As @ hermitian(A)
    M = A.shape[0]
    R = R + sigma2 * torch.eye(M, dtype=R.dtype, device=R.device)
    return R


def cholesky_logdet_and_solve(R: Tensor, B: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
    """Return (logdet(R), X) where X solves R X = B if B is given.
    Uses Cholesky for stability.
    """
    L = torch.linalg.cholesky(R)
    # logdet from Cholesky: 2 * sum(log(diag(L)))
    logdetR = 2.0 * torch.log(L.diagonal(dim1=-2, dim2=-1).real).sum()
    if B is None:
        return logdetR, None
    # Solve R X = B via Cholesky
    # torch.cholesky_solve expects lower-triangular L
    X = torch.cholesky_solve(B, L, upper=False)
    return logdetR, X


def jbl_loss(A: Tensor, p: Tensor, sigma2: float, Rhat: Tensor) -> Tensor:
    """JBLD loss: logdet((R+Rhat)/2) - 0.5*logdet(R*Rhat).
    The term -0.5*logdet(Rhat) is constant wrt p, but we keep the full
    expression for readability; autograd will ignore constants.
    Efficient and stable via Cholesky.
    """
    with torch.no_grad():
        project_nonneg_(p)
        
    R = build_R(A, p, sigma2)
    # (R + Rhat)/2
    Rbar = 0.5 * (R + Rhat)

    logdet_Rbar, _ = cholesky_logdet_and_solve(Rbar)
    logdet_R, _ = cholesky_logdet_and_solve(R)
    logdet_Rhat, _ = cholesky_logdet_and_solve(Rhat)

    return logdet_Rbar - 0.5 * logdet_R # - 0.5 * logdet_Rhat


def jbl_grad(A: Tensor, p: Tensor, sigma2: float, Rhat: Tensor) -> Tensor:
    """Explicit gradient for monitoring/debugging (optional).
    dL/dp_k = 0.5 * a_k^H [ (R+Rhat)/2 ]^{-1} a_k - 0.5 * a_k^H R^{-1} a_k.
    Uses multi-RHS solves for efficiency.
    """
    with torch.enable_grad():
        p_ = p.detach().requires_grad_(True)
        loss = jbl_loss(A, p_, sigma2, Rhat)
        g, = torch.autograd.grad(loss, p_)
    return g


def weights_cccp(A: Tensor, p: Tensor, sigma2: float, Rhat: Tensor) -> Tensor:
    """Compute CCCP linearization weights:
    w_k = 0.5 * a_k^H * [ (R(p)+Rhat)/2 ]^{-1} * a_k.
    Vectorized via solving Rbar X = A, then w = 0.5 * Re(diag(A^H X)).
    """
    R = build_R(A, p, sigma2)
    Rbar = 0.5 * (R + Rhat)
    # Solve Rbar X = A for all columns as RHS
    _, X = cholesky_logdet_and_solve(Rbar, A)
    # diag(A^H X) = sum(conj(A) * X, dim=0)
    w = 0.5 * (A.conj() * X).sum(dim=0).real
    return w.detach()  # treat as constant in inner problem


###############################################################################
# Normalization (optional)
###############################################################################

def maybe_normalize(A: Tensor, Rhat: Tensor, sigma2: float, normalize: bool = True) -> Tuple[Tensor, Tensor, float, Optional[Tensor]]:
    """Column-normalize A (\ell2) and optionally rescale Rhat/sigma2 so that
    mean diagonal power is ~1. Returns (A_n, Rhat_n, sigma2_n, col_norms).

    If normalize is False, returns inputs and col_norms=None.
    """
    if not normalize:
        return A, Rhat, sigma2, None

    # Column norms
    col_norms = torch.linalg.vector_norm(A, ord=2, dim=0)
    col_norms = torch.clamp(col_norms, min=1e-12)
    A_n = A / col_norms.unsqueeze(0)

    # Rescale Rhat and sigma2 to unit average diagonal (optional but helpful)
    mean_diag = Rhat.diagonal().real.mean()
    if mean_diag <= 0:
        scale = 1.0
    else:
        scale = (1.0 / mean_diag).real
    Rhat_n = Rhat * scale
    sigma2_n = sigma2 * float(scale)

    return A_n, Rhat_n, sigma2_n, col_norms


###############################################################################
# 1) Projected Gradient with Barzilai–Borwein (BB) + optional Armijo
###############################################################################

def projected_gradient_bb(
    A: Tensor,
    Rhat: Tensor,
    sigma2: float,
    p_init: Tensor,
    max_iters: int = 500,
    tol_rel: float = 1e-6,
    armijo: bool = False,
    armijo_c: float = 1e-4,
    armijo_beta: float = 0.5,
    bb_epsilon: float = 1e-12,
    normalize: bool = True,
) -> Dict[str, Tensor]:
    """Projected gradient descent with BB step sizes.

    If normalization is enabled, the routine optimizes in the normalized space
    (A', Rhat', sigma2'), and returns p in the *original* units by inverting
    the column scaling: p_out_k = p'_k / ||a_k||^2.
    """
    device = A.device
    dtype = A.dtype

    A_n, Rhat_n, sigma2_n, col_norms = maybe_normalize(A, Rhat, sigma2, normalize)

    # Map initial p to normalized coordinates if needed: p' = p * ||a||^2
    if col_norms is not None:
        p = (p_init * (col_norms ** 2)).to(device=device, dtype=torch.float64)
    else:
        p = p_init.to(device=device, dtype=torch.float64)

    A_n = A_n.to(device=device, dtype=dtype)
    Rhat_n = Rhat_n.to(device=device, dtype=dtype)

    project_nonneg_(p)

    # Compute initial gradient
    p.requires_grad_(True)
    loss = jbl_loss(A_n, p, sigma2_n, Rhat_n)
    g, = torch.autograd.grad(loss, p)
    p = p.detach()

    t = 1.0  # initial stepsize
    prev_p, prev_g = None, None

    history = []

    for it in range(1, max_iters + 1):
        # BB stepsize (use BB1 or fallback)
        if prev_p is not None:
            s = p - prev_p
            y = g - prev_g
            sy = torch.dot(s, y)
            yy = torch.dot(y, y)
            if torch.abs(sy) > bb_epsilon and yy > bb_epsilon:
                # Classic BB1: t = (s^T s)/(s^T y)
                t = (s.dot(s) / sy).item()
                # Safeguards
                t = float(max(min(t, 1e6), 1e-12))
            else:
                t = 1.0

        # Gradient step + projection
        trial_p = p - t * g
        project_nonneg_(trial_p)

        if armijo:
            # Backtracking Armijo on the *projected* point
            # Define phi(alpha) = L(P[p - alpha * g])
            alpha = 1.0
            Lp = loss.detach()
            # directional derivative surrogate: g^T (p - P[p - alpha g]) / alpha
            # Use standard Armijo with recomputed loss.
            while True:
                cand = p - alpha * t * g
                cand = cand.clamp(min=0)
                cand.requires_grad_(True)
                cand_loss = jbl_loss(A_n, cand, sigma2_n, Rhat_n)
                # Armijo condition: f(cand) <= f(p) - c * alpha * t * ||g||^2
                if cand_loss <= Lp - armijo_c * alpha * t * (g @ g):
                    trial_p = cand.detach()
                    loss = cand_loss.detach()
                    break
                alpha *= armijo_beta
                if alpha < 1e-8:
                    # accept the non-backtracked step
                    trial_p = trial_p.detach()
                    # recompute loss at trial_p
                    trial_p.requires_grad_(True)
                    loss = jbl_loss(A_n, trial_p, sigma2_n, Rhat_n)
                    trial_p = trial_p.detach()
                    break
        else:
            # Recompute loss at trial
            trial_p = trial_p.detach().requires_grad_(True)
            loss = jbl_loss(A_n, trial_p, sigma2_n, Rhat_n)

        # Prepare for next iter: compute new gradient
        g_new, = torch.autograd.grad(loss, trial_p)
        trial_p = trial_p.detach()

        # Save history
        history.append(loss.detach().cpu())

        # Check convergence
        rel = torch.norm(trial_p - p) / (1e-12 + torch.norm(p))
        p, prev_p = trial_p, p
        prev_g, g = g, g_new.detach()
        if rel.item() < tol_rel:
            break

    # Map back to original units: p = p' / ||a||^2
    if col_norms is not None:
        p_out = (p / (col_norms ** 2)).to(dtype=torch.float64)
    else:
        p_out = p

    return {"p": p_out.detach(), "loss_history": torch.stack(history)}


###############################################################################
# 2) CCCP with inner iterations using Adam or (projected) LBFGS
###############################################################################

def inner_objective_cccp(A: Tensor, p: Tensor, sigma2: float, w: Tensor) -> Tensor:
    """Inner convex objective for CCCP:
    F(p) = -0.5 * logdet(R(p)) + w^T p. The linear term uses fixed w.
    """
    with torch.no_grad():
        project_nonneg_(p)

    R = build_R(A, p, sigma2)

    logdet_R, _ = cholesky_logdet_and_solve(R)
    return (-0.5 * logdet_R + (w @ p))


def optimize_JBLD_cccp(
    _A: Tensor,
    _R_hat: Tensor,
    _sigma2: float,
    _p_init: Tensor,
    _max_iter: int = None,
    outer_iters: int = 500,
    inner_iters: int = 20,
    inner_opt: str = "adam",  # "adam" or "lbfgs"
    _lr: float = 0.05,
    line_search_fn_bfgs = None,
    normalize: bool = False,
    do_store_history: bool = False,
    do_verbose: bool = False
) -> Dict[str, Tensor]:
    """CCCP for JBLD: linearize h(p) = -logdet((R+Rhat)/2) and solve
    min_p >= 0 of  g(p) - <w, p>, with g(p) = -0.5 logdet R(p), w fixed.

    inner_opt:
        - "adam": Adam with projection (clamp) after each step.
        - "lbfgs": torch.optim.LBFGS with projection after each step.
          (Note: this is *projected* LBFGS, not true L-BFGS-B.)
    """
    R_hat = torch.as_tensor(_R_hat, dtype=TORCH_DTYPE)
    p_init = torch.as_tensor(_p_init, dtype=torch.float)
    A = torch.as_tensor(_A, dtype=TORCH_DTYPE)

    device = A.device
    dtype = A.dtype

    A_n, Rhat_n, sigma2_n, col_norms = maybe_normalize(A, R_hat, _sigma2, normalize)

    # Map initial p to normalized coordinates if needed
    if col_norms is not None:
        p = (p_init * (col_norms ** 2)).to(device=device, dtype=torch.float)
    else:
        p = p_init.to(device=device, dtype=torch.float)

    A_n = A_n.to(device=device, dtype=dtype)
    Rhat_n = Rhat_n.to(device=device, dtype=dtype)

    project_nonneg_(p)

    loss_history = []
    rel_change_history = []
    i_iter = -1
    p_prev = p.clone().detach()
    # assert that either _max_iter is different from None or outer_iters is different from None
    assert (_max_iter is not None) or (outer_iters is not None), "Must specify either _max_iter or outer_iters"
    _max_iter = outer_iters*inner_iters if _max_iter is None else _max_iter
    while i_iter < _max_iter:
        # 1) Compute CCCP weights at current p (no grad)
        w = weights_cccp(A_n, p, sigma2_n, Rhat_n)  # shape (K,)

        # 2) Solve inner convex problem approximately from warm start p
        p_var = p.clone().detach().requires_grad_(True)

        if inner_opt.lower() == "adam":
            opt = torch.optim.Adam([p_var], lr=_lr)
            for _ in range(inner_iters):
                opt.zero_grad(set_to_none=True)
                loss = inner_objective_cccp(A_n, p_var, sigma2_n, w)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    project_nonneg_(p_var)
        elif inner_opt.lower() == "lbfgs":
            # Projected LBFGS via closure + clamp after step
            opt = torch.optim.LBFGS([p_var], lr=_lr, max_iter=inner_iters, line_search_fn=line_search_fn_bfgs)

            def closure():
                opt.zero_grad(set_to_none=True)
                
                with torch.no_grad():
                    p_var.clamp_(min=0.0)

                loss = inner_objective_cccp(A_n, p_var, sigma2_n, w)
                loss.backward()
                return loss

            opt.step(closure)
            with torch.no_grad():
                project_nonneg_(p_var)
        else:
            raise ValueError("inner_opt must be 'adam' or 'lbfgs'")
        i_iter += inner_iters
        p = p_var.detach()

        rel = torch.norm(p - p_prev) / (1e-5 + torch.norm(p_prev)).item()
        if do_store_history:
            # Track true JBLD after each outer iteration
            p_req = p.clone().detach().requires_grad_(False)
            loss_val = jbl_loss(A_n, p_req, sigma2_n, Rhat_n).detach().cpu()
            loss_history.extend(inner_iters * [loss_val])

            rel_change_history.extend(inner_iters * [rel])
        if rel < EPS_REL_CHANGE:
            break
        p_prev = p.clone()

    # Map back to original coordinates
    if col_norms is not None:
        p_out = (p / (col_norms ** 2)).to(dtype=torch.float64)
    else:
        p_out = p

    return p_out.detach(), i_iter, (loss_history, rel_change_history)

