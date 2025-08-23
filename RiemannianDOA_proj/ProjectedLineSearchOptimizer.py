import torch
import copy

class ProjectedLineSearchOptimizer:
    """
    Wrapper for torch optimizers that adds:
      1. Projection onto nonnegative orthant
      2. Backtracking line search with Armijo condition
      3. General support for SGD/Adam/RMSProp/L-BFGS (any torch optimizer)

    Args:
        params (iterable): parameters to optimize (as in torch.optim)
        base_optimizer_cls: class of the underlying torch optimizer (e.g. torch.optim.SGD)
        base_optimizer_kwargs: kwargs for the base optimizer (lr, momentum, etc.)
        loss_fn: function that computes the loss (for gradients)
        eval_fn: function to evaluate in line search (can be f or tilde{f})
        c (float): Armijo parameter in (0,1)
        tau (float): step size shrink factor in (0,1)
        max_ls_steps (int): maximum line search backtracking steps
    """

    def __init__(self, params, base_optimizer_cls, base_optimizer_kwargs,
                 loss_fn, eval_fn=None, c=1e-4, tau=0.5, max_ls_steps=20):
        self.params = list(params)
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn if eval_fn is not None else loss_fn
        self.c = c
        self.tau = tau
        self.max_ls_steps = max_ls_steps

        # Keep the user-chosen optimizer and lr
        self.base_optimizer = base_optimizer_cls(self.params, **base_optimizer_kwargs)
        self.base_lr = base_optimizer_kwargs.get("lr", 1.0)

        self.alpha_vals = [self.base_lr]*self.max_ls_steps
        for i_ls in range(1,self.max_ls_steps):
            if i_ls%2 == 0:
                self.alpha_vals[i_ls] = self.alpha_vals[i_ls-2]*0.1 
            else:
                self.alpha_vals[i_ls] = self.alpha_vals[i_ls-1]*0.5 


        # A clone of the optimizer but with lr=1.0, to extract directions
        opt_kwargs_lr1 = copy.deepcopy(base_optimizer_kwargs)
        opt_kwargs_lr1["lr"] = 1.0
        self.dir_optimizer = base_optimizer_cls(self.params, **opt_kwargs_lr1)

    def step(self):
        """
        Perform one optimization step:
          1. Compute direction from optimizer (lr=1)
          2. Backtracking line search starting from base_lr
          3. Projection onto nonnegative orthant
        """

        # Store current parameters
        with torch.no_grad():
            old_params = [p.clone() for p in self.params]

        # --- compute loss and gradient ---
        loss = self.loss_fn()
        loss.backward()
        old_loss = self.eval_fn().item()

        # --- get direction using optimizer with lr=1 ---
        with torch.no_grad():
            # Temporarily apply dir_optimizer step
            self.dir_optimizer.step()

            # Direction = new - old
            direction = [p.clone() - p_old for p, p_old in zip(self.params, old_params)]

            # Restore original parameters
            for p, p_old in zip(self.params, old_params):
                p.copy_(p_old)

        # --- line search ---
        alpha = self.base_lr
        grad_dot_dir = 0.0
        for g, d in zip([p.grad for p in self.params], direction):
            if g is not None:
                grad_dot_dir += (g * d).sum().item()

        success = False
        no_ls_new_loss = None
        for i_ls_step in range(self.max_ls_steps):
            with torch.no_grad():
                # Trial step
                for p, p_old, d in zip(self.params, old_params, direction):
                    p.copy_(p_old + alpha * d)

                # Projection: enforce nonnegativity
                for p in self.params:
                    p.clamp_(min=0.0)

            # Evaluate objective on trial point
            new_loss = self.eval_fn().item()
            if i_ls_step == 0:
                no_ls_new_loss = new_loss
            # Armijo condition: f(x+αd) ≤ f(x) + c α ∇f·d
            if new_loss <= old_loss + self.c * alpha * grad_dot_dir:
                success = True
                break
            alpha *= self.tau
        do_print = False
        if success:
            if do_print:
                print(f"[LineSearch] num steps={i_ls_step+1}, selected_alpha={alpha:.6f}, old_loss={old_loss:.6e}, new_loss={new_loss:.6e}")
        else:
            # use step without linesearch
            with torch.no_grad():
                for p, p_old, d in zip(self.params, old_params, direction):
                    p.copy_(p_old + self.base_lr * d)

                for p in self.params:
                    p.clamp_(min=0.0)
            if do_print:
                print(f"[LineSearch] failed. num steps={i_ls_step+1}, old_loss={old_loss:.6e}, new_loss={no_ls_new_loss:.6e}")

        

        # Clear gradients (since we did .backward())
        self.zero_grad()

    def zero_grad(self):
        self.base_optimizer.zero_grad()
        self.dir_optimizer.zero_grad()
