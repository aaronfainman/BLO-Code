import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from copy import deepcopy

from torch.func import jvp, grad, functional_call, vjp




# =============================================================================
#  PABLO  (Preconditioned Augmented Lagrangian BiLevel Optimisation)
# =============================================================================
#  Reference: BLO_Proof___prox.pdf, Algorithm 1.
#
#      K(W, theta, u, y_W, y_theta)
#          = F + u^T grad_W J + (rho/2)||grad_W J||^2
#            + (tau/2)||W - y_W||^2 + (tau/2)||theta - y_theta||^2
#
#      grad_W K     = grad_W F(xi_3)     + grad^2_W J(xi_2)      (u + rho grad_W J(xi_1)) + tau(W - y_W)
#      grad_theta K = grad_theta F(xi_3) + grad^2_{W theta} J(xi_2)^T (u + rho grad_W J(xi_1)) + tau(theta - y_theta)
#      p_i     = beta p_{i-1} + (1 - beta)(grad K)^2                       [RMSProp]
#      Z_i     = Z_{i-1} - mu grad_Z K / sqrt(p_i + eps)
#      u_i     = P_{B_R}( u_{i-1} + eta grad_W J(W_i, theta_i) )           [NEW iterate]
#      y_i     = y_{i-1} + gamma tau (Z_i - y_{i-1})
#
#  Three classes, forming the ablation ladder:
#      BLO_optimiser_PABLO             L only              (no prox, no projection)
#      BLO_optimiser_PABLO_Prox        L + proximal term
#      BLO_optimiser_PABLO_Prox_Bound  L + proximal + projection of u onto B_R   <- Algorithm 1
#
#  All three share the same sampling and multiplier timing so that the ablation
#  isolates the proximal term and the projection.
#
#  SAMPLING. Lemma 1 factorises E[grad^2 J(xi_2)(u + rho grad J(xi_1))] using the
#  independence of xi_1 and xi_2, so the two must be different samples. Each step
#  therefore draws xi_1 (LL, gradient vector), xi_2 (LL, Hessian / mixed Jacobian),
#  xi_3 (UL) and one further LL sample for the multiplier update. Under
#  stochastic=False the loss functions ignore the seed, so all four coincide with
#  the full batch and the behaviour is the deterministic one.
#
#  HYPERPARAMETERS. The paper writes the proximal term as (tau/2)||W - y_W||^2 and
#  the reference update as y <- y + gamma tau (Z - y). The older code used
#  prox_penalty_param ||W - y_W||^2 and y <- y + lr_prox (Z - y), i.e.
#      tau = 2 prox_penalty_param        gamma = lr_prox / tau
#  Both spellings are accepted; passing tau/gamma is preferred because those are
#  the names that appear in the paper.
# =============================================================================


class BLO_optimiser_PABLO:
    """PABLO without the proximal term and without the multiplier projection.

    Augmented Lagrangian only:  L = F + u^T grad_W J + (rho/2)||grad_W J||^2.
    See the section header above for the update equations and the sampling scheme.
    """

    def __init__(self,
                 ll_model: nn.Module,
                 lower_loss_fn: Callable,
                 upper_loss_fn: Callable,
                 ul_model: nn.Module | None = None,
                 stochastic: bool = None,
                 device: torch.device = torch.device("cpu"),
                 debug: bool = False,
                 w_optimiser: torch.optim.Optimizer = torch.optim.RMSprop,
                 theta_optimiser: torch.optim.Optimizer = torch.optim.RMSprop,
                 lambda_optimiser: torch.optim.Optimizer = torch.optim.SGD,
                 lr_w: float = 1e-3,
                 lr_theta: float = 1e-3,
                 rho: float = 1e-3,
                 lr_lambda: float = 1e-3,
                 w_optimiser_kwargs: dict | None = None,
                 theta_optimiser_kwargs: dict | None = None,
                 lambda_optimiser_kwargs: dict | None = None,
                 total_steps: int | None = None,
                 warmup_ratio: float = 0.1,
                 lambda_init: list | None = None,
                 eta_min: float = 0.0,
                 grad_clip: float | None = None):

        self.ll_model = ll_model
        self.ul_model = ul_model
        self.lower_loss_fn = lower_loss_fn
        self.upper_loss_fn = upper_loss_fn

        self.w_params = list(ll_model.parameters())
        self.theta_params = list(ul_model.parameters()) if ul_model else []

        if lambda_init is not None:
            self.lamda_params = [nn.Parameter(lm) if not isinstance(lm, nn.Parameter) else lm
                                 for lm in lambda_init]
        else:
            self.lamda_params = [nn.Parameter(torch.zeros_like(w.flatten()))
                                 for w in self.w_params]
        self.rho = rho
        self.lr_lambda = lr_lambda
        self.grad_clip = grad_clip

        self.w_optimiser      = w_optimiser(self.w_params,     lr=lr_w,      **(w_optimiser_kwargs      or {}))
        self.theta_optimiser  = theta_optimiser(self.theta_params, lr=lr_theta, **(theta_optimiser_kwargs or {}))
        self.lambda_optimiser = lambda_optimiser(self.lamda_params, lr=lr_lambda, maximize=True,
                                                 **(lambda_optimiser_kwargs or {}))

        self.scheduler_w     = None
        self.scheduler_theta = None

        self.stochastic = stochastic
        self.device = device
        self.iter_counter = 0
        self.debug = debug

        self.g_seed  = 0   # xi_1
        self.gg_seed = 3   # xi_2
        self.f_seed  = 1   # xi_3
        self.u_seed  = 2   # LL sample for the multiplier update

        if total_steps is not None:
            warmup_steps = max(1, int(warmup_ratio * total_steps))
            cosine_steps = max(1, total_steps - warmup_steps)
            for optimiser, attr in [(self.w_optimiser, 'scheduler_w'), (self.theta_optimiser, 'scheduler_theta')]:
                warmup  = torch.optim.lr_scheduler.LinearLR(
                    optimiser, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
                cosine  = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser, T_max=cosine_steps, eta_min=eta_min)
                setattr(self, attr, torch.optim.lr_scheduler.SequentialLR(
                    optimiser, schedulers=[warmup, cosine], milestones=[warmup_steps]))

    # ------------------------------------------------------------------ helpers

    def _draw_seeds(self):
        """xi_1, xi_2, xi_3 and the multiplier sample, all distinct."""
        if self.stochastic:
            base = 4 * self.iter_counter
            self.g_seed  = base + 1
            self.gg_seed = base + 2
            self.f_seed  = base + 3
            self.u_seed  = base + 4

    @staticmethod
    def _filled(grads, params):
        return [g.detach() if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, params)]

    def _lagrangian_grads(self, data, **kwargs):
        """grad_W L, grad_theta L and the pieces needed for logging.

        Returns (grad_W_L, grad_theta_L, ll_loss, ul_loss, grad_W_J_xi1).
        """
        # --- xi_3: upper level
        ul_loss = self.upper_loss_fn(self.ll_model, data, self.ul_model,
                                     self.stochastic, self.f_seed, **kwargs)
        gF = torch.autograd.grad(ul_loss, self.w_params + self.theta_params, allow_unused=True)
        n_w = len(self.w_params)
        grad_W_F  = self._filled(gF[:n_w], self.w_params)
        grad_th_F = self._filled(gF[n_w:], self.theta_params)

        # --- xi_1: the grad_W J that enters the vector (u + rho grad_W J)
        ll_loss = self.lower_loss_fn(self.ll_model, data, self.ul_model,
                                     self.stochastic, self.g_seed, **kwargs)
        # retain_graph: under stochastic=False the same ll_loss is reused for xi_2 below
        g1 = self._filled(torch.autograd.grad(ll_loss, self.w_params,
                                              retain_graph=True, allow_unused=True),
                          self.w_params)

        # v = u + rho grad_W J(xi_1)
        v = [u.detach().view_as(w) + self.rho * g
             for u, g, w in zip(self.lamda_params, g1, self.w_params)]

        # --- xi_2: the sample supplying grad^2_W J and grad^2_{W theta} J
        if self.stochastic:
            ll_loss_2 = self.lower_loss_fn(self.ll_model, data, self.ul_model,
                                           self.stochastic, self.gg_seed, **kwargs)
        else:
            ll_loss_2 = ll_loss
        g2 = torch.autograd.grad(ll_loss_2, self.w_params, create_graph=True,
                                 retain_graph=True, allow_unused=True)

        live = [i for i, g in enumerate(g2) if g is not None and g.requires_grad]
        if live:
            outs = [g2[i] for i in live]
            vs   = [v[i] for i in live]
            hvp_W = torch.autograd.grad(outs, self.w_params, grad_outputs=vs,
                                        retain_graph=True, allow_unused=True)
            hvp_th = (torch.autograd.grad(outs, self.theta_params, grad_outputs=vs,
                                          retain_graph=True, allow_unused=True)
                      if self.theta_params else [])
        else:
            hvp_W  = [None] * len(self.w_params)
            hvp_th = [None] * len(self.theta_params)

        hvp_W  = self._filled(hvp_W,  self.w_params)
        hvp_th = self._filled(hvp_th, self.theta_params)

        grad_W_L  = [f + h for f, h in zip(grad_W_F,  hvp_W)]
        grad_th_L = [f + h for f, h in zip(grad_th_F, hvp_th)]

        return grad_W_L, grad_th_L, ll_loss, ul_loss, g1

    def _apply_primal(self, grad_W_K, grad_th_K):
        """Write grad_Z K onto .grad and step. grad_clip, if set, rescales the JOINT
        gradient over Z = (W, theta) so that ||grad_Z K|| <= grad_clip."""
        if self.grad_clip is not None:
            total = torch.sqrt(sum((g.detach() ** 2).sum()
                                   for g in list(grad_W_K) + list(grad_th_K)))
            if total > self.grad_clip:
                scale = self.grad_clip / (total + 1e-12)
                grad_W_K  = [g * scale for g in grad_W_K]
                grad_th_K = [g * scale for g in grad_th_K]

        self.w_optimiser.zero_grad(set_to_none=True)
        self.theta_optimiser.zero_grad(set_to_none=True)
        for p, g in zip(self.w_params, grad_W_K):
            p.grad = g.detach().clone()
        for p, g in zip(self.theta_params, grad_th_K):
            p.grad = g.detach().clone()
        self.w_optimiser.step()
        self.theta_optimiser.step()

    def _update_multiplier(self, data, **kwargs):
        """u_i = u_{i-1} + eta grad_W J(W_i, theta_i), evaluated at the NEW iterate."""
        ll_loss_u = self.lower_loss_fn(self.ll_model, data, self.ul_model,
                                       self.stochastic, self.u_seed, **kwargs)
        gu = self._filled(torch.autograd.grad(ll_loss_u, self.w_params, allow_unused=True),
                          self.w_params)
        self.lambda_optimiser.zero_grad(set_to_none=True)
        for u, g in zip(self.lamda_params, gu):
            u.grad = g.flatten().clone()
        self.lambda_optimiser.step()   # maximize=True  ->  u += eta * grad
        return gu

    def _step_schedulers(self):
        if self.scheduler_w     is not None: self.scheduler_w.step()
        if self.scheduler_theta is not None: self.scheduler_theta.step()

    def _lagrangian_value(self, ul_loss, g1):
        """L(Z_{i-1}, u_{i-1}) = F + u^T grad_W J + (rho/2)||grad_W J||^2."""
        with torch.no_grad():
            val = ul_loss.detach().clone()
            for u, g in zip(self.lamda_params, g1):
                val = val + (u.detach() * g.flatten()).sum() + (self.rho / 2) * g.pow(2).sum()
        return val

    def _lambda_norm(self):
        with torch.no_grad():
            return torch.sqrt(sum((lm.detach() ** 2).sum() for lm in self.lamda_params)).item()

    # --------------------------------------------------------------------- step

    def step(self, data: Optional[Dict] = None, **kwargs):
        self.iter_counter += 1
        self._draw_seeds()

        grad_W_L, grad_th_L, ll_loss, ul_loss, g1 = self._lagrangian_grads(data, **kwargs)
        aug_loss = self._lagrangian_value(ul_loss, g1)

        grad_w_L_norm     = sum(g.norm().item() for g in grad_W_L)
        grad_theta_L_norm = sum(g.norm().item() for g in grad_th_L)

        self._apply_primal(grad_W_L, grad_th_L)
        self._update_multiplier(data, **kwargs)
        self._step_schedulers()

        if self.debug:
            print(f'{self.iter_counter}: LL={ll_loss.item():.4f}  UL={ul_loss.item():.4f}  '
                  f'aug={aug_loss.item():.4f}  '
                  f'|grad_W L|={grad_w_L_norm:.4f}  |grad_theta L|={grad_theta_L_norm:.4f}  '
                  f'|u|={self._lambda_norm():.4f}  '
                  f'lr_w={self.w_optimiser.param_groups[0]["lr"]:.2e}')

        return {
            'll_loss':           ll_loss.item(),
            'ul_loss':           ul_loss.item(),
            'aug_loss':          aug_loss.item(),
            'grad_w_L_norm':     grad_w_L_norm,
            'grad_theta_L_norm': grad_theta_L_norm,
            'lambda_norm':       self._lambda_norm(),
        }


class BLO_optimiser_PABLO_Prox(BLO_optimiser_PABLO):
    """PABLO with the proximal term, without the multiplier projection.

        K = L + (tau/2)||W - y_W||^2 + (tau/2)||theta - y_theta||^2
        y <- y + gamma tau (Z - y)      applied after the primal step

    tau/gamma are the paper's names; prox_penalty_param = tau/2 and
    lr_prox = gamma tau are the older spelling and are still accepted.
    """

    def __init__(self, *args,
                 lr_prox: float = 1e-3,
                 prox_penalty_param: float = 1e-3,
                 tau: float | None = None,
                 gamma: float | None = None,
                 y_init: list | None = None,
                 prox_seed: int | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # tau/gamma take precedence; otherwise convert from the older spelling.
        if tau is None:
            tau = 2.0 * prox_penalty_param
        if gamma is None:
            gamma = (lr_prox / tau) if tau > 0 else 0.0

        self.tau = tau
        self.gamma = gamma
        self.prox_penalty_param = tau / 2.0   # backwards-compatible aliases
        self.lr_prox = gamma * tau

        # y_0 = Z_0. These MUST be detached: if y is an autograd node built from the
        # parameters then d/dW (tau/2)||W - y_W||^2 = tau(W - y_W)(1 - dy_W/dW) = 0
        # and the proximal term contributes nothing at all.
        z_params = self.w_params + self.theta_params
        if y_init is not None:
            if len(y_init) != len(z_params):
                raise ValueError(f"y_init contains {len(y_init)} tensors, but z contains "
                                 f"{len(z_params)} parameter tensors.")
            y_all = []
            for y, z in zip(y_init, z_params):
                y_t = torch.as_tensor(y, device=z.device, dtype=z.dtype).detach().clone()
                if y_t.shape != z.shape:
                    raise ValueError(f"y shape {y_t.shape} does not match z shape {z.shape}.")
                y_all.append(y_t)
        else:
            y_all = [z.detach().clone() for z in z_params]

        n_w = len(self.w_params)
        self.y_params_w     = y_all[:n_w]
        self.y_params_theta = y_all[n_w:]

        # Accepted for signature compatibility. y_0 = Z_0 is deterministic, so this is
        # unused; it previously reseeded the global torch RNG as a side effect.
        self.prox_seed = prox_seed

    def _prox_terms(self, grad_W_L, grad_th_L):
        """Add tau(Z - y) and return (grad_W K, grad_theta K, ||Z - y||)."""
        grad_W_K  = [g + self.tau * (w - y)
                     for g, w, y in zip(grad_W_L, self.w_params, self.y_params_w)]
        grad_th_K = [g + self.tau * (t - y)
                     for g, t, y in zip(grad_th_L, self.theta_params, self.y_params_theta)]
        with torch.no_grad():
            sq = sum((w - y).pow(2).sum() for w, y in zip(self.w_params, self.y_params_w))
            sq = sq + sum((t - y).pow(2).sum() for t, y in zip(self.theta_params, self.y_params_theta))
        return grad_W_K, grad_th_K, sq.sqrt().item()

    def _update_reference(self):
        """y_i = y_{i-1} + gamma tau (Z_i - y_{i-1})."""
        rate = self.gamma * self.tau
        with torch.no_grad():
            for z, y in zip(self.w_params, self.y_params_w):
                y.add_(rate * (z.detach() - y))
            for z, y in zip(self.theta_params, self.y_params_theta):
                y.add_(rate * (z.detach() - y))

    def _project_multiplier(self):
        """No projection in this variant."""
        return

    def step(self, data: Optional[Dict] = None, **kwargs):
        self.iter_counter += 1
        self._draw_seeds()

        grad_W_L, grad_th_L, ll_loss, ul_loss, g1 = self._lagrangian_grads(data, **kwargs)
        grad_W_K, grad_th_K, z_y_norm = self._prox_terms(grad_W_L, grad_th_L)

        aug_loss = self._lagrangian_value(ul_loss, g1)
        prox_loss = (self.tau / 2) * z_y_norm ** 2
        K_loss = aug_loss.item() + prox_loss

        grad_w_L_norm     = sum(g.norm().item() for g in grad_W_K)
        grad_theta_L_norm = sum(g.norm().item() for g in grad_th_K)

        self._apply_primal(grad_W_K, grad_th_K)
        self._update_reference()
        self._update_multiplier(data, **kwargs)
        self._project_multiplier()
        self._step_schedulers()

        if self.debug:
            print(f'{self.iter_counter}: LL={ll_loss.item():.4f}  UL={ul_loss.item():.4f}  '
                  f'aug={aug_loss.item():.4f}  K={K_loss:.4f}  prox={prox_loss:.4f}  '
                  f'|Z-y|={z_y_norm:.4f}  '
                  f'|grad_W K|={grad_w_L_norm:.4f}  |grad_theta K|={grad_theta_L_norm:.4f}  '
                  f'|u|={self._lambda_norm():.4f}  '
                  f'lr_w={self.w_optimiser.param_groups[0]["lr"]:.2e}')

        return {
            'll_loss':           ll_loss.item(),
            'ul_loss':           ul_loss.item(),
            'aug_loss':          aug_loss.item(),
            'K_loss':            K_loss,
            'proximal_loss':     prox_loss,
            'z_y_norm':          z_y_norm,
            'grad_w_L_norm':     grad_w_L_norm,
            'grad_theta_L_norm': grad_theta_L_norm,
            'lambda_norm':       self._lambda_norm(),
        }


class BLO_optimiser_PABLO_Prox_Bound(BLO_optimiser_PABLO_Prox):
    """PABLO exactly as in Algorithm 1: proximal term + projection of u onto B_R.

    This is the canonical version. The projection is what makes ||u|| <= R hold, which
    Lemma 1 step (c) and the Lipschitz constant delta_K = delta_F + R delta_JJ + ...
    of Lemma 2 both rely on.
    """

    def __init__(self, *args, lambda_bound: float = 1e1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_bound = lambda_bound

    def _project_multiplier(self):
        """P_{B_R}(u): rescale the whole multiplier vector if ||u||_2 > R."""
        with torch.no_grad():
            total = torch.sqrt(sum((lm ** 2).sum() for lm in self.lamda_params))
            if total > self.lambda_bound:
                scale = self.lambda_bound / total
                for lm in self.lamda_params:
                    lm.mul_(scale)


# class BLO_optimiser_PABLO_Prox:
#     """PABLO with Adam + optional warmup/cosine schedule for w and θ.

#     Augmented Lagrangian: L_aug = F + λᵀ ∇_w g + (ρ/2) ‖∇_w g‖²
#     adam_kwargs  — passed to both Adam optimizers (betas, eps, weight_decay, …)
#     total_steps  — required to use warmup/cosine schedule
#     warmup_ratio — fraction of total_steps used for linear LR warmup (default 0.1)
#     eta_min      — cosine schedule floor (default 0)

#     Stochastic gradients: pass stochastic=True; seeds are varied each iteration so
#     the lower- and upper-level samples differ from the HVP sample automatically.
#     """

#     def __init__(self,
#                  ll_model: nn.Module,
#                  lower_loss_fn: Callable,
#                  upper_loss_fn: Callable,
#                  ul_model: nn.Module | None = None,
#                  stochastic: bool = None,
#                  device: torch.device = torch.device("cpu"),
#                  debug: bool = False,
#                  w_optimiser: torch.optim.Optimizer = torch.optim.Adam,
#                  theta_optimiser: torch.optim.Optimizer = torch.optim.Adam,
#                  lambda_optimiser: torch.optim.Optimizer = torch.optim.SGD,
#                  lr_w: float = 1e-3,
#                  lr_theta: float = 1e-3,
#                  rho: float = 1e-3,
#                  lr_lambda: float = 1e-3,
#                  lr_prox: float=1e-3,
#                  prox_penalty_param: float=1e-3,
#                  w_optimiser_kwargs: dict | None = None,
#                  theta_optimiser_kwargs: dict | None = None,
#                  lambda_optimiser_kwargs: dict | None = None,
#                  total_steps: int | None = None,
#                  warmup_ratio: float = 0.1,
#                  lambda_init: list | None = None,
#                  eta_min: float = 0.0):

#         self.ll_model = ll_model
#         self.ul_model = ul_model
#         self.lower_loss_fn = lower_loss_fn
#         self.upper_loss_fn = upper_loss_fn

#         self.w_params = list(ll_model.parameters()) # variable w
#         self.theta_params = list(ul_model.parameters()) if ul_model else [] # variable theta
#         self.y_params_w = [torch.randn_like(p) for p in self.w_params]
#         self.y_params_theta = [torch.randn_like(p) for p in self.theta_params]

#         if lambda_init is not None:
#             self.lamda_params = [nn.Parameter(lm) if not isinstance(lm, nn.Parameter) else lm
#                                  for lm in lambda_init]
#         else:
#             self.lamda_params = [nn.Parameter(torch.zeros_like(w.flatten()))
#                                  for w in self.w_params]
#         self.rho = rho
#         self.lr_lambda = lr_lambda
#         self.lr_prox = lr_prox
#         self.prox_penalty_param = prox_penalty_param

#         self.w_optimiser      = w_optimiser(self.w_params,    lr=lr_w,     **(w_optimiser_kwargs      or {}))
#         self.theta_optimiser  = theta_optimiser(self.theta_params, lr=lr_theta, **(theta_optimiser_kwargs  or {}))
#         self.lambda_optimiser = lambda_optimiser(self.lamda_params, lr=lr_lambda, maximize=True,
#                                                  **(lambda_optimiser_kwargs or {}))


#         self.scheduler_w     = None
#         self.scheduler_theta = None

#         self.stochastic = stochastic
#         self.device = device
#         self.iter_counter = 0
#         self.debug = debug

#         self.g_seed = 0
#         self.gg_seed = 3
#         self.f_seed = 1

#         if total_steps is not None:
#             warmup_steps = max(1, int(warmup_ratio * total_steps))
#             cosine_steps = max(1, total_steps - warmup_steps)
#             for optimiser, attr in [(self.w_optimiser, 'scheduler_w'), (self.theta_optimiser, 'scheduler_theta')]:
#                 warmup  = torch.optim.lr_scheduler.LinearLR(
#                     optimiser, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
#                 cosine  = torch.optim.lr_scheduler.CosineAnnealingLR(
#                     optimiser, T_max=cosine_steps, eta_min=eta_min)
#                 setattr(self, attr, torch.optim.lr_scheduler.SequentialLR(
#                     optimiser, schedulers=[warmup, cosine], milestones=[warmup_steps]))

#     def step(self,data: Optional[Dict] = None, **kwargs):
#         self.iter_counter += 1

#         if self.stochastic:
#             self.g_seed  = self.iter_counter + 1
#             self.gg_seed = self.iter_counter + 2
#             self.f_seed  = self.iter_counter + self.g_seed + 1

#         # -------------- 1. Calculate losses --------------
#         ll_loss = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic, self.g_seed, **kwargs)
#         ul_loss = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic, self.f_seed, **kwargs)
#         # ∇_w g with graph retained for higher-order backprop through aug_loss
#         grad_w_g = torch.autograd.grad(
#             ll_loss, self.w_params, create_graph=True, retain_graph=True, allow_unused=True)

#         # -------------- Calculate Prox. Augmented Lagrangian Loss --------------
#         # L_aug = F + λᵀ ∇_w g + (ρ/2) ‖∇_w g‖² + (p/2) ||z-y||^2
#         aug_loss = ul_loss
#         for lm, gw in zip(self.lamda_params, grad_w_g):
#             if gw is not None:
#                 aug_loss = aug_loss + (lm * gw.flatten()).sum() + (self.rho / 2) * gw.pow(2).sum()
#         for w, y_w in zip(self.w_params, self.y_params_w):
#             aug_loss = aug_loss + self.prox_penalty_param * (w-y_w).pow(2).sum()
#         for t, y_t in zip(self.theta_params, self.y_params_theta):
#             aug_loss = aug_loss + self.prox_penalty_param * (t-y_t).pow(2).sum()

        
#         # -------------- 2. Update dual variable lambda ----------
#         self.lambda_optimiser.zero_grad()
#         aug_loss.backward(retain_graph=True)
#         self.lambda_optimiser.step()


#         # -------------- 3. URecalculate aug loss with new lambda -----------------
#         # L_aug = F + λᵀ ∇_w g + (ρ/2) ‖∇_w g‖² + (p/2) ||z-y||^2
#         aug_loss = ul_loss
#         for lm, gw in zip(self.lamda_params, grad_w_g):
#             if gw is not None:
#                 aug_loss = aug_loss + (lm * gw.flatten()).sum() + (self.rho / 2) * gw.pow(2).sum()
#         for w, y_w in zip(self.w_params, self.y_params_w):
#             aug_loss = aug_loss + self.prox_penalty_param * (w-y_w).pow(2).sum()
#         for t, y_t in zip(self.theta_params, self.y_params_theta):
#             aug_loss = aug_loss + self.prox_penalty_param * (t-y_t).pow(2).sum()

        

#         # -------------- Update steps --------------
#         self.w_optimiser.zero_grad()
#         self.theta_optimiser.zero_grad()
#         aug_loss.backward(retain_graph=True)

#         # For logging: Read ∇_w L_aug and ∇_θ L_aug from .grad
#         grad_w_L_norm = sum(
#             w.grad.norm().item() for w in self.w_params if w.grad is not None)
#         grad_theta_L_norm = sum(
#             th.grad.norm().item() for th in self.theta_params if th.grad is not None)

#         self.w_optimiser.step()
#         self.theta_optimiser.step()

#         # update prox. var.s
#         with torch.no_grad():
#             for i, (w, y_w) in enumerate(zip(self.w_params, self.y_params_w)):
#                 self.y_params_w[i].copy_(y_w + self.lr_prox * (w - y_w))
#             for i, (t, y_t) in enumerate(zip(self.theta_params, self.y_params_theta)):
#                 self.y_params_theta[i].copy_(y_t + self.lr_prox * (t - y_t))

#         if self.scheduler_w     is not None: self.scheduler_w.step()
#         if self.scheduler_theta is not None: self.scheduler_theta.step()

#         if self.debug:
#             print(f'{self.iter_counter}: LL={ll_loss.item():.4f}  UL={ul_loss.item():.4f}  '
#                   f'aug={aug_loss.item():.4f}  '
#                   f'|∇_w L|={grad_w_L_norm:.4f}  |∇_θ L|={grad_theta_L_norm:.4f}  '
#                   f'lr_w={self.w_optimiser.param_groups[0]["lr"]:.2e}')

#         return {
#             'll_loss':          ll_loss.item(),
#             'ul_loss':          ul_loss.item(),
#             'aug_loss':        aug_loss.item(),
#             'grad_w_L_norm':   grad_w_L_norm,
#             'grad_theta_L_norm': grad_theta_L_norm,
#         }





# =============================================================================
#  COMPARISON ALGORITHMS
# =============================================================================
#  Extracted (2026-09-15) from the summer-student code in ./ShelleyCode so that
#  every optimiser used in the PABLO comparison study lives in one module.
#
#    BLO_optimiser_PABLO_prime  <- ShelleyCode/blo_optimisers.py
#    BLO_optimiser_BOME         <- ShelleyCode/BOME.py
#    BLO_optimiser_TSP          <- ShelleyCode/TSP.py   (tuned/practical variant)
#    BLO_optimiser_TSP_Paper    <- ShelleyCode/TSP.py   (paper-faithful variant)
#    BLO_optimiser_STABLE       <- blo_optimisers_old.py (ported to this API)
#
#  Common API (matches BLO_optimiser_PABLO above):
#      lower_loss_fn(ll_model, data, ul_model, stochastic, seed, **kwargs) -> scalar
#      upper_loss_fn(ll_model, data, ul_model, stochastic, seed, **kwargs) -> scalar
#      opt.step(data, **kwargs) -> dict of scalars
#
#  NOTE ON SEEDING: PABLO/PABLO' derive their own stochastic seeds from an
#  internal iteration counter, whereas BOME/TSP/STABLE take `seed` as an
#  argument to step(). When comparing algorithms under `stochastic=True`, drive
#  them all through the same seed sequence or the minibatch streams will differ.
# =============================================================================


class BLO_optimiser_PABLO_prime:
    """PABLO' with moving reference y.

    Augmented Lagrangian: L_aug = F + lambda^T grad_w g + (rho/2)||grad_w g||^2
    PABLO' objective:     K     = L_aug + (tau/2)||z - y||^2
    After updating:       y    <- y + mu_y (z - y)          where z = (w, theta)
    """

    def __init__(
        self,
        ll_model: nn.Module,
        lower_loss_fn: Callable,
        upper_loss_fn: Callable,
        ul_model: nn.Module | None = None,
        stochastic: bool = None,
        device: torch.device = torch.device("cpu"),
        debug: bool = False,
        w_optimiser: torch.optim.Optimizer = torch.optim.RMSprop,
        theta_optimiser: torch.optim.Optimizer = torch.optim.RMSprop,
        lambda_optimiser: torch.optim.Optimizer = torch.optim.SGD,
        lr_w: float = 1e-3,
        lr_theta: float = 1e-3,
        rho: float = 1e-3,
        lr_lambda: float = 1e-3,
        tau: float = 1e-3,
        mu_y: float = 0.1,
        y_init: list | None = None,
        y_init_scale: float = 1.0,
        w_optimiser_kwargs: dict | None = None,
        theta_optimiser_kwargs: dict | None = None,
        lambda_optimiser_kwargs: dict | None = None,
        total_steps: int | None = None,
        warmup_ratio: float = 0.1,
        lambda_init: list | None = None,
        eta_min: float = 0.0,
    ):
        self.ll_model = ll_model
        self.ul_model = ul_model
        self.lower_loss_fn = lower_loss_fn
        self.upper_loss_fn = upper_loss_fn

        self.w_params = list(ll_model.parameters())
        self.theta_params = list(ul_model.parameters()) if ul_model else []

        # z = (w, theta)
        self.z_params = self.w_params + self.theta_params

        self.rho = rho
        self.tau = tau
        self.mu_y = mu_y
        self.lr_lambda = lr_lambda

        if tau < 0:
            raise ValueError("tau must be non-negative.")
        if mu_y < 0:
            raise ValueError("mu_y must be non-negative.")

        # Initialise lambda
        if lambda_init is not None:
            self.lamda_params = [nn.Parameter(lm) if not isinstance(lm, nn.Parameter) else lm
                                 for lm in lambda_init]
        else:
            self.lamda_params = [nn.Parameter(torch.zeros_like(w.flatten()))
                                 for w in self.w_params]

        # Initialise y_0
        if y_init is not None:
            if len(y_init) != len(self.z_params):
                raise ValueError(f"y_init contains {len(y_init)} tensors, but z contains "
                                 f"{len(self.z_params)} parameter tensors.")
            self.y_params = []
            for y, z in zip(y_init, self.z_params):
                y_tensor = torch.as_tensor(y, device=z.device, dtype=z.dtype).detach().clone()
                if y_tensor.shape != z.shape:
                    raise ValueError(f"y shape {y_tensor.shape} does not match z shape {z.shape}.")
                self.y_params.append(y_tensor)
        else:
            self.y_params = [y_init_scale * torch.randn_like(z) for z in self.z_params]

        self.w_optimiser = w_optimiser(self.w_params, lr=lr_w, **(w_optimiser_kwargs or {}))
        self.theta_optimiser = theta_optimiser(self.theta_params, lr=lr_theta,
                                               **(theta_optimiser_kwargs or {}))
        self.lambda_optimiser = lambda_optimiser(self.lamda_params, lr=lr_lambda, maximize=True,
                                                 **(lambda_optimiser_kwargs or {}))

        self.scheduler_w = None
        self.scheduler_theta = None

        self.stochastic = stochastic
        self.device = device
        self.iter_counter = 0
        self.debug = debug

        self.g_seed = 0
        self.gg_seed = 3
        self.f_seed = 1

        if total_steps is not None:
            warmup_steps = max(1, int(warmup_ratio * total_steps))
            cosine_steps = max(1, total_steps - warmup_steps)
            for optimiser, attr in [(self.w_optimiser, 'scheduler_w'),
                                    (self.theta_optimiser, 'scheduler_theta')]:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimiser, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser, T_max=cosine_steps, eta_min=eta_min)
                setattr(self, attr, torch.optim.lr_scheduler.SequentialLR(
                    optimiser, schedulers=[warmup, cosine], milestones=[warmup_steps]))

    def step(self, data: Optional[Dict] = None, **kwargs):
        self.iter_counter += 1

        if self.stochastic:
            self.g_seed = self.iter_counter + 1
            self.gg_seed = self.iter_counter + 2
            self.f_seed = self.iter_counter + self.g_seed + 1

        ll_loss = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                     self.g_seed, **kwargs)
        ul_loss = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                     self.f_seed, **kwargs)

        # Lower-level stationarity constraint grad_w g
        grad_w_g = torch.autograd.grad(ll_loss, self.w_params, create_graph=True,
                                       retain_graph=True, allow_unused=True)

        # Augmented Lagrangian
        aug_loss = ul_loss
        for lm, gw in zip(self.lamda_params, grad_w_g):
            if gw is not None:
                aug_loss = aug_loss + (lm * gw.flatten()).sum() + (self.rho / 2) * gw.pow(2).sum()

        # Proximal term (tau/2)||z - y||^2
        z_y_squared_norm = sum((z - y).pow(2).sum() for z, y in zip(self.z_params, self.y_params))
        proximal_loss = (self.tau / 2) * z_y_squared_norm
        prime_loss = aug_loss + proximal_loss

        self.w_optimiser.zero_grad()
        self.theta_optimiser.zero_grad()
        self.lambda_optimiser.zero_grad()

        prime_loss.backward()

        grad_w_L_norm = sum(w.grad.norm().item() for w in self.w_params if w.grad is not None)
        grad_theta_L_norm = sum(th.grad.norm().item() for th in self.theta_params
                                if th.grad is not None)
        z_y_norm = z_y_squared_norm.detach().sqrt().item()

        self.w_optimiser.step()
        self.theta_optimiser.step()
        self.lambda_optimiser.step()

        if self.scheduler_w is not None:
            self.scheduler_w.step()
        if self.scheduler_theta is not None:
            self.scheduler_theta.step()

        # Update y last, using the new z
        with torch.no_grad():
            for z, y in zip(self.z_params, self.y_params):
                y.add_(self.mu_y * (z.detach() - y))

        if self.debug:
            print(f"{self.iter_counter}: LL={ll_loss.item():.4f} UL={ul_loss.item():.4f} "
                  f"L_aug={aug_loss.item():.4f} K={prime_loss.item():.4f} "
                  f"prox={proximal_loss.item():.4f} |z-y|={z_y_norm:.4f} "
                  f"|grad_w K|={grad_w_L_norm:.4f} |grad_theta K|={grad_theta_L_norm:.4f}")

        return {
            "ll_loss": ll_loss.item(),
            "ul_loss": ul_loss.item(),
            "aug_loss": aug_loss.item(),
            "prime_loss": prime_loss.item(),
            "proximal_loss": proximal_loss.item(),
            "z_y_norm": z_y_norm,
            "grad_w_L_norm": grad_w_L_norm,
            "grad_theta_L_norm": grad_theta_L_norm,
        }


class BLO_optimiser_BOME:
    """BOME! Bilevel Optimization Made Easy (first-order, fully single-level).

    Each step:
      1. theta_tilde <- T steps of plain GD on g(w, .) started from w   (inner approx.)
      2. q_hat       <- g(w, theta) - g(theta_tilde, theta)             (value-function gap)
      3. phi         <- eta ||grad q_hat||^2      ('grad_norm' barrier)
                        eta max(q_hat, 0)         ('gap' barrier)
      4. lambda      <- max(0, (phi - <grad f, grad q>) / ||grad q||^2)
      5. descend along grad f + lambda grad q   in both w and theta
    """

    def __init__(self, ll_model, ul_model, lower_loss_fn, upper_loss_fn, inner_steps=10,
                 inner_lr=0.01, eta=0.5, lr_w=0.001, lr_theta=0.001,
                 w_optimiser=torch.optim.Adam, theta_optimiser=torch.optim.Adam,
                 barrier="grad_norm", stochastic=False, device=None, eps=1e-12):
        self.ll_model, self.ul_model = ll_model, ul_model
        self.lower_loss_fn, self.upper_loss_fn = lower_loss_fn, upper_loss_fn
        self.inner_steps, self.inner_lr, self.eta = inner_steps, inner_lr, eta
        self.barrier, self.stochastic, self.eps = barrier, stochastic, eps
        self.device = device or next(ll_model.parameters()).device
        self.w_optimizer = w_optimiser(ll_model.parameters(), lr=lr_w)
        self.theta_optimizer = theta_optimiser(ul_model.parameters(), lr=lr_theta)
        self.inner_model = deepcopy(ll_model).to(self.device)
        if barrier not in {"grad_norm", "gap"}:
            raise ValueError("barrier must be 'grad_norm' or 'gap'")

    @staticmethod
    def _grads(loss, parameters):
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        return [torch.zeros_like(parameter) if gradient is None else gradient
                for parameter, gradient in zip(parameters, gradients)]

    def _make_inner_approximation(self, data, seed, **kwargs):
        self.inner_model.load_state_dict(self.ll_model.state_dict())
        self.inner_model.train(self.ll_model.training)
        inner_parameters = list(self.inner_model.parameters())
        for parameter in inner_parameters:
            parameter.requires_grad_(True)

        for inner_iteration in range(self.inner_steps):
            inner_loss = self.lower_loss_fn(self.inner_model, data, self.ul_model,
                                            self.stochastic, seed + inner_iteration, **kwargs)
            inner_gradients = torch.autograd.grad(inner_loss, inner_parameters, allow_unused=True)
            with torch.no_grad():
                for parameter, gradient in zip(inner_parameters, inner_gradients):
                    if gradient is not None:
                        parameter.add_(gradient, alpha=-self.inner_lr)

        # Stop gradient through theta_tilde, but g(theta_tilde, theta) is still
        # differentiated w.r.t. the upper-level model.
        for parameter in inner_parameters:
            parameter.requires_grad_(False)
        return self.inner_model

    def step(self, data=None, seed=0, **kwargs):
        theta_tilde = self._make_inner_approximation(data, seed, **kwargs)

        lower_loss = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                        seed, **kwargs)
        inner_lower_loss = self.lower_loss_fn(theta_tilde, data, self.ul_model, self.stochastic,
                                              seed, **kwargs)
        q_hat = lower_loss - inner_lower_loss
        upper_loss = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                        seed, **kwargs)

        lower_parameters = list(self.ll_model.parameters())
        upper_parameters = list(self.ul_model.parameters())
        parameters = lower_parameters + upper_parameters

        grad_f = self._grads(upper_loss, parameters)
        grad_q = self._grads(q_hat, parameters)

        f_q_inner_product = sum((fg * qg).sum() for fg, qg in zip(grad_f, grad_q))
        q_gradient_squared = sum(qg.square().sum() for qg in grad_q)

        if self.barrier == "grad_norm":
            phi = self.eta * q_gradient_squared
        else:
            phi = self.eta * q_hat.detach().clamp_min(0)

        if q_gradient_squared.detach().item() <= self.eps:
            lambda_k = torch.zeros((), device=self.device)
            lambda_stationarity = torch.zeros((), device=self.device)
        else:
            lambda_k = ((phi - f_q_inner_product) / q_gradient_squared).detach().clamp_min(0)
            lambda_stationarity = (-f_q_inner_product / q_gradient_squared).detach().clamp_min(0)

        direction = [(fg + lambda_k * qg).detach() for fg, qg in zip(grad_f, grad_q)]
        stationarity_direction = [fg.detach() + lambda_stationarity * qg.detach()
                                  for fg, qg in zip(grad_f, grad_q)]

        self.w_optimizer.zero_grad(set_to_none=True)
        self.theta_optimizer.zero_grad(set_to_none=True)
        for parameter, parameter_direction in zip(parameters, direction):
            parameter.grad = parameter_direction.clone()
        self.w_optimizer.step()
        self.theta_optimizer.step()

        grad_f_norm = sum(g.detach().square().sum() for g in grad_f).sqrt()
        grad_q_norm = q_gradient_squared.detach().sqrt()
        K_hat = sum(d.square().sum() for d in stationarity_direction) + q_hat.detach().clamp_min(0)

        return {
            "ll_loss": lower_loss.detach().item(),
            "inner_ll_loss": inner_lower_loss.detach().item(),
            "ul_loss": upper_loss.detach().item(),
            "q_hat": q_hat.detach().item(),
            "lambda": lambda_k.item(),
            "phi": phi.detach().item(),
            "grad_f_norm": grad_f_norm.item(),
            "grad_q_norm": grad_q_norm.item(),
            "K_hat": K_hat.item(),
        }


class BLO_optimiser_TSP:
    """TSP: Moreau-envelope constrained reformulation, practical variant.

    Constraint:  g(x,y) - g(x,z) - ||z-y||^2/(2 gamma) - delta <= 0
    Smoothed Lagrangian: K = f(x,y) + lambda * constraint + (p/2)(||x-xhat||^2 + ||y-yhat||^2)

    This variant differs from `BLO_optimiser_TSP_Paper` by driving x and y through
    torch.optim optimisers (so Adam/RMSProp can be used) instead of plain SGD.
    """

    def __init__(self, ll_model, ul_model, lower_loss_fn, upper_loss_fn, gamma=0.1, delta=0.0,
                 p=1.0, eta_z=0.001, lr_y=0.001, lr_x=0.001, dual_lr=0.01, constraint_ema=0.1,
                 lambda_ema=0.1, anchor_ema=0.1, lambda_init=0.0,
                 y_optimiser=torch.optim.SGD, x_optimiser=torch.optim.SGD,
                 stochastic=False, device=None, eps=1e-12):
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if delta < 0:
            raise ValueError("delta must be nonnegative")
        if p < 0:
            raise ValueError("p must be nonnegative")

        self.ll_model, self.ul_model = ll_model, ul_model
        self.lower_loss_fn, self.upper_loss_fn = lower_loss_fn, upper_loss_fn
        self.gamma, self.delta, self.p, self.eta_z = gamma, delta, p, eta_z
        self.dual_lr, self.constraint_ema = dual_lr, constraint_ema
        self.lambda_ema, self.anchor_ema = lambda_ema, anchor_ema
        self.stochastic, self.eps = stochastic, eps
        self.device = device or next(ll_model.parameters()).device

        self.y_optimizer = y_optimiser(ll_model.parameters(), lr=lr_y)
        self.x_optimizer = x_optimiser(ul_model.parameters(), lr=lr_x)

        self.z_model = deepcopy(ll_model).to(self.device)
        self.y_hat = [p_.detach().clone() for p_ in ll_model.parameters()]
        self.x_hat = [p_.detach().clone() for p_ in ul_model.parameters()]

        self.lambda_value = torch.tensor(float(lambda_init), device=self.device)
        self.h = torch.zeros((), device=self.device)

    @staticmethod
    def _grads(loss, parameters):
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        return [torch.zeros_like(parameter) if gradient is None else gradient
                for parameter, gradient in zip(parameters, gradients)]

    @staticmethod
    def _norm(tensors):
        return sum(t.detach().square().sum() for t in tensors).sqrt()

    def step(self, data=None, seed=0, **kwargs):
        y_parameters = list(self.ll_model.parameters())
        x_parameters = list(self.ul_model.parameters())
        z_parameters = list(self.z_model.parameters())

        # Moreau-gap estimate: g(x,y) - g(x,z) - ||z-y||^2/(2 gamma) - delta
        lower_y_for_gap = self.lower_loss_fn(self.ll_model, data, self.ul_model,
                                            self.stochastic, seed, **kwargs)
        lower_z_for_gap = self.lower_loss_fn(self.z_model, data, self.ul_model,
                                             self.stochastic, seed, **kwargs)
        proximal_value = sum((zp - yp.detach()).square().sum()
                             for zp, yp in zip(z_parameters, y_parameters)) / (2.0 * self.gamma)
        raw_gap = lower_y_for_gap - lower_z_for_gap - proximal_value
        constraint = raw_gap - self.delta

        # h is the moving average of the noisy constraint.
        with torch.no_grad():
            self.h.mul_(1.0 - self.constraint_ema).add_(self.constraint_ema * constraint.detach())
            lambda_plus = torch.clamp(self.lambda_value + self.dual_lr * self.h, min=0.0)
            self.lambda_value.mul_(1.0 - self.lambda_ema).add_(self.lambda_ema * lambda_plus)
            lambda_used = self.lambda_value.clone()

        # z direction: grad_z g(x,z) + (z - y)/gamma
        grad_g_z = self._grads(lower_z_for_gap, z_parameters)
        direction_z = [(gr + (zp - yp.detach()) / self.gamma).detach()
                       for gr, zp, yp in zip(grad_g_z, z_parameters, y_parameters)]

        # y direction: grad_y f + lambda[grad_y g + (z - y)/gamma] + p(y - yhat)
        upper_y = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                     seed, **kwargs)
        lower_y = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                     seed, **kwargs)
        grad_f_y = self._grads(upper_y, y_parameters)
        grad_g_y = self._grads(lower_y, y_parameters)

        direction_y = [
            (fg + lambda_used * (gg + (zp.detach() - yp) / self.gamma) + self.p * (yp - yh)).detach()
            for fg, gg, yp, zp, yh in zip(grad_f_y, grad_g_y, y_parameters, z_parameters, self.y_hat)
        ]

        x_anchor_distance_squared = sum((xp.detach() - xh).square().sum()
                                        for xp, xh in zip(x_parameters, self.x_hat))
        y_anchor_distance_squared = sum((yp.detach() - yh).square().sum()
                                        for yp, yh in zip(y_parameters, self.y_hat))
        z_y_distance_squared = sum((zp.detach() - yp.detach()).square().sum()
                                   for zp, yp in zip(z_parameters, y_parameters))

        K_hat = (upper_y.detach() + lambda_used * constraint.detach()
                 + 0.5 * self.p * (x_anchor_distance_squared + y_anchor_distance_squared))

        self.y_optimizer.zero_grad(set_to_none=True)
        for parameter, direction in zip(y_parameters, direction_y):
            parameter.grad = direction.clone()
        self.y_optimizer.step()

        # y_hat follows the new y
        with torch.no_grad():
            for yh, yp in zip(self.y_hat, y_parameters):
                yh.add_(self.anchor_ema * (yp - yh))

        # x direction uses the newly updated y and the old z:
        # grad_x f + lambda[grad_x g(x,y) - grad_x g(x,z)] + p(x - xhat)
        upper_x = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                     seed, **kwargs)
        lower_xy = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                      seed, **kwargs)
        lower_xz = self.lower_loss_fn(self.z_model, data, self.ul_model, self.stochastic,
                                      seed, **kwargs)

        grad_f_x = self._grads(upper_x, x_parameters)
        grad_g_xy_x = self._grads(lower_xy, x_parameters)
        grad_g_xz_x = self._grads(lower_xz, x_parameters)

        direction_x = [
            (fg + lambda_used * (gyg - gzg) + self.p * (xp - xh)).detach()
            for fg, gyg, gzg, xp, xh in zip(grad_f_x, grad_g_xy_x, grad_g_xz_x,
                                            x_parameters, self.x_hat)
        ]

        self.x_optimizer.zero_grad(set_to_none=True)
        for parameter, direction in zip(x_parameters, direction_x):
            parameter.grad = direction.clone()
        self.x_optimizer.step()

        # Move x_hat and apply the previously computed z step
        with torch.no_grad():
            for xh, xp in zip(self.x_hat, x_parameters):
                xh.add_(self.anchor_ema * (xp - xh))
            for zp, direction in zip(z_parameters, direction_z):
                zp.add_(direction, alpha=-self.eta_z)

        return {
            "ll_loss": lower_y_for_gap.detach().item(),
            "z_ll_loss": lower_z_for_gap.detach().item(),
            "ul_loss": upper_y.detach().item(),
            "moreau_prox": proximal_value.detach().item(),
            "raw_gap": raw_gap.detach().item(),
            "constraint": constraint.detach().item(),
            "h": self.h.item(),
            "lambda": self.lambda_value.item(),
            "z_y_norm": z_y_distance_squared.sqrt().item(),
            "grad_z_norm": self._norm(direction_z).item(),
            "grad_y_K_norm": self._norm(direction_y).item(),
            "grad_x_K_norm": self._norm(direction_x).item(),
            "K_hat": K_hat.item(),
        }


class BLO_optimiser_TSP_Paper:
    """TSP, paper-faithful variant (plain SGD updates, paper's symbol names).

    Hyperparameters follow the paper's notation:
      gamma  Moreau smoothing parameter        delta  constraint slack
      p      proximal (anchor) weight          tau    dual step size
      mu     dual smoothing (lambda EMA)       theta  constraint smoothing (h EMA)
      eta    z step size                       alpha  x step size
      beta   y step size                       omega  anchor EMA
    """

    def __init__(self, ll_model, ul_model, lower_loss_fn, upper_loss_fn, gamma=0.1, delta=0.0,
                 p=1.0, tau=0.01, mu=0.1, theta=0.1, eta=0.001, alpha=0.001, beta=0.001,
                 omega=0.1, lambda_init=0.5, stochastic=False, device=None):
        self.y_model, self.x_model = ll_model, ul_model
        self.lower_loss_fn, self.upper_loss_fn = lower_loss_fn, upper_loss_fn
        self.gamma, self.delta, self.p = gamma, delta, p
        self.tau, self.mu, self.theta = tau, mu, theta
        self.eta, self.alpha, self.beta, self.omega = eta, alpha, beta, omega
        self.stochastic = stochastic
        self.device = device or next(ll_model.parameters()).device

        self.z_model = deepcopy(ll_model).to(self.device)
        self.y_bar = [p_.detach().clone() for p_ in ll_model.parameters()]
        self.x_bar = [p_.detach().clone() for p_ in ul_model.parameters()]
        self.lambda_value = torch.tensor(float(lambda_init), device=self.device)
        self.h = torch.zeros((), device=self.device)

    @staticmethod
    def _grads(loss, parameters):
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        return [torch.zeros_like(parameter) if gradient is None else gradient
                for parameter, gradient in zip(parameters, gradients)]

    @staticmethod
    def _norm(tensors):
        return torch.stack([t.detach().square().sum() for t in tensors]).sum().sqrt()

    def step(self, data=None, seed=0, **kwargs):
        x_parameters = list(self.x_model.parameters())
        y_parameters = list(self.y_model.parameters())
        z_parameters = list(self.z_model.parameters())

        # Eq. 7a-7c: h and lambda updates using x^r, y^r, z^r
        g_xy_gap = self.lower_loss_fn(self.y_model, data, self.x_model, self.stochastic,
                                      seed, **kwargs)
        g_xz_gap = self.lower_loss_fn(self.z_model, data, self.x_model, self.stochastic,
                                      seed, **kwargs)
        moreau_prox = sum((z - y.detach()).square().sum()
                          for z, y in zip(z_parameters, y_parameters)) / (2 * self.gamma)
        raw_gap = g_xy_gap - g_xz_gap - moreau_prox
        constraint = raw_gap - self.delta

        with torch.no_grad():
            self.h.mul_(1 - self.theta).add_(self.theta * constraint.detach())
            lambda_plus = torch.clamp(self.lambda_value + self.tau * self.h, min=0)
            self.lambda_value.mul_(1 - self.mu).add_(self.mu * lambda_plus)
            lambda_used = self.lambda_value.clone()

        # Eq. 8: z direction at x^r, y^r, z^r
        grad_g_z = self._grads(g_xz_gap, z_parameters)
        direction_z = [(gr + (z - y.detach()) / self.gamma).detach()
                       for gr, z, y in zip(grad_g_z, z_parameters, y_parameters)]

        # Eq. 9: y direction at x^r, y^r, z^r
        f_y = self.upper_loss_fn(self.y_model, data, self.x_model, self.stochastic, seed, **kwargs)
        g_xy = self.lower_loss_fn(self.y_model, data, self.x_model, self.stochastic, seed, **kwargs)
        grad_f_y = self._grads(f_y, y_parameters)
        grad_g_y = self._grads(g_xy, y_parameters)

        direction_y = [(df + lambda_used * (dg + (z.detach() - y) / self.gamma)
                        + self.p * (y - y_bar)).detach()
                       for df, dg, y, z, y_bar in zip(grad_f_y, grad_g_y, y_parameters,
                                                      z_parameters, self.y_bar)]

        x_anchor_squared = sum((x.detach() - x_bar).square().sum()
                               for x, x_bar in zip(x_parameters, self.x_bar))
        y_anchor_squared = sum((y.detach() - y_bar).square().sum()
                               for y, y_bar in zip(y_parameters, self.y_bar))
        z_y_squared = sum((z.detach() - y.detach()).square().sum()
                          for z, y in zip(z_parameters, y_parameters))
        K_hat = (f_y.detach() + lambda_used * constraint.detach()
                 + 0.5 * self.p * (x_anchor_squared + y_anchor_squared))

        # y^{r+1} = y^r - beta * direction_y
        with torch.no_grad():
            for y, direction in zip(y_parameters, direction_y):
                y.add_(direction, alpha=-self.beta)
            for y_bar, y in zip(self.y_bar, y_parameters):
                y_bar.add_(y - y_bar, alpha=self.omega)

        # Eq. 11: x uses y^{r+1} and z^r
        f_x = self.upper_loss_fn(self.y_model, data, self.x_model, self.stochastic, seed, **kwargs)
        g_xy_new = self.lower_loss_fn(self.y_model, data, self.x_model, self.stochastic,
                                      seed, **kwargs)
        g_xz_old = self.lower_loss_fn(self.z_model, data, self.x_model, self.stochastic,
                                      seed, **kwargs)

        grad_f_x = self._grads(f_x, x_parameters)
        grad_g_xy_x = self._grads(g_xy_new, x_parameters)
        grad_g_xz_x = self._grads(g_xz_old, x_parameters)

        direction_x = [(df + lambda_used * (dg_y - dg_z) + self.p * (x - x_bar)).detach()
                       for df, dg_y, dg_z, x, x_bar in zip(grad_f_x, grad_g_xy_x, grad_g_xz_x,
                                                           x_parameters, self.x_bar)]

        # Apply x, x_bar and the previously calculated z step
        with torch.no_grad():
            for x, direction in zip(x_parameters, direction_x):
                x.add_(direction, alpha=-self.alpha)
            for x_bar, x in zip(self.x_bar, x_parameters):
                x_bar.add_(x - x_bar, alpha=self.omega)
            for z, direction in zip(z_parameters, direction_z):
                z.add_(direction, alpha=-self.eta)

        return {
            "ll_loss": g_xy_gap.detach().item(),
            "z_ll_loss": g_xz_gap.detach().item(),
            "ul_loss": f_y.detach().item(),
            "moreau_prox": moreau_prox.detach().item(),
            "raw_gap": raw_gap.detach().item(),
            "constraint": constraint.detach().item(),
            "h": self.h.item(),
            "lambda": self.lambda_value.item(),
            "z_y_norm": z_y_squared.sqrt().item(),
            "grad_z_norm": self._norm(direction_z).item(),
            "grad_y_K_norm": self._norm(direction_y).item(),
            "grad_x_K_norm": self._norm(direction_x).item(),
            "K_hat": K_hat.item(),
        }


class BLO_optimiser_AID_Neumann:
    """Approximate-implicit-differentiation bilevel method with a Neumann inverse-HVP.

    This is the method that used to be called `BLO_optimiser_STABLE` in
    blo_optimisers_old.py. It is NOT the STABLE algorithm of Chen, Sun, Xiao & Yin
    (see BLO_optimiser_STABLE below) -- it has neither the corrected lower-level
    update nor the recursive second-order estimates -- so it is named for what it
    actually does. It is still a reasonable AID / TTSA-style baseline.

    Per step, with y = lower level (ll_model) and x = upper level (ul_model):
        Hv    <- Hv - tau (grad^2_yy g) Hv + tau grad_y f
        hyper  = grad_x f - (grad^2_xy g) Hv
        y     <- y - lr_w     grad_y g
        x     <- x - lr_theta hyper

    The Hv recursion is Richardson iteration for [grad^2_yy g]^{-1} grad_y f: its
    fixed point satisfies (grad^2_yy g) Hv = grad_y f, and it converges when
    tau < 2/L with L the largest eigenvalue of grad^2_yy g.
    """

    def __init__(self,
                 ll_model: nn.Module,
                 lower_loss_fn: Callable,
                 upper_loss_fn: Callable,
                 ul_model: nn.Module | None = None,
                 stochastic: bool = None,
                 device: torch.device = torch.device("cpu"),
                 debug: bool = False,
                 w_optimiser: torch.optim.Optimizer = torch.optim.SGD,
                 theta_optimiser: torch.optim.Optimizer = torch.optim.SGD,
                 lr_w: float = 1e-3,          # mu in the old code
                 lr_theta: float = 1e-3,      # alpha in the old code
                 tau: float = 1e-3,           # Neumann/Richardson step size
                 beta: float | None = None,   # Hv initialisation scale (defaults to tau)
                 stop_threshold: float | None = None,   # 'mp' in the old code
                 w_optimiser_kwargs: dict | None = None,
                 theta_optimiser_kwargs: dict | None = None):

        self.ll_model = ll_model
        self.ul_model = ul_model
        self.lower_loss_fn = lower_loss_fn
        self.upper_loss_fn = upper_loss_fn

        self.w_params = list(ll_model.parameters())
        self.theta_params = list(ul_model.parameters()) if ul_model else []

        self.tau = tau
        self.beta = tau if beta is None else beta
        self.stop_threshold = stop_threshold

        self.w_optimiser = w_optimiser(self.w_params, lr=lr_w, **(w_optimiser_kwargs or {}))
        self.theta_optimiser = theta_optimiser(self.theta_params, lr=lr_theta,
                                               **(theta_optimiser_kwargs or {}))

        self.stochastic = stochastic
        self.device = device
        self.debug = debug
        self.iter_counter = 0
        self.current_Hv = None

        self.g_seed = 0
        self.f_seed = 1

    def step(self, data: Optional[Dict] = None, **kwargs):
        self.iter_counter += 1

        if self.stochastic:
            self.g_seed = 2 * self.iter_counter + 1
            self.f_seed = 2 * self.iter_counter + 2

        g_loss = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                    self.g_seed, **kwargs)
        f_loss = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                    self.f_seed, **kwargs)

        if (self.stop_threshold is not None
                and g_loss.item() <= self.stop_threshold
                and f_loss.item() <= self.stop_threshold):
            return {'ll_loss': g_loss.item(), 'ul_loss': f_loss.item(),
                    'grad_w_L_norm': 0.0, 'grad_theta_L_norm': 0.0, 'Hv_norm': 0.0}

        grad_y_g = torch.autograd.grad(g_loss, self.w_params, create_graph=True,
                                       retain_graph=True, allow_unused=True)
        grad_y_g = [gr if gr is not None else torch.zeros_like(w)
                    for gr, w in zip(grad_y_g, self.w_params)]

        grads_f = torch.autograd.grad(f_loss, self.w_params + self.theta_params,
                                      retain_graph=True, allow_unused=True)
        n_w = len(self.w_params)
        grad_y_f = [gr.detach() if gr is not None else torch.zeros_like(w)
                    for gr, w in zip(grads_f[:n_w], self.w_params)]
        grad_x_f = [gr.detach() if gr is not None else torch.zeros_like(t)
                    for gr, t in zip(grads_f[n_w:], self.theta_params)]

        if self.current_Hv is None:
            self.current_Hv = [self.beta * v for v in grad_y_f]

        hvp = torch.autograd.grad(grad_y_g, self.w_params, grad_outputs=self.current_Hv,
                                  retain_graph=True, allow_unused=True)
        hvp = [h if h is not None else torch.zeros_like(w) for h, w in zip(hvp, self.w_params)]

        self.current_Hv = [(hv - self.tau * h + self.tau * v).detach()
                           for hv, h, v in zip(self.current_Hv, hvp, grad_y_f)]

        if self.theta_params:
            mixed = torch.autograd.grad(grad_y_g, self.theta_params,
                                        grad_outputs=self.current_Hv,
                                        retain_graph=True, allow_unused=True)
            mixed = [m if m is not None else torch.zeros_like(t)
                     for m, t in zip(mixed, self.theta_params)]
        else:
            mixed = []

        hypergrad = [(ft - mx).detach() for ft, mx in zip(grad_x_f, mixed)]

        self.w_optimiser.zero_grad(set_to_none=True)
        self.theta_optimiser.zero_grad(set_to_none=True)
        for w, gw in zip(self.w_params, grad_y_g):
            w.grad = gw.detach().clone()
        for th, hg in zip(self.theta_params, hypergrad):
            th.grad = hg.clone()
        self.w_optimiser.step()
        self.theta_optimiser.step()

        grad_w_norm = torch.sqrt(sum((gw.detach() ** 2).sum() for gw in grad_y_g)).item()
        grad_theta_norm = (torch.sqrt(sum((hg ** 2).sum() for hg in hypergrad)).item()
                           if hypergrad else 0.0)
        Hv_norm = torch.sqrt(sum((hv ** 2).sum() for hv in self.current_Hv)).item()

        if self.debug:
            print(f"{self.iter_counter}: LL={g_loss.item():.4f} UL={f_loss.item():.4f} "
                  f"|grad_y g|={grad_w_norm:.4f} |hyper|={grad_theta_norm:.4f} "
                  f"|Hv|={Hv_norm:.4f}")

        return {
            'll_loss':           g_loss.item(),
            'ul_loss':           f_loss.item(),
            'grad_w_L_norm':     grad_w_norm,
            'grad_theta_L_norm': grad_theta_norm,
            'Hv_norm':           Hv_norm,
        }


class BLO_optimiser_STABLE:
    """STABLE -- Single-Timescale stochAstic BiLEvel optimization.

    Chen, Sun, Xiao & Yin, arXiv:2102.04671, Algorithm 1 with eqs. (8)-(9).
    Paper notation: x is the UPPER level (ul_model), y is the LOWER level (ll_model).

        H_xy^k  = P ((1 - tau)(H_xy^{k-1} - h_xy^{k-1}(phi_k)) + h_xy^k(phi_k))   (9a)
        H_yy^k  = P~((1 - tau)(H_yy^{k-1} - h_yy^{k-1}(phi_k)) + h_yy^k(phi_k))   (9b)
        x^{k+1} = P_X( x^k - alpha (grad_x f - H_xy H_yy^{-1} grad_y f) )         (8a)
        y^{k+1} = y^k - beta grad_y g - H_yy^{-1} H_xy^T (x^{k+1} - x^k)          (8b)

    where P projects onto the spectral-norm ball {||X|| <= C_gxy} and P~ onto
    {X >= mu_g I} (eigenvalue clamp after symmetrisation).

    The two features that distinguish STABLE from plain AID are both here:
      (F1) the correction term  -H_yy^{-1} H_xy^T (x^{k+1} - x^k)  in the y-update,
           which is what lets a single timescale work; and
      (F2) the recursive, variance-reduced second-order estimates (9), in which
           h^{k-1} and h^k are evaluated at the PREVIOUS and CURRENT iterates using
           the SAME sample phi_k. Copies of both models at the previous iterate are
           kept so that h^{k-1}(phi_k) can be formed.

    COST. H_yy is (d_y, d_y) and H_xy is (d_x, d_y), held as explicit matrices, and
    building them takes d_y backward passes per evaluation (two evaluations per
    step). That is fine for the toy problems and small models; it does not scale to
    the CIFAR networks. STABLE also assumes g is strongly convex in y -- the
    projection P~ enforces H_yy >= mu_g I, so a nonconvex g will run but the
    guarantees do not apply.
    """

    def __init__(self,
                 ll_model: nn.Module,
                 lower_loss_fn: Callable,
                 upper_loss_fn: Callable,
                 ul_model: nn.Module | None = None,
                 stochastic: bool = None,
                 device: torch.device = torch.device("cpu"),
                 debug: bool = False,
                 lr_w: float = 1e-3,        # beta  : lower-level step size
                 lr_theta: float = 1e-3,    # alpha : upper-level step size
                 tau: float = 0.1,          # tau_k : second-order recursion step size
                 C_gxy: float = 1e3,        # radius of the spectral-norm ball for H_xy
                 mu_g: float = 1e-2,        # strong-convexity floor for H_yy
                 x_projection: Callable | None = None):   # P_X, acting on the flat x vector

        if not ul_model:
            raise ValueError("STABLE needs an upper-level model (x).")

        self.ll_model = ll_model
        self.ul_model = ul_model
        self.lower_loss_fn = lower_loss_fn
        self.upper_loss_fn = upper_loss_fn

        self.w_params = list(ll_model.parameters())      # y
        self.theta_params = list(ul_model.parameters())  # x

        self.lr_w, self.lr_theta, self.tau = lr_w, lr_theta, tau
        self.C_gxy, self.mu_g = C_gxy, mu_g
        self.x_projection = x_projection

        self.stochastic = stochastic
        self.device = device
        self.debug = debug
        self.iter_counter = 0

        self.H_xy = None
        self.H_yy = None

        # (x^{k-1}, y^{k-1}), needed to form h^{k-1}(phi_k)
        self.prev_ll_model = deepcopy(ll_model).to(device)
        self.prev_ul_model = deepcopy(ul_model).to(device)
        self.has_prev = False

        self.g_seed = 0   # phi
        self.f_seed = 1   # xi

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _flat(tensors, params):
        return torch.cat([(t if t is not None else torch.zeros_like(p)).reshape(-1)
                          for t, p in zip(tensors, params)])

    @staticmethod
    def _unflatten_into(vec, params):
        i = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.copy_(vec[i:i + n].view_as(p))
                i += n

    def _second_order(self, ll_model, ul_model, data, seed, **kwargs):
        """h_yy = grad^2_yy g and h_xy = grad^2_xy g at (x, y) of the given models.

        Returns (h_xy (d_x, d_y), h_yy (d_y, d_y), grad_y g (d_y,)).
        """
        y_params = list(ll_model.parameters())
        x_params = list(ul_model.parameters())

        g = self.lower_loss_fn(ll_model, data, ul_model, self.stochastic, seed, **kwargs)
        gy = torch.autograd.grad(g, y_params, create_graph=True, retain_graph=True,
                                 allow_unused=True)
        gy_flat = torch.cat([(t if t is not None else torch.zeros_like(p, requires_grad=True)).reshape(-1)
                             for t, p in zip(gy, y_params)])

        d_y = gy_flat.numel()
        rows_yy, rows_yx = [], []
        for i in range(d_y):
            grads = torch.autograd.grad(gy_flat[i], y_params + x_params,
                                        retain_graph=True, allow_unused=True)
            n_y = len(y_params)
            rows_yy.append(self._flat(grads[:n_y], y_params))
            rows_yx.append(self._flat(grads[n_y:], x_params))

        h_yy = torch.stack(rows_yy).detach()          # (d_y, d_y)
        h_xy = torch.stack(rows_yx).detach().T        # (d_x, d_y)
        return h_xy, h_yy, gy_flat.detach()

    @staticmethod
    def _proj_norm_ball(X, radius):
        """P: projection onto {X : ||X||_2 <= radius}."""
        n = torch.linalg.matrix_norm(X, ord=2)
        return X * (radius / n) if n > radius else X

    @staticmethod
    def _proj_psd(X, floor):
        """P~: projection onto {X : X >= floor * I}."""
        X = 0.5 * (X + X.T)
        evals, evecs = torch.linalg.eigh(X)
        return evecs @ torch.diag(evals.clamp_min(floor)) @ evecs.T

    def _snapshot_previous(self):
        with torch.no_grad():
            for p_prev, p in zip(self.prev_ll_model.parameters(), self.w_params):
                p_prev.copy_(p)
            for p_prev, p in zip(self.prev_ul_model.parameters(), self.theta_params):
                p_prev.copy_(p)
        self.has_prev = True

    # --------------------------------------------------------------------- step

    def step(self, data: Optional[Dict] = None, **kwargs):
        self.iter_counter += 1

        if self.stochastic:
            self.g_seed = 2 * self.iter_counter + 1   # phi_k
            self.f_seed = 2 * self.iter_counter + 2   # xi_k

        # --- (9): second-order estimates. h^k and h^{k-1} share the sample phi_k.
        h_xy_cur, h_yy_cur, grad_y_g = self._second_order(
            self.ll_model, self.ul_model, data, self.g_seed, **kwargs)

        if not self.has_prev:
            H_xy_raw, H_yy_raw = h_xy_cur, h_yy_cur
        else:
            h_xy_prev, h_yy_prev, _ = self._second_order(
                self.prev_ll_model, self.prev_ul_model, data, self.g_seed, **kwargs)
            H_xy_raw = (1 - self.tau) * (self.H_xy - h_xy_prev) + h_xy_cur
            H_yy_raw = (1 - self.tau) * (self.H_yy - h_yy_prev) + h_yy_cur

        self.H_xy = self._proj_norm_ball(H_xy_raw, self.C_gxy)
        self.H_yy = self._proj_psd(H_yy_raw, self.mu_g)

        # --- upper-level gradients, sample xi_k
        f_loss = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                    self.f_seed, **kwargs)
        gf = torch.autograd.grad(f_loss, self.w_params + self.theta_params, allow_unused=True)
        n_w = len(self.w_params)
        grad_y_f = self._flat(gf[:n_w], self.w_params).detach()
        grad_x_f = self._flat(gf[n_w:], self.theta_params).detach()

        x_flat = self._flat([p.detach() for p in self.theta_params], self.theta_params)
        y_flat = self._flat([p.detach() for p in self.w_params], self.w_params)

        # --- (8a): x^{k+1} = P_X(x^k - alpha (grad_x f - H_xy H_yy^{-1} grad_y f))
        hyper = grad_x_f - self.H_xy @ torch.linalg.solve(self.H_yy, grad_y_f)
        x_new = x_flat - self.lr_theta * hyper
        if self.x_projection is not None:
            x_new = self.x_projection(x_new)
        delta_x = x_new - x_flat

        # --- (8b): y^{k+1} = y^k - beta grad_y g - H_yy^{-1} H_xy^T (x^{k+1} - x^k)
        correction = torch.linalg.solve(self.H_yy, self.H_xy.T @ delta_x)
        y_new = y_flat - self.lr_w * grad_y_g - correction

        # the previous iterate for the next step is the CURRENT one, before the write
        self._snapshot_previous()

        self._unflatten_into(x_new, self.theta_params)
        self._unflatten_into(y_new, self.w_params)

        with torch.no_grad():
            g_loss = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                        self.g_seed, **kwargs)
            min_eig = torch.linalg.eigvalsh(self.H_yy).min().item()

        if self.debug:
            print(f"{self.iter_counter}: LL={g_loss.item():.4f} UL={f_loss.item():.4f} "
                  f"|grad_y g|={grad_y_g.norm().item():.4f} |hyper|={hyper.norm().item():.4f} "
                  f"|correction|={correction.norm().item():.4f} "
                  f"min_eig(H_yy)={min_eig:.4g}")

        return {
            'll_loss':           g_loss.item(),
            'ul_loss':           f_loss.item(),
            'grad_w_L_norm':     grad_y_g.norm().item(),
            'grad_theta_L_norm': hyper.norm().item(),
            'correction_norm':   correction.norm().item(),
            'H_yy_min_eig':      min_eig,
            'H_xy_norm':         torch.linalg.matrix_norm(self.H_xy, ord=2).item(),
        }


# =============================================================================
#  AAL  (Approximated Augmented Lagrangian)
# =============================================================================
#  Reference: Hallak & Suissa, "An Augmented Lagrangian Approach to Bi-Level
#  Optimization via a Smooth Equilibrium Constrained Problem", JOTA 208:105 (2026),
#  Algorithm 1. Paper notation: x = upper level (theta here), y = lower level (W here),
#  z = multiplier, constraint grad_y g(x, y) = 0.
#
#      v_k      = z_k + rho_k grad_y g(x_k, y_k),        vbar_k = v_k / ||v_k||
#      gx_k     = grad_x f + (grad_x g(x_k, y_k + tau_k vbar_k) - grad_x g(x_k, y_k)) ||v_k|| / tau_k
#      gy_k     = grad_y f + (grad_y g(x_k, y_k + tau_k vbar_k) - grad_y g(x_k, y_k)) ||v_k|| / tau_k
#      x_{k+1}  = prox_{h/t_k}(x_k - gx_k / t_k)          (h = 0 here, so prox = identity)
#      y_{k+1}  = y_k - gy_k / t_k
#      if ||grad_y g(x_{k+1}, y_{k+1})|| > eps:
#          z_{k+1} = z_k + rho_k grad_y g(x_{k+1}, y_{k+1}),   rho_{k+1} = rho_k + alpha
#
#  Step size, Assumption 2 / (3.8):
#      t_k = 2 gamma + rho_k L_grad_g^2 + L_f + L_H ||z_k + rho_k grad_y g(x_k, y_k)||
#
#  Relative to PABLO: no RMSProp preconditioning, no proximal term, finite-difference
#  (DDA) Hessian-vector products, a single step 1/t_k shared by x and y, and the
#  multiplier / penalty change only while the iterate is eps-infeasible.
#
#  PRECISION. The paper's experiments use tau = 1e-8, which is below float32 resolution:
#  y + 1e-8 vbar == y and the DDA term is exactly 0. With float32 parameters use
#  tau ~ sqrt(machine eps) ~ 1e-3. tau_decay=True uses tau_k = tau / (k+1)^2 (the
#  summable example after Assumption 2), which hits the same floor after a few steps.
# =============================================================================


class BLO_optimiser_AAL:
    """AAL, Algorithm 1 of Hallak & Suissa (JOTA 2026). See the section header above."""

    def __init__(self, ll_model, ul_model, lower_loss_fn, upper_loss_fn,
                 L_f=1.0, L_grad_g=1.0, L_H=0.0, gamma=1.0, rho0=0.0, alpha=1e-3,
                 eps=1e-5, tau=1e-3, tau_decay=False, z_init=None,
                 stochastic=False, device=None, debug=False):
        if not rho0 > alpha > 0 and not (rho0 == 0.0 and alpha > 0):
            # Algorithm 1 asks for rho0 > alpha > 0; the paper's experiments use rho0 = 0.
            raise ValueError("need alpha > 0 and either rho0 > alpha or rho0 = 0")
        self.ll_model, self.ul_model = ll_model, ul_model
        self.lower_loss_fn, self.upper_loss_fn = lower_loss_fn, upper_loss_fn
        self.w_params = list(ll_model.parameters())
        self.theta_params = list(ul_model.parameters())
        self.L_f, self.L_grad_g, self.L_H, self.gamma = L_f, L_grad_g, L_H, gamma
        self.rho, self.alpha, self.eps = float(rho0), alpha, eps
        self.tau, self.tau_decay = tau, tau_decay
        self.stochastic, self.debug = stochastic, debug
        self.device = device or self.w_params[0].device
        self.iter_counter = 0

        y0 = torch.nn.utils.parameters_to_vector(self.w_params).detach()
        self.z = (torch.zeros_like(y0) if z_init is None
                  else torch.as_tensor(z_init, dtype=y0.dtype, device=y0.device).flatten().clone())

    def _flat_grads(self, loss, params):
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
                          for g, p in zip(grads, params)])

    def _grads_g(self, data, seed, **kwargs):
        """(g, grad_x g, grad_y g) at the current parameters."""
        g = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic, seed, **kwargs)
        gx = self._flat_grads(g, self.theta_params) if self.theta_params else torch.zeros(0)
        g = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic, seed, **kwargs)
        gy = self._flat_grads(g, self.w_params)
        return g.detach(), gx, gy

    def step(self, data: Optional[Dict] = None, **kwargs):
        seed = self.iter_counter
        tau_k = self.tau / (self.iter_counter + 1) ** 2 if self.tau_decay else self.tau
        self.iter_counter += 1

        x = torch.nn.utils.parameters_to_vector(self.theta_params).detach().clone()
        y = torch.nn.utils.parameters_to_vector(self.w_params).detach().clone()

        # line 3: v_k from grad_y g at (x_k, y_k)
        g_val, gx_g, gy_g = self._grads_g(data, seed, **kwargs)
        v = self.z + self.rho * gy_g
        v_norm = v.norm()

        f = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic, seed, **kwargs)
        gx_f = self._flat_grads(f, self.theta_params) if self.theta_params else torch.zeros(0)
        f = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic, seed, **kwargs)
        gy_f = self._flat_grads(f, self.w_params)

        # lines 4-5: directional derivative approximation along vbar
        if v_norm > 0:
            with torch.no_grad():
                torch.nn.utils.vector_to_parameters(y + tau_k * v / v_norm, self.w_params)
            _, gx_g_pert, gy_g_pert = self._grads_g(data, seed, **kwargs)
            with torch.no_grad():
                torch.nn.utils.vector_to_parameters(y, self.w_params)
            scale = v_norm / tau_k
            dir_x = gx_f + (gx_g_pert - gx_g) * scale
            dir_y = gy_f + (gy_g_pert - gy_g) * scale
        else:
            dir_x, dir_y = gx_f, gy_f

        # (3.8) step size, lines 6-7
        t_k = (2 * self.gamma + self.rho * self.L_grad_g ** 2 + self.L_f
               + self.L_H * v_norm.item())
        with torch.no_grad():
            if self.theta_params:
                torch.nn.utils.vector_to_parameters(x - dir_x / t_k, self.theta_params)
            torch.nn.utils.vector_to_parameters(y - dir_y / t_k, self.w_params)

        # lines 8-12: multiplier and penalty only while eps-infeasible at the new iterate
        _, _, gy_new = self._grads_g(data, seed, **kwargs)
        feas = gy_new.norm().item()
        if feas > self.eps:
            self.z = self.z + self.rho * gy_new
            self.rho += self.alpha

        if self.debug:
            print(f"{self.iter_counter}: LL={g_val.item():.4f} UL={f.item():.4f} "
                  f"|grad_y g|={feas:.3e} rho={self.rho:.4g} |z|={self.z.norm().item():.4g} "
                  f"t_k={t_k:.4g}")

        return {
            'll_loss':      g_val.item(),
            'ul_loss':      f.detach().item(),
            'feasibility':  feas,
            'rho':          self.rho,
            'z_norm':       self.z.norm().item(),
            't_k':          t_k,
            'grad_x_norm':  dir_x.norm().item(),
            'grad_y_norm':  dir_y.norm().item(),
        }


# =============================================================================
#  PBGD-Free  (value-function-free penalty-based gradient descent)
# =============================================================================
#  Reference: Jiang, Xiao, Chen & Chen, "Beyond Value Functions: Single-Loop Bilevel
#  Optimization under Flatness Conditions", arXiv:2507.20400, Algorithm 1 / eq. (8).
#  Paper notation: x = upper level (theta here), y = lower level (W here).
#
#      for k = 0..K-1  (y warm-started from the previous outer iteration):
#          y <- y - eta_gamma ( grad_y f(x_t, y) / gamma + grad_y g(x_t, y) )
#      x_{t+1} = x_t - eta grad_x f(x_t, y)          at the updated y
#
#  The value-function part gamma (grad_x g(x, y*_gamma) - grad_x g(x, y*_g)) of the
#  penalty gradient (7) is dropped, so the method only reaches a neighbourhood of a
#  stationary point unless f is flat in y (Proposition 2, Theorem 3). Plain gradient
#  steps, as in the paper. K = 1 is the paper's "common choice".
# =============================================================================


class BLO_optimiser_PBGD_Free:
    """PBGD-Free, Algorithm 1 of Jiang et al. (2025). See the section header above."""

    def __init__(self, ll_model, ul_model, lower_loss_fn, upper_loss_fn,
                 gamma=10.0, eta=1e-2, eta_gamma=1e-2, K=1,
                 stochastic=False, device=None, debug=False):
        self.ll_model, self.ul_model = ll_model, ul_model
        self.lower_loss_fn, self.upper_loss_fn = lower_loss_fn, upper_loss_fn
        self.w_params = list(ll_model.parameters())
        self.theta_params = list(ul_model.parameters())
        self.gamma, self.eta, self.eta_gamma, self.K = gamma, eta, eta_gamma, K
        self.stochastic, self.debug = stochastic, debug
        self.device = device or self.w_params[0].device
        self.iter_counter = 0

    @staticmethod
    def _grads(loss, params):
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return [torch.zeros_like(p) if g is None else g.detach() for g, p in zip(grads, params)]

    def step(self, data: Optional[Dict] = None, **kwargs):
        seed = self.iter_counter
        self.iter_counter += 1

        with torch.no_grad():
            ll_loss = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                         seed, **kwargs)
            ul_loss = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                         seed, **kwargs)

        # lines 3-5: K steps on the penalised lower level f / gamma + g
        for k in range(self.K):
            f = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                   seed, **kwargs)
            g = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                   seed, **kwargs)
            grad_y = self._grads(f / self.gamma + g, self.w_params)
            with torch.no_grad():
                for p, gr in zip(self.w_params, grad_y):
                    p.add_(gr, alpha=-self.eta_gamma)

        # line 6: x step on grad_x f at the updated y
        f = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic, seed, **kwargs)
        grad_x = self._grads(f, self.theta_params) if self.theta_params else []
        with torch.no_grad():
            for p, gr in zip(self.theta_params, grad_x):
                p.add_(gr, alpha=-self.eta)

        grad_y_norm = torch.sqrt(sum((gr ** 2).sum() for gr in grad_y)).item()
        grad_x_norm = torch.sqrt(sum((gr ** 2).sum() for gr in grad_x)).item() if grad_x else 0.0
        if self.debug:
            print(f"{self.iter_counter}: LL={ll_loss.item():.4f} UL={ul_loss.item():.4f} "
                  f"|grad_y P|={grad_y_norm:.3e} |grad_x f|={grad_x_norm:.3e}")

        return {
            'll_loss':      ll_loss.item(),
            'ul_loss':      ul_loss.item(),
            'grad_y_norm':  grad_y_norm,
            'grad_x_norm':  grad_x_norm,
        }
