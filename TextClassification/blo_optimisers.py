import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from copy import deepcopy
from math import sqrt

from torch.func import jvp, grad, functional_call, vjp


def _trainable(module) -> list:
    """The parameters of `module` that are optimised: those with requires_grad=True.

    Frozen parameters (requires_grad=False) are excluded everywhere -- they get no
    optimiser state, no multiplier / reference copy, and are never differentiated.
    A bare tensor is accepted for ul_model (as in AAL); None gives [].
    """
    if module is None:
        return []
    if isinstance(module, torch.Tensor):
        return [module] if module.requires_grad else []
    return [p for p in module.parameters() if p.requires_grad]


# Seeds of different runs are kept SEED_STRIDE apart, so run r uses
# [r * SEED_STRIDE + 1, (r + 1) * SEED_STRIDE). PABLO consumes 4 seeds per step, so runs
# stay disjoint for up to SEED_STRIDE / 4 = 250,000 PABLO steps (1,000,000 for the
# others). The largest seed is below 2**32 for run seeds up to 4294, so the seeds are
# also valid for np.random.seed.
SEED_STRIDE = 1_000_000




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
#  Variant with the proximal term outside the preconditioner (p tracks grad L only):
#      BLO_optimiser_PABLO_Prox_Decouple, BLO_optimiser_PABLO_Prox_Decouple_Bound
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
                 grad_clip: float | None = None,
                 seed: int = 0):

        self.ll_model = ll_model
        self.ul_model = ul_model
        self.lower_loss_fn = lower_loss_fn
        self.upper_loss_fn = upper_loss_fn

        self.w_params = _trainable(ll_model)
        self.theta_params = _trainable(ul_model)

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
        self.seed = int(seed)   # run seed: offsets the sample stream
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
        """xi_1, xi_2, xi_3 and the multiplier sample, all distinct.

        Step k uses run_offset + 4k + 1 .. run_offset + 4k + 4, run_offset = seed * SEED_STRIDE.
        """
        if self.stochastic:
            base = self.seed * SEED_STRIDE + 4 * self.iter_counter
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

        grad_w_L_norm     = sqrt(sum(g.norm().item()**2 for g in grad_W_L))
        grad_theta_L_norm = sqrt(sum(g.norm().item()**2 for g in grad_th_L))

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

        grad_w_L_norm     = sqrt(sum(g.norm().item()**2 for g in grad_W_K))
        grad_theta_L_norm = sqrt(sum(g.norm().item()**2 for g in grad_th_K))

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





class BLO_optimiser_PABLO_Prox_Decouple_Bound(BLO_optimiser_PABLO_Prox):
    """PABLO_Prox with the proximal term taken OUTSIDE the preconditioner.

    The RMSProp second-moment estimate tracks grad L only (not grad K), and the
    proximal pull is applied as a plain, unpreconditioned step:

        p_i     = beta p_{i-1} + (1 - beta)(grad L_{i-1})^2
        Z_i     = Z_{i-1} - mu grad L_{i-1} / sqrt(p_i + eps) - mu tau (Z_{i-1} - y_{i-1})
        y_i     = y_{i-1} + gamma tau (Z_i - y_{i-1})

    mu is the current lr of the W / theta optimiser (so schedulers apply to both
    terms). grad_clip, if set, now clips grad L only. Proj. of u onto ball of radius R
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

    def _prox_terms(self, grad_W_L, grad_th_L):
        """Store Z_{i-1} - y_{i-1} for the prox step; return grad L unchanged."""
        with torch.no_grad():
            self._z_y_w  = [(w - y).detach().clone() for w, y in zip(self.w_params, self.y_params_w)]
            self._z_y_th = [(t - y).detach().clone() for t, y in zip(self.theta_params, self.y_params_theta)]
            sq = sum(d.pow(2).sum() for d in self._z_y_w + self._z_y_th)
        return grad_W_L, grad_th_L, sq.sqrt().item()

    def _apply_primal(self, grad_W_L, grad_th_L):
        """Preconditioned step on grad L, then Z -= mu tau (Z_{i-1} - y_{i-1})."""
        # lr is read before the optimiser step; schedulers are only stepped later
        diffs = {id(p): d for p, d in zip(self.w_params + self.theta_params,
                                          self._z_y_w + self._z_y_th)}
        mus = [(group['lr'], group['params'])
               for opt in (self.w_optimiser, self.theta_optimiser)
               for group in opt.param_groups]
        super()._apply_primal(grad_W_L, grad_th_L)
        with torch.no_grad():
            for mu, params in mus:
                for p in params:
                    p.sub_(mu * self.tau * diffs[id(p)])




# =============================================================================
#  COMPARISON ALGORITHMS
# =============================================================================
#  Extracted (2026-09-15) from the summer-student code in ./ShelleyCode so that
#  every optimiser used in the PABLO comparison study lives in one module.
#
#    BLO_optimiser_BOME         <- ShelleyCode/BOME.py
#    BLO_optimiser_TSP_Paper    <- ShelleyCode/TSP.py   (paper-faithful variant)
#    BLO_optimiser_STABLE       <- blo_optimisers_old.py (ported to this API)
#
#  Common API (matches BLO_optimiser_PABLO above):
#      lower_loss_fn(ll_model, data, ul_model, stochastic, seed, **kwargs) -> scalar
#      upper_loss_fn(ll_model, data, ul_model, stochastic, seed, **kwargs) -> scalar
#      opt.step(data, **kwargs) -> dict of scalars
#
#  SEEDING. Every class takes a run seed `seed` (default 0) in its constructor and keeps
#  an internal iter_counter. At iteration k = 1, 2, ... each algorithm below uses the
#  single seed  seed * SEED_STRIDE + k  (its _draw_seed method) for EVERY loss evaluation
#  in that step, upper and lower level and any inner loop. step() takes no seed argument.
#  PABLO is the exception: it draws four distinct samples per step,
#  seed * SEED_STRIDE + 4k + 1 .. + 4k + 4 (its _draw_seeds method).
#
#  FROZEN PARAMETERS. Only parameters with requires_grad=True are optimised
#  (via _trainable). Model copies (BOME inner model, TSP z, F2SA z, VPBGD omega,
#  STABLE previous iterate) are deepcopies, which keep the requires_grad flags, so
#  the same filter selects the matching tensors in the copy.
# =============================================================================



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
                 barrier="grad_norm", stochastic=False, device=None, eps=1e-12,
                 seed=0):
        self.ll_model, self.ul_model = ll_model, ul_model
        self.lower_loss_fn, self.upper_loss_fn = lower_loss_fn, upper_loss_fn
        self.inner_steps, self.inner_lr, self.eta = inner_steps, inner_lr, eta
        self.barrier, self.stochastic, self.eps = barrier, stochastic, eps
        self.device = device or next(ll_model.parameters()).device
        self.w_params = _trainable(ll_model)
        self.theta_params = _trainable(ul_model)
        self.w_optimizer = w_optimiser(self.w_params, lr=lr_w)
        self.theta_optimizer = theta_optimiser(self.theta_params, lr=lr_theta)
        self.inner_model = deepcopy(ll_model).to(self.device)
        # Fixed here, before any requires_grad toggling: the copies of the trainable W.
        self.inner_params = _trainable(self.inner_model)
        self.iter_counter = 0
        self.seed = int(seed)   # run seed: offsets the sample stream
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
        inner_parameters = self.inner_params
        for parameter in inner_parameters:
            parameter.requires_grad_(True)

        for inner_iteration in range(self.inner_steps):
            inner_loss = self.lower_loss_fn(self.inner_model, data, self.ul_model,
                                            self.stochastic, seed, **kwargs)
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

    def _draw_seed(self):
        """The ONE sample seed used by every loss evaluation in this step.

        Run offset (self.seed * SEED_STRIDE) plus the iteration k = 1, 2, ...
        Under stochastic=False the loss functions ignore the seed; 0 is passed.
        """
        return self.seed * SEED_STRIDE + self.iter_counter if self.stochastic else 0

    def step(self, data=None, **kwargs):
        self.iter_counter += 1
        seed = self._draw_seed()
        theta_tilde = self._make_inner_approximation(data, seed, **kwargs)

        lower_loss = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                        seed, **kwargs)
        inner_lower_loss = self.lower_loss_fn(theta_tilde, data, self.ul_model, self.stochastic,
                                              seed, **kwargs)
        q_hat = lower_loss - inner_lower_loss
        upper_loss = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                        seed, **kwargs)

        parameters = self.w_params + self.theta_params

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
                 omega=0.1, lambda_init=0.5, stochastic=False, device=None, seed=0):
        self.y_model, self.x_model = ll_model, ul_model
        self.lower_loss_fn, self.upper_loss_fn = lower_loss_fn, upper_loss_fn
        self.gamma, self.delta, self.p = gamma, delta, p
        self.tau, self.mu, self.theta = tau, mu, theta
        self.eta, self.alpha, self.beta, self.omega = eta, alpha, beta, omega
        self.stochastic = stochastic
        self.device = device or next(ll_model.parameters()).device

        self.y_params = _trainable(ll_model)
        self.x_params = _trainable(ul_model)
        self.z_model = deepcopy(ll_model).to(self.device)
        self.z_params = _trainable(self.z_model)
        self.y_bar = [p_.detach().clone() for p_ in self.y_params]
        self.x_bar = [p_.detach().clone() for p_ in self.x_params]
        self.lambda_value = torch.tensor(float(lambda_init), device=self.device)
        self.h = torch.zeros((), device=self.device)
        self.iter_counter = 0
        self.seed = int(seed)   # run seed: offsets the sample stream

    @staticmethod
    def _grads(loss, parameters):
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        return [torch.zeros_like(parameter) if gradient is None else gradient
                for parameter, gradient in zip(parameters, gradients)]

    @staticmethod
    def _norm(tensors):
        return torch.stack([t.detach().square().sum() for t in tensors]).sum().sqrt()

    def _draw_seed(self):
        """The ONE sample seed used by every loss evaluation in this step.

        Run offset (self.seed * SEED_STRIDE) plus the iteration k = 1, 2, ...
        Under stochastic=False the loss functions ignore the seed; 0 is passed.
        """
        return self.seed * SEED_STRIDE + self.iter_counter if self.stochastic else 0

    def step(self, data=None, **kwargs):
        self.iter_counter += 1
        seed = self._draw_seed()
        x_parameters = self.x_params
        y_parameters = self.y_params
        z_parameters = self.z_params

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
                 theta_optimiser_kwargs: dict | None = None,
                 seed: int = 0):

        self.ll_model = ll_model
        self.ul_model = ul_model
        self.lower_loss_fn = lower_loss_fn
        self.upper_loss_fn = upper_loss_fn

        self.w_params = _trainable(ll_model)
        self.theta_params = _trainable(ul_model)

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
        self.seed = int(seed)   # run seed: offsets the sample stream
        self.current_Hv = None

    def _draw_seed(self):
        """The ONE sample seed used by every loss evaluation in this step.

        Run offset (self.seed * SEED_STRIDE) plus the iteration k = 1, 2, ...
        Under stochastic=False the loss functions ignore the seed; 0 is passed.
        """
        return self.seed * SEED_STRIDE + self.iter_counter if self.stochastic else 0

    def step(self, data: Optional[Dict] = None, **kwargs):
        self.iter_counter += 1
        seed = self._draw_seed()

        g_loss = self.lower_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                    seed, **kwargs)
        f_loss = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                    seed, **kwargs)

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
                 x_projection: Callable | None = None,    # P_X, acting on the flat x vector
                 seed: int = 0):

        if not ul_model:
            raise ValueError("STABLE needs an upper-level model (x).")

        self.ll_model = ll_model
        self.ul_model = ul_model
        self.lower_loss_fn = lower_loss_fn
        self.upper_loss_fn = upper_loss_fn

        self.w_params = _trainable(ll_model)      # y
        self.theta_params = _trainable(ul_model)  # x

        self.lr_w, self.lr_theta, self.tau = lr_w, lr_theta, tau
        self.C_gxy, self.mu_g = C_gxy, mu_g
        self.x_projection = x_projection

        self.stochastic = stochastic
        self.device = device
        self.debug = debug
        self.iter_counter = 0
        self.seed = int(seed)   # run seed: offsets the sample stream

        self.H_xy = None
        self.H_yy = None

        # (x^{k-1}, y^{k-1}), needed to form h^{k-1}(phi_k)
        self.prev_ll_model = deepcopy(ll_model).to(device)
        self.prev_ul_model = deepcopy(ul_model).to(device)
        self.prev_w_params = _trainable(self.prev_ll_model)
        self.prev_theta_params = _trainable(self.prev_ul_model)
        self.has_prev = False

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
        y_params = _trainable(ll_model)
        x_params = _trainable(ul_model)

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
        """P~: projection onto {X : X >= floor * I}.

        The eigendecomposition is done in float64. In float32 the LAPACK driver
        fails to converge (`error code: 9`) on the ill-conditioned H_yy that the
        higher-condition-number problems produce, and the projected matrix itself
        has many eigenvalues pinned to exactly `floor`, which is the repeated-
        eigenvalue case the same driver is worst at.
        """
        X = 0.5 * (X + X.T)
        if not torch.isfinite(X).all():
            raise FloatingPointError(
                "STABLE: H_yy holds NaN/Inf before the PSD projection -- the iterates "
                "have diverged. Reduce lr_theta/lr_w, or raise mu_g: the correction "
                "term H_yy^{-1} H_xy^T dx scales like 1/mu_g.")
        evals, evecs = torch.linalg.eigh(X.double())
        return (evecs @ torch.diag(evals.clamp_min(floor)) @ evecs.T).to(X.dtype)

    def _snapshot_previous(self):
        with torch.no_grad():
            for p_prev, p in zip(self.prev_w_params, self.w_params):
                p_prev.copy_(p)
            for p_prev, p in zip(self.prev_theta_params, self.theta_params):
                p_prev.copy_(p)
        self.has_prev = True

    # --------------------------------------------------------------------- step

    def _draw_seed(self):
        """The ONE sample seed used by every loss evaluation in this step.

        Run offset (self.seed * SEED_STRIDE) plus the iteration k = 1, 2, ...
        Under stochastic=False the loss functions ignore the seed; 0 is passed.
        """
        return self.seed * SEED_STRIDE + self.iter_counter if self.stochastic else 0

    def step(self, data: Optional[Dict] = None, **kwargs):
        self.iter_counter += 1
        seed = self._draw_seed()   # phi_k = xi_k

        # --- (9): second-order estimates. h^k and h^{k-1} share the sample phi_k.
        h_xy_cur, h_yy_cur, grad_y_g = self._second_order(
            self.ll_model, self.ul_model, data, seed, **kwargs)

        if not self.has_prev:
            H_xy_raw, H_yy_raw = h_xy_cur, h_yy_cur
        else:
            h_xy_prev, h_yy_prev, _ = self._second_order(
                self.prev_ll_model, self.prev_ul_model, data, seed, **kwargs)
            H_xy_raw = (1 - self.tau) * (self.H_xy - h_xy_prev) + h_xy_cur
            H_yy_raw = (1 - self.tau) * (self.H_yy - h_yy_prev) + h_yy_cur

        self.H_xy = self._proj_norm_ball(H_xy_raw, self.C_gxy)
        self.H_yy = self._proj_psd(H_yy_raw, self.mu_g)

        # --- upper-level gradients, sample xi_k
        f_loss = self.upper_loss_fn(self.ll_model, data, self.ul_model, self.stochastic,
                                    seed, **kwargs)
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
                                        seed, **kwargs)
            min_eig = torch.linalg.eigvalsh(self.H_yy.double()).min().item()

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
    """AAL — Algorithm 1 of Hallak & Suissa (JOTA 2026).

    Variables: x = θ (upper/outer), y = w (lower/inner), z = multiplier (same dim as w).
    Constraint enforced: ∇_y g(x, y) = 0  (stationarity of inner problem).

    Parameters
    ----------
    rho0  : initial penalty ρ₀ ≥ 0
    alpha : penalty increment α > 0; ρ increases by α each infeasible step
    eps   : feasibility threshold ε; multiplier only updated when ‖∇_y g‖ > ε
    t     : prox parameter tₖ; effective step size is 1/t (constant, tunable)
    tau   : DDA finite-difference precision τ
    """

    def __init__(self, ll_model, ul_model, lower_loss_fn, upper_loss_fn,
                 rho0=1.0, alpha=0.1, eps=1e-3, t=0.1, tau=1e-3,
                 z_init=None, stochastic=False, device=None, seed=0):
        self.ll_model, self.ul_model = ll_model, ul_model
        self.lower_loss_fn, self.upper_loss_fn = lower_loss_fn, upper_loss_fn
        self.w_params     = _trainable(ll_model)
        self.theta_params = _trainable(ul_model)   # also accepts a bare tensor
        self.rho, self.alpha, self.eps = float(rho0), float(alpha), float(eps)
        self.t, self.tau = float(t), float(tau)
        self.stochastic = stochastic
        self.device = device or self.w_params[0].device
        self.iter_counter = 0
        self.seed = int(seed)   # run seed: offsets the sample stream

        y0 = torch.nn.utils.parameters_to_vector(self.w_params).detach()
        self.z = (torch.zeros_like(y0) if z_init is None
                  else torch.as_tensor(z_init, dtype=y0.dtype, device=y0.device).flatten().clone())

    def _grads(self, loss_fn, data, seed, wrt_x=True, wrt_y=True, **kwargs):
        """Compute flat gradients of loss_fn w.r.t. theta (x) and/or w (y)."""
        params = (self.theta_params if wrt_x else []) + (self.w_params if wrt_y else [])
        loss = loss_fn(self.ll_model, data, self.ul_model, self.stochastic, seed, **kwargs)
        loss_val = loss.detach().item()
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        flat = [((g if g is not None else torch.zeros_like(p)).detach().flatten())
                for g, p in zip(grads, params)]
        n_x = len(self.theta_params) if wrt_x else 0
        gx = torch.cat(flat[:n_x])        if wrt_x and self.theta_params else torch.zeros(0)
        gy = torch.cat(flat[n_x:])        if wrt_y else torch.zeros(0)
        return loss_val, gx, gy

    def _draw_seed(self):
        """The ONE sample seed used by every loss evaluation in this step.

        Run offset (self.seed * SEED_STRIDE) plus the iteration k = 1, 2, ...
        Under stochastic=False the loss functions ignore the seed; 0 is passed.
        """
        return self.seed * SEED_STRIDE + self.iter_counter if self.stochastic else 0

    def step(self, data: Optional[Dict] = None, **kwargs):
        self.iter_counter += 1
        seed = self._draw_seed()

        y = torch.nn.utils.parameters_to_vector(self.w_params).detach().clone()
        x = (torch.nn.utils.parameters_to_vector(self.theta_params).detach().clone()
             if self.theta_params else torch.zeros(0))


        ll_val, gx_g, gy_g = self._grads(self.lower_loss_fn, data, seed, **kwargs)
        ul_val, gx_f, gy_f = self._grads(self.upper_loss_fn, data, seed, **kwargs)

        # v = z + rho * grad_y g   (the vector the Hessian of g is applied to)
        v      = self.z + self.rho * gy_g
        v_norm = v.norm()

        if v_norm.item() <= 1e-12:
            # nothing to differentiate along: the AL gradient collapses to grad f
            update_x, update_y = gx_f, gy_f
        else:
            v_bar = v / v_norm
            # DDA: perturb ONLY y, then re-evaluate BOTH gradients of g.
            with torch.no_grad():
                torch.nn.utils.vector_to_parameters(y + self.tau * v_bar, self.w_params)
            _, gx_g_pert, gy_g_pert = self._grads(self.lower_loss_fn, data, seed, **kwargs)
            with torch.no_grad():
                torch.nn.utils.vector_to_parameters(y, self.w_params)

            # grad_x L = grad_x f + (grad_xy^2 g) v  ~=  grad_x f + ||v||/tau (grad_x g(y+tau v_bar) - grad_x g(y))
            # grad_y L = grad_y f + (grad_yy^2 g) v  ~=  grad_y f + ||v||/tau (grad_y g(y+tau v_bar) - grad_y g(y))
            update_x = gx_f + (v_norm / self.tau) * (gx_g_pert - gx_g)
            update_y = gy_f + (v_norm / self.tau) * (gy_g_pert - gy_g)

        # prox-linear step with parameter t (step size 1/t), from the UNperturbed iterate
        with torch.no_grad():
            if self.theta_params:
                torch.nn.utils.vector_to_parameters(x - update_x / self.t, self.theta_params)
            torch.nn.utils.vector_to_parameters(y - update_y / self.t, self.w_params)

        _, _, gy_new = self._grads(self.lower_loss_fn, data, seed, wrt_x=False, **kwargs)
        if gy_new.norm().item() > self.eps:
            self.z   = self.z + self.rho * gy_new
            self.rho += self.alpha


        return {
            'll_loss':     ll_val,
            'ul_loss':     ul_val,
            'feasibility': gy_new.norm().item(),
            'rho':         self.rho,
            'z_norm':      self.z.norm().item(),
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
                 stochastic=False, device=None, debug=False, seed=0):
        self.ll_model, self.ul_model = ll_model, ul_model
        self.lower_loss_fn, self.upper_loss_fn = lower_loss_fn, upper_loss_fn
        self.w_params = _trainable(ll_model)
        self.theta_params = _trainable(ul_model)
        self.gamma, self.eta, self.eta_gamma, self.K = gamma, eta, eta_gamma, K
        self.stochastic, self.debug = stochastic, debug
        self.device = device or self.w_params[0].device
        self.iter_counter = 0
        self.seed = int(seed)   # run seed: offsets the sample stream

    @staticmethod
    def _grads(loss, params):
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return [torch.zeros_like(p) if g is None else g.detach() for g, p in zip(grads, params)]

    def _draw_seed(self):
        """The ONE sample seed used by every loss evaluation in this step.

        Run offset (self.seed * SEED_STRIDE) plus the iteration k = 1, 2, ...
        Under stochastic=False the loss functions ignore the seed; 0 is passed.
        """
        return self.seed * SEED_STRIDE + self.iter_counter if self.stochastic else 0

    def step(self, data: Optional[Dict] = None, **kwargs):
        self.iter_counter += 1
        seed = self._draw_seed()

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


# =============================================================================
#  F²SA  (Fully First-order Stochastic Approximation)
# =============================================================================
#  Reference: Kwon, Kwon, Wright & Nowak, "A Fully First-Order Method for
#  Stochastic Bilevel Optimization", ICML 2023, Algorithm 1.
#
#  Problem: min_x F(x) = f(x, y*(x))   s.t.  y*(x) ∈ argmin_y g(x, y)
#
#  Reformulated via increasing-multiplier Lagrangian:
#      L_λ(x,y) = f(x,y) + λ (g(x,y) − g*(x))
#
#  Three parallel sequences (all first-order):
#      y_k tracks y*_λk(x_k) = argmin_y L_{λk}(x_k,·)   via GD on L_λ
#      z_k tracks y*(x_k)    = argmin_y g(x_k,·)          via GD on g
#      x_k is the outer variable, updated with an approximation of ∇F
#
#  Algorithm 1 (single outer step k):
#      for t = 0 … T−1:
#          z_{t+1} = z_t − γ_k ∇_z g(x_k, z_t)
#          y_{t+1} = y_t − α_k (∇_y f(x_k, y_t) + λ_k ∇_y g(x_k, y_t))
#      x_{k+1} = x_k − ξ α_k (∇_x f(x_k,y_{T}) + λ_k (∇_x g(x_k,y_{T}) − ∇_x g(x_k,z_{T})))
#      λ_{k+1} = λ_k + δ_k
#
#  Notation map:  x → ul_model (θ),  y → ll_model (w),  z → z_model (copy of w)
#                 g → lower_loss_fn,  f → upper_loss_fn
# =============================================================================


class BLO_optimiser_F2SA:
    """F²SA — Fully First-order Stochastic Approximation (Kwon et al., ICML 2023).

    See the section header above for the algorithm and notation.

    Hyperparameters
    ---------------
    lr_w   (α_k)   : step size for y (inner Lagrangian tracking)
    lr_z   (γ_k)   : step size for z (lower-level solution tracking; defaults to lr_w)
    lr_x   (ξ α_k) : step size for x (outer update)
    lambda_init    : initial multiplier λ₀ (paper recommends ≥ 2 l_{f,1} / μ_g)
    delta  (δ_k)   : multiplier increment per outer step (fixed δ, not decaying)
    inner_steps (T): inner GD iterations per outer step; T=1 gives the single-loop variant
    """

    def __init__(self, ll_model, lower_loss_fn, upper_loss_fn, ul_model=None,
                 stochastic=False, device=None, debug=False,
                 lr_w=1e-3, lr_z=None, lr_x=1e-3,
                 lambda_init=0.0, delta=0.01, inner_steps=1, grad_clip=None,
                 seed=0):
        self.ll_model = ll_model
        self.ul_model = ul_model
        self.lower_loss_fn = lower_loss_fn
        self.upper_loss_fn = upper_loss_fn
        self.stochastic = stochastic
        self.device = device or next(ll_model.parameters()).device
        self.debug = debug
        self.grad_clip = grad_clip

        self.w_params = _trainable(ll_model)
        self.theta_params = _trainable(ul_model)

        # z: auxiliary copy chasing y*(x_k) via plain GD on g
        self.z_model = deepcopy(ll_model).to(self.device)
        self.z_params = _trainable(self.z_model)

        self.lr_w = lr_w
        self.lr_z = lr_z if lr_z is not None else lr_w
        self.lr_x = lr_x
        self.lambda_val = float(lambda_init)
        self.delta = delta
        self.inner_steps = inner_steps
        self.iter_counter = 0
        self.seed = int(seed)   # run seed: offsets the sample stream

    @staticmethod
    def _filled(grads, params):
        return [g.detach() if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, params)]

    def _draw_seed(self):
        """The ONE sample seed used by every loss evaluation in this step.

        Run offset (self.seed * SEED_STRIDE) plus the iteration k = 1, 2, ...
        Under stochastic=False the loss functions ignore the seed; 0 is passed.
        """
        return self.seed * SEED_STRIDE + self.iter_counter if self.stochastic else 0

    def step(self, data=None, **kwargs):
        self.iter_counter += 1
        seed = self._draw_seed()   # every draw in this step

        # ── Inner loop: T steps on z (GD on g) and y (GD on L_λ) ──────────────
        lag_grads = [torch.zeros_like(w) for w in self.w_params]
        for t in range(self.inner_steps):
            # z_{t+1} = z_t − γ ∇_z g(x_k, z_t)
            g_z = self.lower_loss_fn(self.z_model, data, self.ul_model,
                                     self.stochastic, seed, **kwargs)
            gz_grads = self._filled(
                torch.autograd.grad(g_z, self.z_params, allow_unused=True), self.z_params)
            with torch.no_grad():
                for z, g in zip(self.z_params, gz_grads):
                    z.sub_(self.lr_z * g)

            # y_{t+1} = y_t − α (∇_y f(x_k, y_t) + λ_k ∇_y g(x_k, y_t))
            g_y = self.lower_loss_fn(self.ll_model, data, self.ul_model,
                                     self.stochastic, seed, **kwargs)
            f_y = self.upper_loss_fn(self.ll_model, data, self.ul_model,
                                     self.stochastic, seed, **kwargs)
            lag_grads = self._filled(
                torch.autograd.grad(f_y + self.lambda_val * g_y, self.w_params,
                                    allow_unused=True),
                self.w_params)
            with torch.no_grad():
                for w, g in zip(self.w_params, lag_grads):
                    w.sub_(self.lr_w * g)

        # ── Outer x update using y_{k+1} and z_{k+1} ───────────────────────────
        # x_{k+1} = x_k − lr_x (∇_x f(x_k,y_T) + λ_k (∇_x g(x_k,y_T) − ∇_x g(x_k,z_T)))
        f_x   = self.upper_loss_fn(self.ll_model, data, self.ul_model,
                                    self.stochastic, seed, **kwargs)
        g_xy  = self.lower_loss_fn(self.ll_model, data, self.ul_model,
                                    self.stochastic, seed, **kwargs)
        g_xz  = self.lower_loss_fn(self.z_model,  data, self.ul_model,
                                    self.stochastic, seed, **kwargs)

        grad_f_x   = self._filled(torch.autograd.grad(f_x,  self.theta_params, allow_unused=True),
                                   self.theta_params)
        grad_gxy_x = self._filled(torch.autograd.grad(g_xy, self.theta_params, allow_unused=True),
                                   self.theta_params)
        grad_gxz_x = self._filled(torch.autograd.grad(g_xz, self.theta_params, allow_unused=True),
                                   self.theta_params)

        direction = [gf + self.lambda_val * (gxy - gxz)
                     for gf, gxy, gxz in zip(grad_f_x, grad_gxy_x, grad_gxz_x)]

        if self.grad_clip is not None:
            total = torch.sqrt(sum((g.detach() ** 2).sum() for g in direction))
            if total > self.grad_clip:
                scale = self.grad_clip / (total + 1e-12)
                direction = [g * scale for g in direction]

        with torch.no_grad():
            for th, d in zip(self.theta_params, direction):
                th.sub_(self.lr_x * d)

        # ── λ update ────────────────────────────────────────────────────────────
        self.lambda_val += self.delta

        if self.debug:
            print(f"{self.iter_counter}: LL={g_xy.detach():.4f}  UL={f_x.detach():.4f}  "
                  f"λ={self.lambda_val:.4f}  "
                  f"|grad_y L|={sqrt(sum(g.norm().item()**2 for g in lag_grads)):.4f}  "
                  f"|grad_x|={sqrt(sum(g.norm().item()**2 for g in direction)):.4f}")

        return {
            'll_loss':           g_xy.detach().item(),
            'ul_loss':           f_x.detach().item(),
            'lambda':            self.lambda_val,
            'grad_w_L_norm':     sqrt(sum(g.norm().item()**2 for g in lag_grads)),
            'grad_theta_L_norm': sqrt(sum(g.norm().item()**2 for g in direction)),
            'lambda_norm':       abs(self.lambda_val),
        }


# =============================================================================
#  V-PBGD  (Value-function gap based Penalized Bilevel Gradient Descent)
# =============================================================================
#  Reference: Shen & Chen, "On Penalty-based Bilevel Gradient Descent Method",
#  ICML 2023, Algorithm 2.
#
#  Problem: min_{x,y} f(x,y)  s.t.  x ∈ C,  y ∈ argmin_{y'} g(x, y')
#
#  Penalty reformulation with the function-value gap p(x,y) = g(x,y) − v(x):
#      F_γ(x,y) = f(x,y) + γ (g(x,y) − v(x)),    v(x) = min_{y'} g(x,y')
#
#  Since ∇v(x) = ∇_x g(x, y*), it is approximated via T_k inner GD steps on g:
#      ω_0 = y_k,   ω_{t+1} = ω_t − β ∇_ω g(x_k, ω_t)  →  ŷ_k ≈ y*(x_k)
#  so ∇v(x_k) ≈ ∇_x g(x_k, ŷ_k).
#
#  Outer update (Algorithm 2, eq. 7b):
#      ∇_x F_γ ≈ ∇_x f(x_k,y_k) + γ (∇_x g(x_k,y_k) − ∇_x g(x_k,ŷ_k))
#      ∇_y F_γ  = ∇_y f(x_k,y_k) + γ ∇_y g(x_k,y_k)
#      (x_{k+1}, y_{k+1}) = (x_k,y_k) − α (∇_x F_γ, ∇_y F_γ)
#
#  Notation map:  x → ul_model (θ),  y → ll_model (w),  ω → omega_model (copy of w)
#                 g → lower_loss_fn,  f → upper_loss_fn
# =============================================================================


class BLO_optimiser_VPBGD:
    """V-PBGD — Value-function gap Penalized Bilevel GD (Shen & Chen, ICML 2023).

    See the section header above for the algorithm and notation.

    Hyperparameters
    ---------------
    lr_w        (α)   : outer step size for y (lower-level variable)
    lr_theta    (α)   : outer step size for x (upper-level variable; defaults to lr_w)
    inner_lr    (β)   : inner GD step size for ω (approximating y*(x))
    gamma       (γ)   : penalty coefficient on the function-value gap g(x,y)−v(x)
    inner_steps (T_k) : inner GD iterations per outer step
    """

    def __init__(self, ll_model, lower_loss_fn, upper_loss_fn, ul_model=None,
                 stochastic=False, device=None, debug=False,
                 lr_w=1e-3, lr_theta=None, inner_lr=1e-3, gamma=1.0,
                 inner_steps=5, grad_clip=None, seed=0):
        self.ll_model = ll_model
        self.ul_model = ul_model
        self.lower_loss_fn = lower_loss_fn
        self.upper_loss_fn = upper_loss_fn
        self.stochastic = stochastic
        self.device = device or next(ll_model.parameters()).device
        self.debug = debug
        self.grad_clip = grad_clip

        self.w_params = _trainable(ll_model)
        self.theta_params = _trainable(ul_model)

        # ω: inner copy that runs T_k steps of GD on g to approximate y*(x_k)
        self.omega_model = deepcopy(ll_model).to(self.device)
        self.omega_params = _trainable(self.omega_model)

        self.lr_w = lr_w
        self.lr_theta = lr_theta if lr_theta is not None else lr_w
        self.inner_lr = inner_lr
        self.gamma = gamma
        self.inner_steps = inner_steps
        self.iter_counter = 0
        self.seed = int(seed)   # run seed: offsets the sample stream

    @staticmethod
    def _filled(grads, params):
        return [g.detach() if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, params)]

    def _draw_seed(self):
        """The ONE sample seed used by every loss evaluation in this step.

        Run offset (self.seed * SEED_STRIDE) plus the iteration k = 1, 2, ...
        Under stochastic=False the loss functions ignore the seed; 0 is passed.
        """
        return self.seed * SEED_STRIDE + self.iter_counter if self.stochastic else 0

    def step(self, data=None, **kwargs):
        self.iter_counter += 1
        seed = self._draw_seed()   # every draw in this step

        # ── Inner loop: ω_0 = y_k, then T_k steps of GD on g(x_k, ω) ──────────
        with torch.no_grad():
            for om, w in zip(self.omega_params, self.w_params):
                om.copy_(w)

        for t in range(self.inner_steps):
            g_om = self.lower_loss_fn(self.omega_model, data, self.ul_model,
                                      self.stochastic, seed, **kwargs)
            gom_grads = self._filled(
                torch.autograd.grad(g_om, self.omega_params, allow_unused=True),
                self.omega_params)
            with torch.no_grad():
                for om, g in zip(self.omega_params, gom_grads):
                    om.sub_(self.inner_lr * g)

        # ── Outer update: compute gradients at (x_k, y_k) and (x_k, ŷ_k) ───────
        f_y   = self.upper_loss_fn(self.ll_model,   data, self.ul_model,
                                    self.stochastic, seed, **kwargs)
        g_y   = self.lower_loss_fn(self.ll_model,   data, self.ul_model,
                                    self.stochastic, seed, **kwargs)
        g_hat = self.lower_loss_fn(self.omega_model, data, self.ul_model,
                                    self.stochastic, seed, **kwargs)

        # ∇_y (f + γg) — retain graph so f_y and g_y are still alive for x grads
        penalized = f_y + self.gamma * g_y
        grad_y = self._filled(
            torch.autograd.grad(penalized, self.w_params, allow_unused=True, retain_graph=True),
            self.w_params)

        # ∇_x f and ∇_x g(y_k): use f_y and g_y (still alive)
        grad_f_x   = self._filled(
            torch.autograd.grad(f_y,  self.theta_params, allow_unused=True, retain_graph=True),
            self.theta_params)
        grad_gy_x  = self._filled(
            torch.autograd.grad(g_y,  self.theta_params, allow_unused=True),
            self.theta_params)

        # ∇_x g(ŷ_k): independent graph
        grad_ghat_x = self._filled(
            torch.autograd.grad(g_hat, self.theta_params, allow_unused=True),
            self.theta_params)

        # ∇_x F_γ ≈ ∇_x f + γ (∇_x g(y_k) − ∇_x g(ŷ_k))
        direction_x = [gf + self.gamma * (gxy - gxz)
                       for gf, gxy, gxz in zip(grad_f_x, grad_gy_x, grad_ghat_x)]

        if self.grad_clip is not None:
            total = torch.sqrt(sum((g.detach() ** 2).sum()
                                   for g in list(grad_y) + direction_x))
            if total > self.grad_clip:
                scale = self.grad_clip / (total + 1e-12)
                grad_y     = [g * scale for g in grad_y]
                direction_x = [g * scale for g in direction_x]

        with torch.no_grad():
            for w, g in zip(self.w_params, grad_y):
                w.sub_(self.lr_w * g)
            for th, d in zip(self.theta_params, direction_x):
                th.sub_(self.lr_theta * d)

        if self.debug:
            penalty = (g_y - g_hat).detach().item()
            print(f"{self.iter_counter}: LL={g_y.detach():.4f}  UL={f_y.detach():.4f}  "
                  f"gap={penalty:.4f}  "
                  f"|grad_y|={sqrt(sum(g.norm().item()**2 for g in grad_y)):.4f}  "
                  f"|grad_x|={sqrt(sum(g.norm().item()**2 for g in direction_x)):.4f}")

        return {
            'll_loss':           g_y.detach().item(),
            'ul_loss':           f_y.detach().item(),
            'penalty':           (g_y - g_hat).detach().item(),
            'grad_w_L_norm':     sqrt(sum(g.norm().item()**2 for g in grad_y)),
            'grad_theta_L_norm': sqrt(sum(g.norm().item()**2 for g in direction_x)),
            'lambda_norm':       0.0,
        }
