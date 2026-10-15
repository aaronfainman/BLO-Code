"""Quadratic bilevel problem with explicit 2×2 diagonal matrices.

Upper level:  F(w, θ) = wᵀ A w + θᵀ B θ
Lower level:  J(w, θ) = wᵀ C w + wᵀ D θ

w ∈ ℝ², θ ∈ ℝ²;  A, B, C, D are 2×2 diagonal matrices.

Analytical structure:
  ∇_w J = 2Cw + Dθ = 0  →  w*(θ) = −½ C⁻¹ D θ
  Substituting into F:     F(w*(θ), θ) = θᵀ M θ,
                            M = ¼ D C⁻¹ A C⁻¹ D + B
  Minimising over θ:       2Mθ = 0  →  θ* = 0,  w* = 0
  (provided A, B, C with positive diagonals so M ≻ 0)

Convergence is therefore tracked via  ‖w_t‖ + ‖θ_t‖.
"""

import multiprocessing as mp
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blo_optimisers import (
    BLO_optimiser_AAL,
    BLO_optimiser_BOME,
    BLO_optimiser_PABLO_Prox_Bound,
    BLO_optimiser_PBGD_Free,
    BLO_optimiser_STABLE,
    BLO_optimiser_TSP,
)

METHODS    = ["PABLO", "BOME", "TSP", "STABLE", "AAL", "PBGD-Free"]
NEEDS_SEED = {"BOME", "TSP"}

OPTIMISERS = {
    "sgd":     lambda params, lr: torch.optim.SGD(params, lr=lr),
    "sgdm":    lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
    "adam":    lambda params, lr: torch.optim.Adam(params, lr=lr),
    "rmsprop": lambda params, lr: torch.optim.RMSprop(params, lr=lr),
}


# =============================================================================
# Model classes
# =============================================================================

class NonDiagLower(nn.Module):
    """Lower-level variable w ∈ ℝ²."""
    def __init__(self, w0=None):
        super().__init__()
        if w0 is None:
            w0 = np.zeros(2)
        self.omega = nn.Parameter(
            torch.tensor(np.asarray(w0, dtype=np.float32)).clone()
        )


class NonDiagUpper(nn.Module):
    """Upper-level variable θ ∈ ℝ²."""
    def __init__(self, theta0=None):
        super().__init__()
        if theta0 is None:
            theta0 = np.ones(2)
        self.theta = nn.Parameter(
            torch.tensor(np.asarray(theta0, dtype=np.float32)).clone()
        )


# =============================================================================
# Loss functions
# =============================================================================

def make_losses(A, B, C, D):
    """Return (lower_loss_fn, upper_loss_fn) closed over the problem matrices.

    Accepts 2×2 numpy arrays or plain lists.  Internally converts to float32
    tensors; no gradients flow through the matrices themselves.

    Signature matches the rest of the BLO codebase:
        fn(ll_model, data, ul_model, stochastic, seed, **kwargs) -> scalar
    """
    At = torch.tensor(np.asarray(A, dtype=np.float32))
    Bt = torch.tensor(np.asarray(B, dtype=np.float32))
    Ct = torch.tensor(np.asarray(C, dtype=np.float32))
    Dt = torch.tensor(np.asarray(D, dtype=np.float32))

    def lower(ll_model, _data, ul_model, stochastic, seed, **kwargs):
        w = ll_model.omega    # (2,)
        t = ul_model.theta    # (2,)
        return w @ (Ct @ w) + w @ (Dt @ t)   # wᵀCw + wᵀDθ

    def upper(ll_model, _data, ul_model, stochastic, seed, **kwargs):
        w = ll_model.omega
        t = ul_model.theta
        return w @ (At @ w) + t @ (Bt @ t)   # wᵀAw + θᵀBθ

    return lower, upper


# =============================================================================
# Analytical solution and diagnostics
# =============================================================================

def analytical_solution():
    """Optimal (w*, θ*) = (0, 0) for any PD A, B, C, D."""
    return np.zeros(2), np.zeros(2)


def problem_diagnostics(A, B, C, D, verbose:bool=True):
    """Print condition numbers and the effective upper-level Hessian M."""
    A, B, C, D = (np.asarray(x, dtype=np.float64) for x in (A, B, C, D))
    Cinv = np.diag(1.0 / np.diag(C))           # C⁻¹ for diagonal C
    M    = 0.25 * D @ Cinv @ A @ Cinv @ D + B  # curvature of F(w*(θ),θ) in θ
    KKT_CURVE = np.block([[2*A, np.zeros_like(A), 2*C], [np.zeros_like(A), 2*B, D], [2*C, D, np.zeros_like(A)]])
    if verbose:
        print("Condition numbers:")
        print(f"  κ(A) = {np.linalg.cond(A):.3g}")
        print(f"  κ(B) = {np.linalg.cond(B):.3g}")
        print(f"  κ(C) = {np.linalg.cond(C):.3g}  ← lower-level curvature in w")
        print(f"  κ(D) = {np.linalg.cond(D):.3g}  ← coupling strength")
        print(f"Effective upper-level curvature M = ¼ D C⁻¹ A C⁻¹ D + B:")
        print(f"  diag(M) = {np.diag(M)}")
        print(f"  κ(M)    = {np.linalg.cond(M):.3g}")
        print(f"Analytical optimum: w* = 0,  θ* = 0")
        print(f"KKT Conditioning: {np.linalg.cond(KKT_CURVE):0.3g}")
    return M, KKT_CURVE


# =============================================================================
# Optimiser factory
# =============================================================================

def build(method, cfg, ll, ul, A, B, C, D):
    """Construct the BLO optimiser for `method` with hyperparameters `cfg`."""
    lower, upper = make_losses(A, B, C, D)
    common = dict(ll_model=ll, ul_model=ul, lower_loss_fn=lower, upper_loss_fn=upper)
    g = cfg.get

    if method == "PABLO":
        tau = g("tau", 0.02)
        ok  = {"alpha": g("alpha", 0.99), "eps": g("eps", 1e-8)}
        return BLO_optimiser_PABLO_Prox_Bound(
            **common,
            w_optimiser=torch.optim.RMSprop, theta_optimiser=torch.optim.RMSprop,
            w_optimiser_kwargs=ok, theta_optimiser_kwargs=ok,
            lr_w=g("lr", 0.01), lr_theta=g("lr", 0.01),
            lr_lambda=g("lr_lambda", 0.01), rho=g("rho", 0.01),
            tau=tau, gamma=g("y_rate", 0.5) / tau,
            lambda_bound=g("lambda_bound", 10.0), grad_clip=g("grad_clip", None),
            total_steps=g("total_steps", None), warmup_ratio=g("warmup_ratio", 0.1),
            eta_min=g("eta_min", 0.0))

    if method == "BOME":
        o  = OPTIMISERS[g("opt", "adam")]
        lr = g("lr", 0.01)
        return BLO_optimiser_BOME(
            **common, inner_steps=g("inner_steps", 10), inner_lr=g("inner_lr", 0.01),
            eta=g("eta", 0.5), lr_w=lr, lr_theta=lr, barrier=g("barrier", "grad_norm"),
            w_optimiser=lambda p, lr=lr: o(p, lr),
            theta_optimiser=lambda p, lr=lr: o(p, lr))

    if method == "TSP":
        return BLO_optimiser_TSP(
            **common, gamma=g("gamma", 0.1), delta=g("delta", 0.0), p=g("p", 1e-3),
            tau=g("dual_tau", 0.05), mu=g("mu", 0.5), theta=g("theta", 0.5),
            eta=g("eta", 0.05), alpha=g("step", 0.01), beta=g("step", 0.01),
            omega=g("omega", 0.05), lambda_init=g("lambda_init", 1.0))

    if method == "STABLE":
        return BLO_optimiser_STABLE(
            **common, lr_w=g("lr_w", 0.01), lr_theta=g("lr_theta", 0.01),
            tau=g("tau", 0.3), mu_g=g("mu_g", 0.05), C_gxy=g("C_gxy", 50.0))

    if method == "AAL":
        return BLO_optimiser_AAL(
            **common, L_f=g("L_f", 10.0), L_grad_g=g("L_grad_g", 10.0), L_H=g("L_H", 5.0),
            gamma=g("gamma", 1.0), rho0=g("rho0", 0.0), alpha=g("alpha", 1e-3),
            eps=g("eps", 1e-5), tau=g("tau", 1e-3), tau_decay=g("tau_decay", False))

    if method == "PBGD-Free":
        return BLO_optimiser_PBGD_Free(
            **common, gamma=g("gamma", 10.0), eta=g("eta", 1e-2),
            eta_gamma=g("eta_gamma", 1e-2), K=g("K", 1))

    raise ValueError(f"unknown method: {method}")


# =============================================================================
# Single-trajectory runner
# =============================================================================

def run_single(method, cfg, A, B, C, D, w0=1.0, theta0=1.0, n_iters=1000):
    """Run one trajectory from (w0*[1,1], theta0*[1,1]).

    w0, theta0 : scalars broadcast to 2-D starting vectors.
    Returns dict with w (n_iters, 2), theta (n_iters, 2), ul_loss, ll_loss, diverged.
    """
    torch.set_num_threads(1)
    w0_vec = np.full(2, float(w0),     dtype=np.float32)
    t0_vec = np.full(2, float(theta0), dtype=np.float32)
    ll = NonDiagLower(w0_vec)
    ul = NonDiagUpper(t0_vec)

    try:
        opt = build(method, cfg, ll, ul, A, B, C, D)
    except Exception as exc:
        return {"diverged": True, "error": repr(exc),
                "w": np.empty((0, 2)), "theta": np.empty((0, 2)),
                "ul_loss": np.array([]), "ll_loss": np.array([])}

    w_arr     = np.empty((n_iters, 2), dtype=np.float64)
    theta_arr = np.empty((n_iters, 2), dtype=np.float64)
    ul_arr    = np.empty(n_iters,      dtype=np.float64)
    ll_arr    = np.empty(n_iters,      dtype=np.float64)

    for i in range(n_iters):
        try:
            stats = opt.step(None, seed=i) if method in NEEDS_SEED else opt.step(None)
        except Exception as exc:
            return {"diverged": True, "error": repr(exc),
                    "w": w_arr[:i], "theta": theta_arr[:i],
                    "ul_loss": ul_arr[:i], "ll_loss": ll_arr[:i]}
        w_arr[i]     = ll.omega.detach().numpy()
        theta_arr[i] = ul.theta.detach().numpy()
        ll_arr[i]    = stats["ll_loss"]
        ul_arr[i]    = stats["ul_loss"]
        if not (np.all(np.isfinite(w_arr[i])) and np.all(np.isfinite(theta_arr[i]))):
            return {"diverged": True, "error": "non-finite iterate",
                    "w": w_arr[:i + 1], "theta": theta_arr[:i + 1],
                    "ul_loss": ul_arr[:i + 1], "ll_loss": ll_arr[:i + 1]}

    return {"diverged": False, "error": None,
            "w": w_arr, "theta": theta_arr,
            "ul_loss": ul_arr, "ll_loss": ll_arr}


# =============================================================================
# Parallel sweep
# =============================================================================

def _probe():
    return "ok"


def spawn_available(timeout=90):
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(1) as pool:
            return pool.apply_async(_probe).get(timeout=timeout) == "ok"
    except Exception:
        return False


def _sweep_worker(args):
    method, cfg, A_list, B_list, C_list, D_list, w0, t0, n_iters = args
    A, B, C, D = (np.array(x) for x in (A_list, B_list, C_list, D_list))
    r = run_single(method, cfg, A, B, C, D, w0, t0, n_iters)
    if r["diverged"]:
        return method, w0, t0, None, None, None, None
    return (method, w0, t0,
            r["w"].astype(np.float32),
            r["theta"].astype(np.float32),
            r["ul_loss"].astype(np.float32),
            r["ll_loss"].astype(np.float32))


def run_sweep(configs, A, B, C, D,
              grid_lo=-2.0, grid_hi=2.0, grid_n=10,
              n_iters=1000, n_jobs=None, progress=True):
    """Run all methods in `configs` from a grid_n × grid_n grid of starts.

    Starting points:  w0 = s_w * [1,1],  θ0 = s_θ * [1,1]
    where s_w, s_θ ∈ linspace(grid_lo, grid_hi, grid_n).

    configs : {method_name: cfg_dict}
    Returns {method: {"w": (runs, iters, 2), "theta": (runs, iters, 2),
                       "ul_loss": ..., "ll_loss": ...,
                       "starts": [(w0, t0), ...],
                       "diverged": int, "total": int}}
    """
    A_list, B_list = np.asarray(A).tolist(), np.asarray(B).tolist()
    C_list, D_list = np.asarray(C).tolist(), np.asarray(D).tolist()
    grid = np.linspace(grid_lo, grid_hi, grid_n)

    jobs = [
        (m, cfg, A_list, B_list, C_list, D_list, float(w0), float(t0), n_iters)
        for m, cfg in configs.items()
        for w0 in grid
        for t0 in grid
    ]

    if progress:
        M_val = problem_diagnostics(A, B, C, D)
        print(f"\n{grid_n}×{grid_n} = {grid_n**2} starts on [{grid_lo}, {grid_hi}]²;  "
              f"{n_iters} iters;  {len(jobs)} total runs")

    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    raw = None
    if n_jobs > 1 and spawn_available():
        try:
            ctx = mp.get_context("spawn")
            with ctx.Pool(n_jobs) as pool:
                raw = pool.map(_sweep_worker, jobs, chunksize=4)
        except Exception as exc:
            print(f"  (pool failed: {exc!r}; running serially)")
            raw = None
    if raw is None:
        raw = [_sweep_worker(j) for j in jobs]

    total = grid_n ** 2
    out = {m: {"w": [], "theta": [], "ul_loss": [], "ll_loss": [],
               "starts": [], "diverged": 0, "total": total}
           for m in configs}
    for method, w0, t0, w, th, ul, ll in raw:
        d = out[method]
        if w is None:
            d["diverged"] += 1
            continue
        d["w"].append(w)
        d["theta"].append(th)
        d["ul_loss"].append(ul)
        d["ll_loss"].append(ll)
        d["starts"].append((w0, t0))

    for m in out:
        d = out[m]
        d["w"]       = np.array(d["w"])       if d["w"]       else np.empty((0, 0, 2))
        d["theta"]   = np.array(d["theta"])   if d["theta"]   else np.empty((0, 0, 2))
        d["ul_loss"] = np.array(d["ul_loss"]) if d["ul_loss"] else np.empty((0, 0))
        d["ll_loss"] = np.array(d["ll_loss"]) if d["ll_loss"] else np.empty((0, 0))
    return out


# =============================================================================
# Summarise / table
# =============================================================================

def summarise(sweep, tol=1e-3):
    """Per-method metrics.  Optimal is (w*, θ*) = (0, 0).

    Convergence: ‖w_t‖ < tol  AND  ‖θ_t‖ < tol  (both 2-norms).
    """
    summary = {}
    for m, d in sweep.items():
        total = d["total"]
        empty = dict(runs=0, total=total, frac_diverged=1.0, frac_solved=0.0,
                     frac_completed=0.0,
                     median_final_w_norm=np.nan, median_final_theta_norm=np.nan,
                     median_curve_w_norm=None, median_curve_theta_norm=None,
                     median_curve_UL=None, median_curve_LL=None)
        if len(d["w"]) == 0:
            summary[m] = empty
            continue

        W_norm = np.linalg.norm(d["w"],     axis=-1)  # (runs, iters)
        T_norm = np.linalg.norm(d["theta"], axis=-1)  # (runs, iters)
        UL     = d["ul_loss"]
        LL     = d["ll_loss"]

        solved = (W_norm[:, -1] < tol) & (T_norm[:, -1] < tol)
        summary[m] = {
            "runs":                   len(W_norm),
            "total":                  total,
            "frac_completed":         len(W_norm) / total,
            "frac_diverged":          d["diverged"] / total,
            "frac_solved":            float(solved.sum()) / total,
            "median_final_w_norm":    float(np.median(W_norm[:, -1])),
            "median_final_theta_norm": float(np.median(T_norm[:, -1])),
            "median_curve_w_norm":    np.median(W_norm, axis=0),
            "median_curve_theta_norm": np.median(T_norm, axis=0),
            "median_curve_UL":        np.median(UL, axis=0),
            "median_curve_LL":        np.median(LL, axis=0),
            "success_curve":          ((W_norm < tol) & (T_norm < tol)).mean(axis=0),
        }
    return summary


def iters_to_converge(summary, tol_w=1e-3, tol_theta=1e-3):
    """First iteration t where median ‖w_t‖ < tol_w  AND  median ‖θ_t‖ < tol_theta."""
    result = {}
    for m, s in summary.items():
        if s["median_curve_w_norm"] is None:
            result[m] = np.nan
            continue
        we = s["median_curve_w_norm"]
        te = s["median_curve_theta_norm"]
        hit = np.where((we < tol_w) & (te < tol_theta))[0]
        result[m] = int(hit[0] + 1) if len(hit) else np.nan
    return result


def summary_table(summary, order=None):
    order = order or list(summary)
    header = (f"{'method':<12} {'solved':>8} {'diverged':>9} "
              f"{'med ‖w‖':>10} {'med ‖θ‖':>10}")
    lines = [header, "-" * len(header)]
    for m in order:
        s = summary[m]
        lines.append(
            f"{m:<12} {s['frac_solved']*100:>7.1f}% "
            f"{s['frac_diverged']*100:>8.1f}% "
            f"{s['median_final_w_norm']:>10.3e} "
            f"{s['median_final_theta_norm']:>10.3e}"
        )
    lines.append("\nfractions over ALL starts; medians over completed runs only")
    return "\n".join(lines)
