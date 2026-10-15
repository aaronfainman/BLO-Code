"""Ridge regression bilevel problem.

Lower level (ridge regression on training data):
    J(w, θ) = 1/(2N) ‖X_t w − y_t‖² + ½ θ ‖w‖²

Upper level (validation MSE):
    F(w, θ) = 1/(2M) ‖X_v w − y_v‖²

Data-generating process:
    h_i ~ N(0, Σ),   y_i = hᵢᵀ w_true + ε_i,   ε_i ~ N(0, σ_noise²)

Σ is passed explicitly so the caller controls the problem conditioning.

Analytical lower-level solution for fixed θ:
    w*(θ) = (XᵢᵀXₜ / N + θ I)⁻¹ Xₜᵀ yₜ / N

This module mirrors toy_lib.py so the same sweep / summary machinery applies.
w is 2-D; θ is a scalar.
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

# ── default data-generation settings ─────────────────────────────────────────
W_TRUE_DEFAULT = np.array([1.0, 1.0])
NOISE_STD_DEFAULT = 0.1
N_TRAIN_DEFAULT   = 200
N_VAL_DEFAULT     = 200


# =============================================================================
# Model classes
# =============================================================================

class RidgeLower(nn.Module):
    """Holds the 2-D weight vector w."""
    def __init__(self, w0=None):
        super().__init__()
        if w0 is None:
            w0 = np.zeros(2)
        self.omega = nn.Parameter(
            torch.tensor(np.asarray(w0, dtype=np.float32)).clone()
        )


class RidgeUpper(nn.Module):
    """Holds the scalar regularisation strength θ."""
    def __init__(self, theta0=1.0):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(theta0)))


# =============================================================================
# Data generation
# =============================================================================

def make_data(cov, w_true=None, n_train=N_TRAIN_DEFAULT, n_val=N_VAL_DEFAULT,
              noise_std=NOISE_STD_DEFAULT, seed=0):
    """Generate train / validation tensors for the ridge BLO problem.

    cov     : (2, 2) array  — feature covariance Σ  (must be positive definite)
    w_true  : (2,) array    — true regression weights  [default: [1, 1]]
    Returns : dict with torch tensors  'X_train', 'y_train', 'X_val', 'y_val'
    """
    if w_true is None:
        w_true = W_TRUE_DEFAULT
    cov    = np.asarray(cov,    dtype=np.float64)
    w_true = np.asarray(w_true, dtype=np.float64)
    L   = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)

    z_t = rng.standard_normal((n_train, 2))
    X_t = z_t @ L.T
    y_t = X_t @ w_true + rng.standard_normal(n_train) * noise_std

    z_v = rng.standard_normal((n_val, 2))
    X_v = z_v @ L.T
    y_v = X_v @ w_true + rng.standard_normal(n_val) * noise_std

    return {
        'X_train': torch.tensor(X_t, dtype=torch.float32),
        'y_train': torch.tensor(y_t, dtype=torch.float32),
        'X_val':   torch.tensor(X_v, dtype=torch.float32),
        'y_val':   torch.tensor(y_v, dtype=torch.float32),
    }


# =============================================================================
# Loss functions (closures over data)
# =============================================================================

def make_losses(data):
    """Return (lower_loss_fn, upper_loss_fn) closed over pre-computed data.

    Signature matches the rest of the BLO codebase:
        fn(ll_model, data, ul_model, stochastic, seed, **kwargs) -> scalar tensor
    The positional `data` / `stochastic` / `seed` args are accepted but ignored.
    """
    X_t = data['X_train']   # (N, 2)
    y_t = data['y_train']   # (N,)
    X_v = data['X_val']     # (M, 2)
    y_v = data['y_val']     # (M,)

    def lower(ll_model, _data, ul_model, stochastic, seed, **kwargs):
        w     = ll_model.omega    # (2,)
        theta = ul_model.theta    # scalar
        res   = X_t @ w - y_t    # (N,)
        return 0.5 * (res ** 2).mean() + 0.5 * theta * (w ** 2).sum()

    def upper(ll_model, _data, ul_model, stochastic, seed, **kwargs):
        w   = ll_model.omega
        res = X_v @ w - y_v
        return 0.5 * (res ** 2).mean()

    return lower, upper


# =============================================================================
# Analytical solution
# =============================================================================

def w_star(data, theta):
    """w*(θ) = (XᵀX/N + θI)⁻¹ Xᵀy/N  (closed-form for fixed θ)."""
    X = data['X_train'].numpy().astype(np.float64)
    y = data['y_train'].numpy().astype(np.float64)
    N = len(y)
    H = X.T @ X / N
    r = X.T @ y / N
    return np.linalg.solve(H + theta * np.eye(2), r)


def upper_loss_numpy(data, theta):
    """F(w*(θ)) evaluated on the validation set."""
    w   = w_star(data, theta)
    X_v = data['X_val'].numpy().astype(np.float64)
    y_v = data['y_val'].numpy().astype(np.float64)
    return 0.5 * np.mean((X_v @ w - y_v) ** 2)


def optimal_theta(data, theta_grid=None):
    """θ* = argmin_{θ>0} F(w*(θ))  found by dense grid search."""
    if theta_grid is None:
        theta_grid = np.logspace(-4, 2, 2000)
    losses = np.array([upper_loss_numpy(data, th) for th in theta_grid])
    return float(theta_grid[np.argmin(losses)])


def optimal_solution(data):
    """Returns (w_opt, theta_opt) for the given data realisation."""
    th = optimal_theta(data)
    return w_star(data, th), th


# =============================================================================
# Optimiser factory  (mirrors toy_lib.build)
# =============================================================================

def build(method, cfg, ll, ul, data):
    """Construct the BLO optimiser for `method` with hyperparameters from `cfg`."""
    lower, upper = make_losses(data)
    common = dict(ll_model=ll, ul_model=ul, lower_loss_fn=lower, upper_loss_fn=upper)
    g = cfg.get

    if method == "PABLO":
        tau = g("tau", 0.02)
        ok  = {"alpha": g("alpha", 0.99), "eps": g("eps", 1e-8)}
        lr_w    = g("lr_w",    g("lr", 0.01))
        lr_theta = g("lr_theta", g("lr", 0.01))
        return BLO_optimiser_PABLO_Prox_Bound(
            **common,
            w_optimiser=torch.optim.RMSprop, theta_optimiser=torch.optim.RMSprop,
            w_optimiser_kwargs=ok, theta_optimiser_kwargs=ok,
            lr_w=lr_w, lr_theta=lr_theta,
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

def run_single(method, cfg, cov, w0=0.0, theta0=1.0, n_iters=2000,
               data_seed=0, data_kwargs=None):
    """Run one trajectory.

    w0     : scalar or (2,) array — initial w; scalar s is broadcast to s * ones(2)
    theta0 : scalar               — initial θ
    data_seed : fixed seed used to generate the dataset (same for all starts)

    Returns dict with keys:
        w          (n_iters, 2),  theta  (n_iters,)
        ul_loss    (n_iters,),    ll_loss (n_iters,)
        ul_opt     float,         ll_opt  float
        w_opt      (2,),          theta_opt  float
        diverged   bool
    """
    torch.set_num_threads(1)
    dkw  = data_kwargs or {}
    data = make_data(np.asarray(cov), seed=data_seed, **dkw)
    w_opt, theta_opt = optimal_solution(data)

    ul_opt = float(upper_loss_numpy(data, theta_opt))
    X_t = data['X_train'].numpy().astype(np.float64)
    y_t = data['y_train'].numpy().astype(np.float64)
    ll_opt = float(0.5 * np.mean((X_t @ w_opt - y_t) ** 2)
                   + 0.5 * theta_opt * np.dot(w_opt, w_opt))

    w0_vec = np.broadcast_to(np.asarray(w0, dtype=np.float32), (2,)).copy()
    ll = RidgeLower(w0_vec)
    ul = RidgeUpper(float(theta0))

    try:
        opt = build(method, cfg, ll, ul, data)
    except Exception as exc:
        return {"diverged": True, "error": repr(exc),
                "w": np.empty((0, 2)), "theta": np.array([]),
                "ul_loss": np.array([]), "ll_loss": np.array([]),
                "ul_opt": ul_opt, "ll_opt": ll_opt,
                "w_opt": w_opt, "theta_opt": theta_opt}

    w_arr     = np.empty((n_iters, 2), dtype=np.float64)
    theta_arr = np.empty(n_iters,      dtype=np.float64)
    ul_arr    = np.empty(n_iters,      dtype=np.float64)
    ll_arr    = np.empty(n_iters,      dtype=np.float64)

    for i in range(n_iters):
        try:
            stats = opt.step(None, seed=i) if method in NEEDS_SEED else opt.step(None)
        except Exception as exc:
            return {"diverged": True, "error": repr(exc),
                    "w": w_arr[:i], "theta": theta_arr[:i],
                    "ul_loss": ul_arr[:i], "ll_loss": ll_arr[:i],
                    "ul_opt": ul_opt, "ll_opt": ll_opt,
                    "w_opt": w_opt, "theta_opt": theta_opt}
        w_arr[i]     = ll.omega.detach().numpy()
        theta_arr[i] = ul.theta.item()
        ll_arr[i]    = stats["ll_loss"]
        ul_arr[i]    = stats["ul_loss"]
        if not (np.all(np.isfinite(w_arr[i])) and np.isfinite(theta_arr[i])):
            return {"diverged": True, "error": "non-finite iterate",
                    "w": w_arr[:i + 1], "theta": theta_arr[:i + 1],
                    "ul_loss": ul_arr[:i + 1], "ll_loss": ll_arr[:i + 1],
                    "ul_opt": ul_opt, "ll_opt": ll_opt,
                    "w_opt": w_opt, "theta_opt": theta_opt}

    return {"diverged": False, "error": None,
            "w": w_arr, "theta": theta_arr,
            "ul_loss": ul_arr, "ll_loss": ll_arr,
            "ul_opt": ul_opt, "ll_opt": ll_opt,
            "w_opt": w_opt, "theta_opt": theta_opt}


# =============================================================================
# Parallel sweep  (mirrors toy_lib.run_sweep)
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
    method, cfg, cov_list, w0, t0, n_iters, data_seed, dkw = args
    cov = np.array(cov_list)
    r   = run_single(method, cfg, cov, w0, t0, n_iters, data_seed, dkw)
    if r["diverged"]:
        return (method, w0, t0, None, None, None, None,
                r["w_opt"], r["theta_opt"], r["ul_opt"], r["ll_opt"])
    return (method, w0, t0,
            r["w"].astype(np.float32),
            r["theta"].astype(np.float32),
            r["ul_loss"].astype(np.float32),
            r["ll_loss"].astype(np.float32),
            r["w_opt"], r["theta_opt"], r["ul_opt"], r["ll_opt"])


def run_sweep(configs, cov,
              w0_lo=-2.0, w0_hi=2.0, w0_n=10,
              theta0_lo=0.1, theta0_hi=5.0, theta0_n=None,
              n_iters=2000, data_seed=0, data_kwargs=None,
              n_jobs=None, progress=True):
    """Run every method in `configs` from a w0_n × theta0_n grid of starts.

    w0 is a scalar initialiser; each start uses w = w0 * ones(2).
    theta0 is the scalar initial regularisation strength.

    configs    : {method_name: cfg_dict}
    cov        : (2, 2) array — feature covariance (controls conditioning)
    data_seed  : fixed seed for dataset generation (same dataset for all starts)

    Returns {method: {"w": (runs, iters, 2), "theta": (runs, iters),
                       "ul_loss": ..., "ll_loss": ...,
                       "excess_ul_loss": ..., "excess_ll_loss": ...,
                       "starts": [(w0, t0), ...],
                       "w_opt": (2,), "theta_opt": float,
                       "diverged": int, "total": int}}
    """
    if theta0_n is None:
        theta0_n = w0_n
    w0_grid  = np.linspace(w0_lo,     w0_hi,     w0_n)
    t0_grid  = np.linspace(theta0_lo, theta0_hi, theta0_n)
    cov_list = np.asarray(cov).tolist()   # plain Python list — pickling-safe
    dkw      = data_kwargs or {}

    jobs = [
        (m, cfg, cov_list, float(w0), float(t0), n_iters, data_seed, dkw)
        for m, cfg in configs.items()
        for w0 in w0_grid
        for t0 in t0_grid
    ]

    kappa = np.linalg.cond(np.asarray(cov))
    if progress:
        print(f"κ(Σ) ≈ {kappa:.1f};  {w0_n}×{theta0_n} = {w0_n*theta0_n} starts;  "
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

    total = w0_n * theta0_n
    out = {m: {"w": [], "theta": [], "ul_loss": [], "ll_loss": [],
               "excess_ul_loss": [], "excess_ll_loss": [],
               "starts": [], "diverged": 0, "total": total,
               "_w_opt_acc": [], "_theta_opt_acc": []}
           for m in configs}

    for method, w0, t0, w, th, ul, ll, wo, to, ul_opt, ll_opt in raw:
        d = out[method]
        if w is None:
            d["diverged"] += 1
            d["_w_opt_acc"].append(wo)
            d["_theta_opt_acc"].append(to)
            continue
        d["w"].append(w)
        d["theta"].append(th)
        d["ul_loss"].append(ul)
        d["ll_loss"].append(ll)
        d["excess_ul_loss"].append(ul - ul_opt)
        d["excess_ll_loss"].append(ll - ll_opt)
        d["starts"].append((w0, t0))
        d["_w_opt_acc"].append(wo)
        d["_theta_opt_acc"].append(to)

    for m in out:
        d = out[m]
        d["w"]             = np.array(d["w"])             if d["w"]             else np.empty((0, 0, 2))
        d["theta"]         = np.array(d["theta"])         if d["theta"]         else np.empty((0, 0))
        d["ul_loss"]       = np.array(d["ul_loss"])       if d["ul_loss"]       else np.empty((0, 0))
        d["ll_loss"]       = np.array(d["ll_loss"])       if d["ll_loss"]       else np.empty((0, 0))
        d["excess_ul_loss"] = np.array(d["excess_ul_loss"]) if d["excess_ul_loss"] else np.empty((0, 0))
        d["excess_ll_loss"] = np.array(d["excess_ll_loss"]) if d["excess_ll_loss"] else np.empty((0, 0))
        all_wo = [x for x in d["_w_opt_acc"] if x is not None]
        all_to = [x for x in d["_theta_opt_acc"] if x is not None]
        d["w_opt"]     = np.mean(all_wo, axis=0) if all_wo else np.zeros(2)
        d["theta_opt"] = float(np.median(all_to)) if all_to else float("nan")
        del d["_w_opt_acc"], d["_theta_opt_acc"]
    return out


# =============================================================================
# Summarise / table  (mirrors toy_lib.summarise)
# =============================================================================

def summarise(sweep, tol=1e-3):
    """Per-method metrics over all starts.

    Convergence is declared when ‖w_t − w*‖ < tol  AND  |θ_t − θ*| < tol.
    Fractions are over ALL starts (diverged runs count as unsolved).
    Medians are over COMPLETED runs only.
    """
    summary = {}
    for m, d in sweep.items():
        total = d["total"]
        empty = dict(runs=0, total=total, frac_diverged=1.0, frac_solved=0.0,
                     frac_completed=0.0,
                     median_final_w_err=np.nan, median_final_theta_err=np.nan,
                     median_curve_UL=None, median_curve_LL=None,
                     median_curve_w_err=None, median_curve_theta_err=None,
                     success_curve=None)
        if len(d["w"]) == 0:
            summary[m] = empty
            continue

        wo = d["w_opt"]           # (2,)
        to = d["theta_opt"]       # float

        W_err = np.linalg.norm(d["w"] - wo[None, None, :], axis=-1)  # (runs, iters)
        T_err = np.abs(d["theta"] - to)                                # (runs, iters)
        UL    = d["ul_loss"]
        LL    = d["ll_loss"]

        solved = (W_err[:, -1] <= tol) & (T_err[:, -1] <= tol)
        summary[m] = {
            "runs":                    len(W_err),
            "total":                   total,
            "frac_completed":          len(W_err) / total,
            "frac_diverged":           d["diverged"] / total,
            "frac_solved":             float(solved.sum()) / total,
            "median_final_w_err":      float(np.median(W_err[:, -1])),
            "median_final_theta_err":  float(np.median(T_err[:, -1])),
            "median_curve_UL":         np.median(UL, axis=0),
            "median_curve_LL":         np.median(LL, axis=0),
            "median_curve_w_err":      np.median(W_err, axis=0),
            "median_curve_theta_err":  np.median(T_err, axis=0),
            "success_curve":           (W_err <= tol).all(axis=-1).mean()
                                       if W_err.ndim > 1 else np.nan,
        }
    return summary


def iters_to_converge(summary, tol_w=1e-3, tol_theta=1e-3):
    """First iteration t where median ‖w_t − w*‖ < tol_w AND median |θ_t − θ*| < tol_theta.
    Returns nan if not reached."""
    result = {}
    for m, s in summary.items():
        if s["median_curve_w_err"] is None:
            result[m] = np.nan
            continue
        we = s["median_curve_w_err"]
        te = s["median_curve_theta_err"]
        hit = np.where((we < tol_w) & (te < tol_theta))[0]
        result[m] = int(hit[0] + 1) if len(hit) else np.nan
    return result


def summary_table(summary, order=None):
    """Plain-text comparison table."""
    order = order or list(summary)
    header = (f"{'method':<12} {'solved':>8} {'diverged':>9} "
              f"{'med |w−w*|':>12} {'med |θ−θ*|':>12}")
    lines = [header, "-" * len(header)]
    for m in order:
        s = summary[m]
        lines.append(
            f"{m:<12} {s['frac_solved']*100:>7.1f}% "
            f"{s['frac_diverged']*100:>8.1f}% "
            f"{s['median_final_w_err']:>12.3e} "
            f"{s['median_final_theta_err']:>12.3e}"
        )
    lines.append("")
    lines.append("fractions are over ALL starts; medians over completed runs only")
    return "\n".join(lines)
