"""Toy bilevel problem + runners, shared by the scripts and Exp_ToyProblem_InitSweep.ipynb.

The problem (c controls the conditioning):

    F(w, theta) = (w+1)^4 - 2(w+1)^2 + (1/c)(theta - theta^2)^2 + 1      [upper]
    J(w, theta) = (w-1)^4 - 2(w-1)^2 + (1/(2c)) theta^2 + 1              [lower]

Structure, independent of c:
  * argmin_w J = {0, 2} for every theta, with a barrier at w = 1.
  * F(0,.) = -1 against F(2,.) = 63, so the bilevel solution is w* = 0.
    A start with w_0 > 1 begins in the WRONG lower-level basin: J pulls it to
    w = 2 while F pulls it to w = 0.
  * theta* minimises (theta - theta^2)^2, so theta* in {0, 1}; the theta error is
    min(|theta|, |theta - 1|).
  * dF/dtheta = (2/c)(theta - theta^2)(1 - 2theta) is exactly 0 at theta = 0.5, and
    d2J/dw dtheta = 0, so EVERY method's theta-direction vanishes at theta_0 = 0.5.
    Avoid that value if you want the theta dynamics to be live.

This module is import-safe and its worker is picklable, so run_sweep() can use
multiprocessing from inside a notebook.
"""
import multiprocessing as mp
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blo_optimisers import (BLO_optimiser_AAL, BLO_optimiser_BOME,
                            BLO_optimiser_PABLO_Prox_Bound, BLO_optimiser_PBGD_Free,
                            BLO_optimiser_STABLE, BLO_optimiser_TSP_Paper)

W_STAR = 0.0
METHODS = ["PABLO", "BOME", "TSP", "STABLE", "AAL", "PBGD-Free"]
NEEDS_SEED = {"BOME", "TSP"}          # these take seed= in step(); PABLO/STABLE do not

OPTIMISERS = {
    "sgd":     lambda params, lr: torch.optim.SGD(params, lr=lr),
    "sgdm":    lambda params, lr: torch.optim.SGD(params, lr=lr, momentum=0.9),
    "adam":    lambda params, lr: torch.optim.Adam(params, lr=lr),
    "rmsprop": lambda params, lr: torch.optim.RMSprop(params, lr=lr),
}


class ToyLower(nn.Module):
    def __init__(self, w0=0.5):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(float(w0)))


class ToyUpper(nn.Module):
    def __init__(self, theta0=0.5):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(theta0)))


def make_losses(c):
    """Returns (lower_loss_fn, upper_loss_fn) for conditioning c."""
    def lower(ll_model, data, ul_model, stochastic, seed, **kwargs):
        w, th = ll_model.omega, ul_model.theta
        return (w - 1) ** 4 - 2 * (w - 1) ** 2 + th ** 2 / (2 * c) + 1

    def upper(ll_model, data, ul_model, stochastic, seed, **kwargs):
        w, th = ll_model.omega, ul_model.theta
        return (w + 1) ** 4 - 2 * (w + 1) ** 2 + (th - th ** 2) ** 2 / c + 1
    return lower, upper


def theta_error(theta):
    """theta* in {0, 1}."""
    return np.minimum(np.abs(theta), np.abs(np.asarray(theta) - 1.0))


def build(method, cfg, ll, ul, c):
    """Construct an optimiser for `method` from the plain-dict `cfg`.

    Expected keys (anything omitted falls back to the optimiser's own default):

      PABLO   lr, alpha, eps, rho, lr_lambda, tau, y_rate, grad_clip,
              lambda_bound, total_steps, warmup_ratio
              (gamma is derived: gamma = y_rate / tau, so y <- y + y_rate (Z - y))
      BOME    lr, inner_lr, inner_steps, eta, opt in {sgd, sgdm, adam, rmsprop},
              barrier in {grad_norm, gap}
      TSP     gamma, delta, p, dual_tau, mu, theta, eta, step, omega, lambda_init
      STABLE  lr_w, lr_theta, tau, mu_g, C_gxy
      AAL     L_f, L_grad_g, L_H, gamma, rho0, alpha, eps, tau, tau_decay
              (step size 1/t_k with t_k from (3.8) of the paper)
      PBGD-Free  gamma, eta, eta_gamma, K
    """
    lower, upper = make_losses(c)
    common = dict(ll_model=ll, ul_model=ul, lower_loss_fn=lower, upper_loss_fn=upper)
    g = cfg.get

    if method == "PABLO":
        tau = g("tau", 0.02)
        opt_kwargs = {"alpha": g("alpha", 0.99), "eps": g("eps", 1e-8)}
        return BLO_optimiser_PABLO_Prox_Bound(
            **common,
            w_optimiser=torch.optim.RMSprop, theta_optimiser=torch.optim.RMSprop,
            w_optimiser_kwargs=opt_kwargs, theta_optimiser_kwargs=opt_kwargs,
            lr_w=g("lr", 0.01), lr_theta=g("lr", 0.01),
            lr_lambda=g("lr_lambda", 0.01), rho=g("rho", 0.01),
            tau=tau, gamma=g("y_rate", 0.5) / tau,
            lambda_bound=g("lambda_bound", 100.0), grad_clip=g("grad_clip", None),
            total_steps=g("total_steps", None), warmup_ratio=g("warmup_ratio", 0.1),
            eta_min=g("eta_min", 0.0))

    if method == "BOME":
        o = OPTIMISERS[g("opt", "adam")]
        lr = g("lr", 0.01)
        return BLO_optimiser_BOME(
            **common, inner_steps=g("inner_steps", 10), inner_lr=g("inner_lr", 0.01),
            eta=g("eta", 0.5), lr_w=lr, lr_theta=lr, barrier=g("barrier", "grad_norm"),
            w_optimiser=lambda p, lr=lr: o(p, lr), theta_optimiser=lambda p, lr=lr: o(p, lr))

    if method == "TSP":
        return BLO_optimiser_TSP_Paper(
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
            **common, L_f=g("L_f", 71.0), L_grad_g=g("L_grad_g", 71.0), L_H=g("L_H", 60.0),
            gamma=g("gamma", 1.0), rho0=g("rho0", 0.0), alpha=g("alpha", 1e-3),
            eps=g("eps", 1e-5), tau=g("tau", 1e-3), tau_decay=g("tau_decay", False))

    if method == "PBGD-Free":
        return BLO_optimiser_PBGD_Free(
            **common, gamma=g("gamma", 10.0), eta=g("eta", 1e-2),
            eta_gamma=g("eta_gamma", 1e-2), K=g("K", 1))

    raise ValueError(f"unknown method: {method}")


def run_single(method, cfg, c=10.0, w0=0.5, theta0=0.5, n_iters=2000, seed=0):
    """One trajectory. Returns dict with w, theta, ll_loss, ul_loss, diverged."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    ll, ul = ToyLower(w0), ToyUpper(theta0)
    try:
        opt = build(method, cfg, ll, ul, c)
    except Exception as exc:
        return {"diverged": True, "error": repr(exc), "w": np.array([]),
                "theta": np.array([]), "ll_loss": np.array([]), "ul_loss": np.array([])}

    w = np.empty(n_iters, dtype=np.float64)
    th = np.empty(n_iters, dtype=np.float64)
    ul_loss = np.empty(n_iters, dtype=np.float64)
    ll_loss = np.empty(n_iters, dtype=np.float64)
    for i in range(n_iters):
        try:
            stats = opt.step(None, seed=i) if method in NEEDS_SEED else opt.step(None)
        except Exception as exc:
            return {"diverged": True, "error": repr(exc), "w": w[:i], "theta": th[:i],
                    "ll_loss": ll_loss[:i], "ul_loss": ul_loss[:i]}
        w[i], th[i] = ll.omega.item(), ul.theta.item()
        ll_loss[i], ul_loss[i] = stats["ll_loss"], stats["ul_loss"]
        if not (np.isfinite(w[i]) and np.isfinite(th[i])):
            return {"diverged": True, "error": "non-finite iterate", "w": w[:i + 1],
                    "theta": th[:i + 1], "ll_loss": ll_loss[:i + 1], "ul_loss": ul_loss[:i + 1]}
    return {"diverged": False, "error": None, "w": w, "theta": th,
            "ll_loss": ll_loss, "ul_loss": ul_loss}


def _probe():
    """Trivial task used to check that worker processes can actually start."""
    return "ok"


def spawn_available(timeout=90):
    """True if a spawn worker can start and return. Guards against environments
    where the child re-imports an unrunnable __main__ (e.g. `python - <<EOF`)."""
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(1) as pool:
            return pool.apply_async(_probe).get(timeout=timeout) == "ok"
    except Exception:
        return False


def _sweep_worker(args):
    method, cfg, c, w0, t0, n_iters, seed = args
    r = run_single(method, cfg, c, w0, t0, n_iters, seed)
    if r["diverged"]:
        return method, w0, t0, None, None, None, None
    return method, w0, t0, r["w"].astype(np.float32), r["theta"].astype(np.float32), r["ul_loss"].astype(np.float32),  r["ll_loss"].astype(np.float32)


def run_sweep(configs, c=10.0, grid_lo=-1.5, grid_hi=1.5, grid_n=20,
              n_iters=2000, seed=0, n_jobs=None, progress=True):
    """Run every method in `configs` from a grid_n x grid_n grid of (w_0, theta_0).

    configs : {method_name: cfg_dict}
    Returns {method: {"w": (runs, iters), "theta": ..., "starts": [...], "diverged": int}}
    Trajectories are kept only for runs that completed; `diverged` counts the rest.
    """
    grid = np.linspace(grid_lo, grid_hi, grid_n)
    jobs = [(m, cfg, c, float(w0), float(t0), n_iters, seed)
            for m, cfg in configs.items() for w0 in grid for t0 in grid]
    if progress:
        print(f"c = {c:g};  {grid_n}x{grid_n} = {grid_n**2} starts on "
              f"[{grid_lo}, {grid_hi}]^2;  {n_iters} iterations;  {len(jobs)} runs")

    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    raw = None
    if n_jobs > 1:
        # Must be 'spawn': torch refuses fork-based multiprocessing once autograd's
        # threading has started. Spawn re-imports this module in each child, which is
        # why the worker lives here rather than in the notebook -- functions defined
        # in a notebook cell cannot be pickled for spawn. Probe first so that a
        # hostile environment degrades to serial instead of hanging.
        if spawn_available():
            try:
                ctx = mp.get_context("spawn")
                with ctx.Pool(n_jobs) as pool:
                    raw = pool.map(_sweep_worker, jobs, chunksize=4)
            except Exception as exc:
                print(f"  (parallel pool failed: {exc!r}; running serially)")
                raw = None
        else:
            print("  (worker processes unavailable here; running serially)")
    if raw is None:
        raw = [_sweep_worker(j) for j in jobs]

    out = {m: {"w": [], "theta": [], "starts": [], "diverged": 0, "ul_loss": [], "ll_loss": []} for m in configs}
    for method, w0, t0, w, th, ul_loss, ll_loss in raw:
        if w is None:
            out[method]["diverged"] += 1
            continue
        out[method]["w"].append(w)
        out[method]["theta"].append(th)
        out[method]["ul_loss"].append(ul_loss)
        out[method]["ll_loss"].append(ll_loss)
        out[method]["starts"].append((w0, t0))
    for m in out:
        out[m]["w"] = np.array(out[m]["w"])
        out[m]["theta"] = np.array(out[m]["theta"])
        out[m]["ul_loss"] = np.array(out[m]["ul_loss"])
        out[m]["ll_loss"] = np.array(out[m]["ll_loss"])
        out[m]["total"] = grid_n ** 2
    return out


def summarise(sweep, tol=1e-6, c=None):
    """Per-method metrics.

    Fractions are over ALL starts (a diverged run counts as not solved). Medians and
    geomeans are over the runs that COMPLETED, so compare them only alongside
    frac_diverged.
    """
    summary = {}
    for m, d in sweep.items():
        total = d["total"]
        if len(d["w"]) == 0:
            summary[m] = {"runs": 0, "total": total, "frac_diverged": 1.0,
                          "frac_solved": 0.0, "frac_wrong_basin": 0.0,
                          "frac_completed": 0.0, "median_final_w_err": np.nan,
                          "geomean_final_w_err": np.nan, "median_final_theta_err": np.nan,
                          "median_curve_w": None, "median_curve_theta": None,
                          "success_curve": None}
            continue
        W = np.abs(d["w"] - W_STAR)
        T = theta_error(d["theta"])
        w, th = d["w"].astype(np.float64), d["theta"].astype(np.float64)
        UL     = w**2 * (w + 2)**2 + (th - th**2)**2 / c    # F(w,θ) − F*, exact
        LL     = (w-1)**2 -2*(w -1)**2 +1/(2*c)* (th**2) + 1    # F(w,θ) − F*, exact
        LL_gap = np.abs(LL - 1/(2*c)* (th**2) )                # J(w,θ) − min_w J(·,θ), exact
        UL_gap = np.abs(UL)
        converged = np.isfinite(LL_gap).all(axis=1) & (np.abs(LL_gap) <= 1e4).all(axis=1)
        final = W[:, -1]
        summary[m] = {
            "runs": len(final), "total": total,
            "frac_completed": len(final) / total,
            "frac_diverged": d["diverged"] / total,
            "frac_solved": float(((UL_gap[:, -1] <= tol) & (LL_gap[:, -1] <= tol)).sum()) / total,
            "frac_wrong_basin": float((np.abs(d["w"][:, -1] - 2.0) < 1e-3).sum()) / total,
            "median_final_w_err": float(np.median(final)),
            "geomean_final_w_err": float(np.exp(np.mean(np.log(np.maximum(final, 1e-16))))),
            "median_final_theta_err": float(np.median(T[:, -1])),
            "median_curve_w": np.median(W, axis=0),
            "median_curve_theta": np.median(T, axis=0),
            "median_curve_UL": np.median(UL, axis=0),
            "median_curve_UL_gap": np.median(UL_gap[converged] if converged.any() else np.full(UL_gap.shape[1],np.nan), axis=0),
            "median_curve_LL": np.median(LL, axis=0),
            "median_curve_LL_gap": np.median(LL_gap, axis=0),
            "median_curve_LL_gap_conv": np.median(LL_gap[converged] if converged.any() else np.full(LL_gap.shape[1],np.nan), axis=0),
            "mean_curve_w": np.mean(W, axis=0),
            "mean_curve_theta": np.mean(T, axis=0),
            "mean_curve_UL_gap": np.mean(UL_gap, axis=0),
            "mean_curve_UL_gap_conv": np.mean(UL_gap[converged] if converged.any() else np.full(UL_gap.shape[1],np.nan), axis=0),
            "mean_curve_LL_gap": np.mean(LL_gap, axis=0),
            "mean_curve_LL_gap_conv": np.mean(LL_gap[converged] if converged.any() else np.full(LL_gap.shape[1],np.nan), axis=0),
            "success_curve": ((UL_gap <= tol) & (LL_gap <= tol)).sum(axis=0) / total,
        }
    return summary


def summary_table(summary, order=None):
    """Plain-text table; also handy for pasting into notes."""
    order = order or list(summary)
    lines = [f"{'method':<8} {'solved':>8} {'wrong basin':>12} {'diverged':>9} "
             f"{'median |w-w*|':>15} {'geomean':>12} {'median theta err':>17}",
             "-" * 86]
    for m in order:
        s = summary[m]
        lines.append(f"{m:<8} {s['frac_solved']*100:>7.1f}% "
                     f"{s['frac_wrong_basin']*100:>11.1f}% {s['frac_diverged']*100:>8.1f}% "
                     f"{s['median_final_w_err']:>15.3e} {s['geomean_final_w_err']:>12.3e} "
                     f"{s['median_final_theta_err']:>17.3e}")
    lines.append("")
    lines.append("fractions are over ALL starts; medians/geomeans over completed runs only")
    return "\n".join(lines)
