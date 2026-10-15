#!/usr/bin/env python3
"""
Coordinate grid search at κ=1.

Metric: first iteration t where ‖w_t−w*‖ < TOL AND |θ_t−θ*| < TOL simultaneously.

Non-PABLO: 5 values per parameter  (logspace ×0.1 … ×10, ref at ×1)
PABLO    : 9 values per parameter  (logspace ×0.1 … ×10, ref at ×1)

Usage: python run_coord_iters.py
"""

import os, sys, json, time
import numpy as np

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in [HERE, PARENT]:
    if _p not in sys.path: sys.path.insert(0, _p)

import ridge_lib
from ridge_lib import run_single, make_data, optimal_solution

KAPPA       = 1.0
COV         = np.diag([KAPPA, 1.0])
DATA_KWARGS = dict(w_true=[1.0, 1.0], n_train=200, n_val=200, noise_std=0.1)
DATA_SEED   = 0
W0          = 2.0
THETA0      = 3.0
N_ITERS     = 2000
TOL         = 1e-2

# 5 values (non-PABLO): ×0.10, ×0.316, ×1.0[ref], ×3.16, ×10
MULTS_5 = np.logspace(-1, 1, 5).tolist()   # [0.1, 0.316, 1.0, 3.162, 10.0]

# 9 values (PABLO): ×0.10 … ×10 with ref at ×1.0
MULTS_9 = np.logspace(-1, 1, 9).tolist()   # [0.1, 0.178, 0.316, 0.562, 1.0, 1.78, 3.16, 5.62, 10.0]

METHODS_RUN = ["PABLO", "BOME", "TSP", "STABLE", "AAL", "PBGD-Free"]

# ── Reference configs (user-provided) ────────────────────────────────────────
REF = {
    "PABLO": dict(lr_w=0.05, lr_theta=0.05, alpha=0.999, eps=1e-4, rho=0.1,
                  lr_lambda=0.05, tau=0.01, y_rate=0.3,
                  grad_clip=10.0, lambda_bound=10.0, total_steps=None),
    "BOME":  dict(lr=0.3, opt="adam", inner_steps=10,
                  inner_lr=0.001, eta=0.1, barrier="grad_norm"),
    "TSP":   dict(gamma=0.2, delta=0.0, p=0.01, dual_tau=0.05,
                  mu=0.5, theta=0.5, eta=0.01, step=0.04,
                  omega=0.2, lambda_init=1.0),
    "STABLE":dict(lr_w=0.02, lr_theta=0.075, tau=0.01,
                  mu_g=0.01, C_gxy=10.0),
    "AAL":   dict(L_f=71.0, L_grad_g=71.0, L_H=60.0,
                  gamma=0.03593814, rho0=0.0, alpha=3.594e-5,
                  eps=1e-5, tau=0.01),
    "PBGD-Free": dict(gamma=0.5, eta=0.0243, eta_gamma=0.012, K=1),
}

# Parameters to sweep per method (skip fixed strings, booleans, and zero-locked)
SWEEP_PARAMS = {
    "PABLO":     ["lr_w", "lr_theta", "alpha", "eps", "rho", "lr_lambda",
                  "tau", "y_rate", "grad_clip", "lambda_bound"],
    "BOME":      ["lr", "inner_lr", "eta", "inner_steps"],
    "TSP":       ["step", "eta", "dual_tau", "gamma", "omega",
                  "lambda_init", "p", "mu", "theta"],
    "STABLE":    ["lr_w", "lr_theta", "tau", "mu_g", "C_gxy"],
    "AAL":       ["alpha", "tau", "L_f", "L_grad_g", "L_H", "gamma"],
    "PBGD-Free": ["gamma", "eta", "eta_gamma", "K"],
}

# Hard bounds for parameters that can't be freely scaled
CLIP_MAX = {"alpha": 0.9999, "mu": 0.999, "theta": 0.999}
CLIP_MIN = {"alpha": 0.01,   "mu": 0.01,  "theta": 0.01, "p": 1e-4}
INT_PARAMS = {"inner_steps", "K"}


def make_sweep_values(param, base_val, n_points):
    """Return n_points log-uniformly spaced values in [base/10, base*10]."""
    mults = np.logspace(-1, 1, n_points).tolist()
    vals = []
    for m in mults:
        v = base_val * m
        if param in CLIP_MAX:  v = min(v, CLIP_MAX[param])
        if param in CLIP_MIN:  v = max(v, CLIP_MIN[param])
        if param in INT_PARAMS: v = max(1, int(round(v)))
        vals.append(v)
    # Deduplicate (important for int params at extremes)
    seen, out = set(), []
    for v in vals:
        key = int(v) if param in INT_PARAMS else round(v, 12)
        if key not in seen:
            seen.add(key); out.append(v)
    return out


def iters_to_joint_converge(w_err, t_err):
    """First t where both w_err[t] < TOL and t_err[t] < TOL."""
    both = (w_err < TOL) & (t_err < TOL)
    idx  = np.where(both)[0]
    return int(idx[0] + 1) if len(idx) else np.nan


def run_cfg(method, cfg):
    r = run_single(method, cfg, COV, w0=W0, theta0=THETA0,
                   n_iters=N_ITERS, data_seed=DATA_SEED,
                   data_kwargs=DATA_KWARGS)
    if r["diverged"] or len(r["w"]) == 0:
        return None
    w_err = np.linalg.norm(r["w"] - r["w_opt"][None, :], axis=-1)
    t_err = np.abs(r["theta"] - r["theta_opt"])
    return w_err, t_err


def fmt_iters(v):
    return "    —" if np.isnan(v) else f"{int(v):>5d}"


def main():
    data = make_data(COV, seed=DATA_SEED, **DATA_KWARGS)
    w_opt, theta_opt = optimal_solution(data)
    print(f"κ={KAPPA}  w*={w_opt.round(4)}  θ*={theta_opt:.5f}")
    print(f"Start: w0={W0}·[1,1]  θ0={THETA0}  N_ITERS={N_ITERS}  TOL={TOL}")
    print(f"Metric: first t where ‖w_t−w*‖ < {TOL} AND |θ_t−θ*| < {TOL}\n")

    best_cfgs = {}   # collect per-method best config

    t_global = time.time()
    for method in METHODS_RUN:
        base = dict(REF[method])
        is_pablo = (method == "PABLO")
        n_vals   = 9 if is_pablo else 5
        params   = SWEEP_PARAMS[method]

        print(f"\n{'='*65}")
        print(f"  {method}   ({'9 vals/param' if is_pablo else '5 vals/param'})")
        print(f"{'='*65}")

        # Baseline run
        res0 = run_cfg(method, base)
        if res0 is None:
            base_iters = np.nan
            print(f"  [reference: DIVERGED]")
        else:
            base_iters = iters_to_joint_converge(*res0)
            print(f"  Reference config → joint iters = {fmt_iters(base_iters).strip()}")

        # Best tracker per-param
        best_cfg  = dict(base)
        best_iters_overall = base_iters

        print(f"\n  {'param':<14} {'value':>10}  {'joint iters':>12}")
        print(f"  {'-'*40}")

        for param in params:
            base_val = base[param]
            if base_val is None or isinstance(base_val, bool) or isinstance(base_val, str):
                continue
            base_val = float(base_val) if param not in INT_PARAMS else int(base_val)

            sweep_vals = make_sweep_values(param, float(base_val), n_vals)
            param_best_iters = np.nan
            param_best_val   = base_val

            for val in sweep_vals:
                cfg = dict(base)
                cfg[param] = int(val) if param in INT_PARAMS else float(val)

                res = run_cfg(method, cfg)
                iters = iters_to_joint_converge(*res) if res is not None else np.nan
                diverged = res is None

                ref_mark = " [ref]" if abs(float(val) / float(base_val) - 1.0) < 0.02 else ""
                div_mark = " div"   if diverged else ""
                print(f"  {param:<14} {float(val):>10.4g}  "
                      f"{fmt_iters(iters)}{ref_mark}{div_mark}")

                # Track param-level best
                if not np.isnan(iters) and (np.isnan(param_best_iters) or iters < param_best_iters):
                    param_best_iters = iters
                    param_best_val   = val

            # Update best config with this param's best value
            if not np.isnan(param_best_iters) and (np.isnan(best_iters_overall)
                                                    or param_best_iters < best_iters_overall):
                best_cfg[param]   = (int(param_best_val) if param in INT_PARAMS
                                     else float(param_best_val))
                best_iters_overall = param_best_iters
                print(f"  ↳ best for {param}: {float(param_best_val):.4g} "
                      f"(iters={int(param_best_iters)})")

        best_cfgs[method] = (best_iters_overall, best_cfg)
        print(f"\n  Best from sweep: joint iters = {fmt_iters(best_iters_overall).strip()}")

    print(f"\n\n{'='*65}")
    print(f"  SUMMARY — best joint iters at κ={KAPPA}")
    print(f"{'='*65}")
    print(f"  {'method':<12}  {'ref iters':>10}  {'best iters':>10}")
    print(f"  {'-'*38}")
    for method in METHODS_RUN:
        ref_res = run_cfg(method, REF[method])
        ref_it  = iters_to_joint_converge(*ref_res) if ref_res else np.nan
        best_it = best_cfgs[method][0]
        print(f"  {method:<12}  {fmt_iters(ref_it).strip():>10}  "
              f"{fmt_iters(best_it).strip():>10}")

    print(f"\n\n  Best configs found:")
    for method in METHODS_RUN:
        _, cfg = best_cfgs[method]
        print(f"\n  {method!r}: {repr(cfg)},")

    print(f"\nTotal: {time.time()-t_global:.0f}s")


if __name__ == "__main__":
    main()
