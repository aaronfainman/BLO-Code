#!/usr/bin/env python3
"""
Coordinate search for κ=1: sweep each hyperparameter independently.

For each (method, param, value), reports the first iteration where the
trajectory crosses below TOL=0.1 for:
    ‖w_t − w*‖,   |θ_t − θ*|,   UL loss F,   LL loss J

Reference config: current κ=1 best (from tuning_results.json).
Sweep: 5 multiplicative values [×0.1, ×0.3, ×1 (base), ×3, ×10].
No step sizes are set to zero.
"""

import os, sys, json, time
import numpy as np

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in [HERE, PARENT]:
    if _p not in sys.path: sys.path.insert(0, _p)

import ridge_lib
from ridge_lib import run_single, make_data, optimal_solution

# ── problem ───────────────────────────────────────────────────────────────────
KAPPA       = 1.0
COV         = np.diag([KAPPA, 1.0])
DATA_KWARGS = dict(w_true=[1.0, 1.0], n_train=200, n_val=200, noise_std=0.1)
DATA_SEED   = 0

# Starting point far enough from the optimum for all metrics to start above TOL
W0     = 2.0   # w_init = [2, 2]
THETA0 = 3.0   # theta_init = 3.0

N_ITERS = 2000
TOL     = 0.1

SWEEP_MULTS = [0.1, 1/3, 1.0, 3.0, 10.0]   # ×0.1, ×0.3, ×1, ×3, ×10
METHODS_RUN = ["PABLO", "BOME", "TSP", "STABLE", "AAL", "PBGD-Free"]

# Parameters to sweep per method (excluding fixed strings, bools, None)
SWEEP_PARAMS = {
    "PABLO":     ["lr", "rho", "tau", "y_rate", "lr_lambda", "lambda_bound"],
    "BOME":      ["lr", "inner_lr", "eta", "inner_steps"],
    "TSP":       ["step", "eta", "dual_tau", "gamma", "omega"],
    "STABLE":    ["lr_w", "lr_theta", "tau", "mu_g", "C_gxy"],
    "AAL":       ["alpha", "tau", "L_f", "L_grad_g", "L_H", "gamma"],
    "PBGD-Free": ["gamma", "eta", "eta_gamma", "K"],
}

# Fallback base for params that are 0 (never sweep through 0)
ZERO_FALLBACK = {
    "lr_lambda": 0.01,
    "rho0":      0.01,
    "delta":     0.01,
}

# Params that must be integers
INT_PARAMS = {"inner_steps", "K"}

# Params where ×10 would exceed valid range (clip to safe max)
CLIP_MAX = {"alpha": 0.9999}


# =============================================================================

def first_crossing(arr, tol):
    idx = np.where(np.asarray(arr) < tol)[0]
    return int(idx[0] + 1) if len(idx) else np.nan


def fmt(v):
    return "    —" if np.isnan(v) else f"{int(v):>5d}"


def run_cfg(method, cfg):
    r = run_single(method, cfg, COV, w0=W0, theta0=THETA0,
                   n_iters=N_ITERS, data_seed=DATA_SEED, data_kwargs=DATA_KWARGS)
    if r["diverged"] or len(r["w"]) == 0:
        return None
    w_err = np.linalg.norm(r["w"] - r["w_opt"][None, :], axis=-1)
    t_err = np.abs(r["theta"] - r["theta_opt"])
    return dict(w=w_err, th=t_err, UL=r["ul_loss"], LL=r["ll_loss"])


def sweep_param(method, base_cfg, param):
    raw_base = base_cfg.get(param, 0)
    if raw_base == 0 or raw_base is None:
        base_val = ZERO_FALLBACK.get(param, 0.01)
    else:
        base_val = float(raw_base)

    sweep_vals = []
    for m in SWEEP_MULTS:
        v = base_val * m
        if param in CLIP_MAX:
            v = min(v, CLIP_MAX[param])
        if v <= 0:
            v = 1e-6
        if param in INT_PARAMS:
            v = max(1, int(round(v)))
        sweep_vals.append(v)

    # Remove duplicates while preserving order
    seen = set(); uniq_vals = []
    for v in sweep_vals:
        key = v if param not in INT_PARAMS else int(v)
        if key not in seen:
            seen.add(key); uniq_vals.append(v)
    sweep_vals = uniq_vals

    results = []
    for val in sweep_vals:
        cfg = dict(base_cfg)
        cfg[param] = int(val) if param in INT_PARAMS else float(val)
        results.append((val, run_cfg(method, cfg)))

    return base_val, results


def print_row(val, base_val, res, param, int_param):
    base_marker = " [*]" if (int_param and int(val) == int(base_val)) \
                  or (not int_param and abs(val - base_val) / (abs(base_val) + 1e-12) < 0.02) \
                  else "    "
    val_str = f"{int(val):>6d}" if int_param else f"{val:>9.4g}"
    if res is None:
        row = "   div   div   div   div"
    else:
        row = (f"{fmt(first_crossing(res['w'],  TOL))}"
               f"{fmt(first_crossing(res['th'], TOL))}"
               f"{fmt(first_crossing(res['UL'], TOL))}"
               f"{fmt(first_crossing(res['LL'], TOL))}")
    print(f"    {val_str}  {row}{base_marker}")


def main():
    # Load best κ=1 configs
    with open(os.path.join(HERE, "tuning_results.json")) as f:
        data = json.load(f)
    base_configs = data["hparams"].get("1.0", {})

    print(f"Coordinate sweep  κ={KAPPA}")
    print(f"Start: w0={W0}·[1,1],  θ0={THETA0}")
    print(f"Sweep: ×0.1  ×1/3  ×1(base)  ×3  ×10   TOL={TOL}   N_ITERS={N_ITERS}\n")
    print(f"Columns: iters until first crossing below {TOL} for")
    print(f"         ‖w_t−w*‖   |θ_t−θ*|   UL loss   LL loss\n")

    t_total = time.time()

    for method in METHODS_RUN:
        cfg = dict(base_configs.get(method, {}))
        if not cfg:
            print(f"\n{'='*60}\n  {method} — no config found\n{'='*60}")
            continue

        print(f"\n{'='*60}")
        print(f"  {method}")
        print(f"{'='*60}")
        print(f"  {'param':<14}  {'value':>9}  {'w_err':>5}  {'θ_err':>5}  {'UL':>5}  {'LL':>5}")
        print(f"  {'-'*55}")

        for param in SWEEP_PARAMS[method]:
            if param not in cfg and param not in ZERO_FALLBACK:
                continue

            t0 = time.time()
            base_val, sweep_results = sweep_param(method, cfg, param)
            elapsed = time.time() - t0

            # Print header for this parameter
            int_p = param in INT_PARAMS
            for val, res in sweep_results:
                base_marker = (
                    " [*]" if (int_p and int(val) == int(base_val)) or
                               (not int_p and abs(val - base_val) / (abs(base_val) + 1e-12) < 0.02)
                    else "    "
                )
                val_str = f"{int(val):>9d}" if int_p else f"{val:>9.4g}"
                if res is None:
                    row = "    div    div    div    div"
                else:
                    row = (f"  {fmt(first_crossing(res['w'],  TOL))}"
                           f"  {fmt(first_crossing(res['th'], TOL))}"
                           f"  {fmt(first_crossing(res['UL'], TOL))}"
                           f"  {fmt(first_crossing(res['LL'], TOL))}")
                print(f"  {param:<14}  {val_str}  {row}{base_marker}")

        # Baseline with unmodified config
        print(f"\n  [full baseline]")
        res0 = run_cfg(method, cfg)
        if res0 is None:
            print(f"  diverged")
        else:
            print(f"  w_err < {TOL}: iter {int(first_crossing(res0['w'],  TOL)) if not np.isnan(first_crossing(res0['w'],  TOL)) else '—'}")
            print(f"  θ_err < {TOL}: iter {int(first_crossing(res0['th'], TOL)) if not np.isnan(first_crossing(res0['th'], TOL)) else '—'}")
            print(f"  UL    < {TOL}: iter {int(first_crossing(res0['UL'], TOL)) if not np.isnan(first_crossing(res0['UL'], TOL)) else '—'}")
            print(f"  LL    < {TOL}: iter {int(first_crossing(res0['LL'], TOL)) if not np.isnan(first_crossing(res0['LL'], TOL)) else '—'}")

    print(f"\nTotal: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
