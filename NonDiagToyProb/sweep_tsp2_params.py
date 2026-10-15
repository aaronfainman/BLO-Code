#!/usr/bin/env python3
"""Hyperparameter sweep for BLO_optimiser_TSP (practical variant) on the ridge BLO problem.

Phase 1: vary each parameter individually (5 log-spaced values from param/10 to param*10)
Phase 2: nested lr_y × lr_x × dual_lr grid (5 values each, same log-spacing)

Convergence criterion: median F < 1e-2 AND median J < 1e-2
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ridge_lib import run_sweep, summarise

# ── Baseline parameters ──────────────────────────────────────────────────────
BASELINE = dict(
    gamma=1.1, delta=0.01, p=1.0, eta_z=0.01,
    lr_y=0.02, lr_x=0.05, dual_lr=0.06,
    constraint_ema=0.5, lambda_ema=0.05, anchor_ema=0.1, lambda_init=1.0,
)

# ── Sweep settings ───────────────────────────────────────────────────────────
COV              = np.diag([10.0, 1.0])   # κ = 10
W0_LO,    W0_HI         = -2.0,  2.0
THETA0_LO, THETA0_HI    = -2.0,  2.0
N_STARTS         = 2               # 2×2 = 4 initial conditions
N_ITERS          = 500
CONV_TOL         = 1e-2
DATA_SEED        = 0
DATA_KWARGS      = dict(w_true=np.array([1.0, 1.0]), n_train=200, n_val=200, noise_std=0.1)
N_JOBS           = max(1, (os.cpu_count() or 2) - 1)
N_VALS           = 5               # values per parameter

EMA_PARAMS       = {"constraint_ema", "lambda_ema", "anchor_ema"}   # must stay ≤ 0.99


# ─────────────────────────────────────────────────────────────────────────────

def sweep_one(cfg):
    """Return (conv_iter | None, last_F, last_J) for one config."""
    sw = run_sweep(
        {"TSP2": cfg}, cov=COV,
        w0_lo=W0_LO, w0_hi=W0_HI, w0_n=N_STARTS,
        theta0_lo=THETA0_LO, theta0_hi=THETA0_HI, theta0_n=N_STARTS,
        n_iters=N_ITERS, data_seed=DATA_SEED, data_kwargs=DATA_KWARGS,
        n_jobs=N_JOBS, progress=False,
    )
    sm = summarise(sw, tol=CONV_TOL)["TSP2"]

    ul = sm["median_curve_UL"]
    ll = sm["median_curve_LL"]
    if ul is None or ll is None:
        return None, float("nan"), float("nan")

    hit = np.where((ul < CONV_TOL) & (ll < CONV_TOL))[0]
    conv = int(hit[0] + 1) if len(hit) else None
    return conv, float(ul[-1]), float(ll[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Individual parameter sweeps
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("PHASE 1  Individual parameter sweeps  (TSP2 practical variant)")
print("=" * 74)
print(f"Baseline: {BASELINE}")
print(f"Settings: w0/theta0 ∈ [{W0_LO},{W0_HI}], {N_STARTS}×{N_STARTS} starts, "
      f"{N_ITERS} iters, conv_tol={CONV_TOL}\n")

t0 = time.time()
base_conv, base_F, base_J = sweep_one(BASELINE)
print(f"BASELINE run:  conv_iter={base_conv},  last_F={base_F:.4e},  last_J={base_J:.4e}"
      f"  ({time.time()-t0:.1f}s)\n")

results_p1 = {}
best_cfg   = dict(BASELINE)

for param, bval in BASELINE.items():
    lo = bval / 10.0 if bval != 0 else 1e-4
    hi = bval * 10.0 if bval != 0 else 1.0
    if param in EMA_PARAMS:
        hi = min(hi, 0.99)
    vals = np.geomspace(lo, hi, N_VALS)

    print(f"── {param}  (baseline = {bval:.4g}) ──────────────────────────────────────")
    print(f"  {'value':>10}  {'conv_iter':>10}  {'last_F':>12}  {'last_J':>12}")
    rows = []
    for v in vals:
        cfg = {**BASELINE, param: float(v)}
        conv, last_F, last_J = sweep_one(cfg)
        conv_str = str(conv) if conv is not None else "---"
        print(f"  {v:>10.4g}  {conv_str:>10}  {last_F:>12.4e}  {last_J:>12.4e}")
        rows.append((float(v), conv, last_F, last_J))
    results_p1[param] = rows

    conv_rows = [(v, c, f, j) for v, c, f, j in rows if c is not None]
    if conv_rows:
        best_v = min(conv_rows, key=lambda x: (x[1], x[2] + x[3]))[0]
    else:
        best_v = min(rows, key=lambda x: (x[2] + x[3]) if np.isfinite(x[2] + x[3]) else 1e30)[0]
    best_cfg[param] = best_v
    print(f"  → best value for {param}: {best_v:.4g}\n")


print("\n── Best config from Phase 1 ──────────────────────────────────────────────")
for k, v in best_cfg.items():
    flag = " *" if abs(v - BASELINE[k]) / (abs(BASELINE[k]) + 1e-12) > 0.05 else ""
    print(f"  {k:20s}: {v:.4g}  (baseline {BASELINE[k]:.4g}){flag}")

t0 = time.time()
p1_conv, p1_F, p1_J = sweep_one(best_cfg)
print(f"\nPhase-1 best config:  conv_iter={p1_conv},  last_F={p1_F:.4e},  last_J={p1_J:.4e}"
      f"  ({time.time()-t0:.1f}s)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Nested lr_y × lr_x × dual_lr sweep
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("PHASE 2  Nested lr_y × lr_x × dual_lr grid  (baseline values)")
print("=" * 74)

lr_y_vals    = np.geomspace(BASELINE["lr_y"]    / 10, BASELINE["lr_y"]    * 10, N_VALS)
lr_x_vals    = np.geomspace(BASELINE["lr_x"]    / 10, BASELINE["lr_x"]    * 10, N_VALS)
dual_lr_vals = np.geomspace(BASELINE["dual_lr"] / 10, BASELINE["dual_lr"] * 10, N_VALS)

print(f"lr_y    : {np.round(lr_y_vals, 5).tolist()}")
print(f"lr_x    : {np.round(lr_x_vals, 5).tolist()}")
print(f"dual_lr : {np.round(dual_lr_vals, 5).tolist()}")
n_nested = len(lr_y_vals) * len(lr_x_vals) * len(dual_lr_vals)
print(f"Total configs: {n_nested}\n")

print(f"{'lr_y':>9}  {'lr_x':>9}  {'dual_lr':>9}  {'conv_iter':>10}  {'last_F':>12}  {'last_J':>12}")
print("-" * 78)

best_nested     = None
best_nested_key = (float("inf"), float("inf"))

for ly in lr_y_vals:
    for lx in lr_x_vals:
        for dl in dual_lr_vals:
            cfg  = {**best_cfg, "lr_y": float(ly), "lr_x": float(lx), "dual_lr": float(dl)}
            conv, last_F, last_J = sweep_one(cfg)
            conv_str = str(conv) if conv is not None else "---"
            print(f"{ly:>9.4g}  {lx:>9.4g}  {dl:>9.4g}  {conv_str:>10}"
                  f"  {last_F:>12.4e}  {last_J:>12.4e}")

            score = (conv if conv is not None else N_ITERS + 1,
                     last_F + last_J if np.isfinite(last_F + last_J) else 1e30)
            if score < best_nested_key:
                best_nested     = dict(cfg)
                best_nested_key = score


print("\n── Best nested config ──────────────────────────────────────────────────")
for k, v in best_nested.items():
    flag = " *" if abs(v - BASELINE[k]) / (abs(BASELINE[k]) + 1e-12) > 0.05 else ""
    print(f"  {k:20s}: {v:.4g}  (baseline {BASELINE[k]:.4g}){flag}")

t0 = time.time()
fin_conv, fin_F, fin_J = sweep_one(best_nested)
print(f"\nFinal best config:  conv_iter={fin_conv},  last_F={fin_F:.4e},  last_J={fin_J:.4e}"
      f"  ({time.time()-t0:.1f}s)")
