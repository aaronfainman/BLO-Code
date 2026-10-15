#!/usr/bin/env python3
"""Quick hyperparameter sweep for BLO_optimiser_TSP (paper-faithful variant).

Phases:
  1. Vary each parameter individually (5 log-spaced values, param/10 → param×10)
  2. Nested grid: step × dual_tau × eta  (5 values each, same spacing)

Convergence: median F < 1e-2 AND median J < 1e-2 within N_ITERS steps.
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ridge_lib import run_sweep, summarise

# ── Baseline TSP parameters (paper-faithful, κ=1) ────────────────────────────
BASELINE = dict(
    step=0.44261068, eta=0.32315575, dual_tau=0.00137256,
    gamma=0.28618074, delta=0.0, p=0.05,
    mu=0.7, theta=1.0, omega=0.27606047, lambda_init=0.5,
)

# ── Sweep settings ───────────────────────────────────────────────────────────
COV              = np.diag([1.0, 1.0])    # κ = 1
W0_LO,    W0_HI         = -2.0,  2.0
THETA0_LO, THETA0_HI    = -2.0,  2.0
N_STARTS         = 2               # 2×2 = 4 initial conditions
N_ITERS          = 500
CONV_TOL         = 1e-2
DATA_SEED        = 0
DATA_KWARGS      = dict(w_true=np.array([1.0, 1.0]), n_train=200, n_val=200, noise_std=0.1)
N_JOBS           = max(1, (os.cpu_count() or 2) - 1)
N_VALS           = 5


def sweep_one(cfg):
    """Return (conv_iter | None, last_F, last_J) for one config."""
    sw = run_sweep(
        {"TSP": cfg}, cov=COV,
        w0_lo=W0_LO, w0_hi=W0_HI, w0_n=N_STARTS,
        theta0_lo=THETA0_LO, theta0_hi=THETA0_HI, theta0_n=N_STARTS,
        n_iters=N_ITERS, data_seed=DATA_SEED, data_kwargs=DATA_KWARGS,
        n_jobs=N_JOBS, progress=False,
    )
    sm = summarise(sw, tol=CONV_TOL)["TSP"]
    ul = sm["median_curve_UL"]
    ll = sm["median_curve_LL"]
    if ul is None or ll is None:
        return None, float("nan"), float("nan")
    hit = np.where((ul < CONV_TOL) & (ll < CONV_TOL))[0]
    return (int(hit[0] + 1) if len(hit) else None), float(ul[-1]), float(ll[-1])


if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: individual parameter sweeps
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 1  Individual parameter sweeps  (TSP paper-faithful)")
    print("=" * 72)
    print(f"Baseline: {BASELINE}")
    print(f"Settings: w0/theta0 ∈ [{W0_LO},{W0_HI}], {N_STARTS}×{N_STARTS} starts, "
          f"{N_ITERS} iters, conv_tol={CONV_TOL}\n")

    t0 = time.time()
    base_conv, base_F, base_J = sweep_one(BASELINE)
    print(f"BASELINE:  conv_iter={base_conv},  last_F={base_F:.4e},  last_J={base_J:.4e}"
          f"  ({time.time()-t0:.1f}s)\n")

    best_cfg = dict(BASELINE)

    for param, bval in BASELINE.items():
        if bval == 0.0:
            vals = np.array([0.0, 0.001, 0.01, 0.1, 0.5])
        else:
            vals = np.geomspace(bval / 10.0, bval * 10.0, N_VALS)

        print(f"── {param}  (baseline = {bval:.5g}) ────────────────────────────────────")
        print(f"  {'value':>10}  {'conv_iter':>10}  {'last_F':>12}  {'last_J':>12}")
        rows = []
        for v in vals:
            cfg  = {**BASELINE, param: float(v)}
            conv, last_F, last_J = sweep_one(cfg)
            print(f"  {v:>10.4g}  {str(conv) if conv else '---':>10}  "
                  f"{last_F:>12.4e}  {last_J:>12.4e}")
            rows.append((float(v), conv, last_F, last_J))

        conv_rows = [(v, c, f, j) for v, c, f, j in rows if c is not None]
        if conv_rows:
            best_v = min(conv_rows, key=lambda x: (x[1], x[2] + x[3]))[0]
        else:
            best_v = min(rows, key=lambda x: (x[2]+x[3]) if np.isfinite(x[2]+x[3]) else 1e30)[0]
        best_cfg[param] = best_v
        print(f"  → best: {best_v:.4g}\n")

    print("\n── Phase-1 best config ──────────────────────────────────────────────────")
    for k, v in best_cfg.items():
        flag = " *" if abs(v - BASELINE[k]) / (abs(BASELINE[k]) + 1e-12) > 0.05 else ""
        print(f"  {k:15s}: {v:.5g}  (baseline {BASELINE[k]:.5g}){flag}")
    t0 = time.time()
    p1_conv, p1_F, p1_J = sweep_one(best_cfg)
    print(f"\nPhase-1 best:  conv_iter={p1_conv},  last_F={p1_F:.4e},  last_J={p1_J:.4e}"
          f"  ({time.time()-t0:.1f}s)\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: nested step × dual_tau × eta grid
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 2  Nested step × dual_tau × eta grid")
    print("=" * 72)

    step_vals     = np.geomspace(BASELINE["step"]    / 10, BASELINE["step"]    * 10, N_VALS)
    dual_tau_vals = np.geomspace(BASELINE["dual_tau"] / 10, BASELINE["dual_tau"] * 10, N_VALS)
    eta_vals      = np.geomspace(BASELINE["eta"]      / 10, BASELINE["eta"]      * 10, N_VALS)

    print(f"step     : {np.round(step_vals,    6).tolist()}")
    print(f"dual_tau : {np.round(dual_tau_vals,6).tolist()}")
    print(f"eta      : {np.round(eta_vals,     6).tolist()}")
    print(f"Total: {len(step_vals)*len(dual_tau_vals)*len(eta_vals)} configs\n")

    print(f"{'step':>10}  {'dual_tau':>10}  {'eta':>10}  {'conv':>8}  {'last_F':>12}  {'last_J':>12}")
    print("-" * 76)

    best_nested = None
    best_key    = (float("inf"), float("inf"))

    for s in step_vals:
        for dt in dual_tau_vals:
            for et in eta_vals:
                cfg  = {**best_cfg, "step": float(s), "dual_tau": float(dt), "eta": float(et)}
                conv, last_F, last_J = sweep_one(cfg)
                print(f"{s:>10.4g}  {dt:>10.4g}  {et:>10.4g}  "
                      f"{str(conv) if conv else '---':>8}  {last_F:>12.4e}  {last_J:>12.4e}")
                score = (conv if conv is not None else N_ITERS + 1,
                         last_F + last_J if np.isfinite(last_F + last_J) else 1e30)
                if score < best_key:
                    best_nested, best_key = dict(cfg), score

    print("\n── Best nested config ──────────────────────────────────────────────────")
    for k, v in best_nested.items():
        flag = " *" if abs(v - BASELINE[k]) / (abs(BASELINE[k]) + 1e-12) > 0.05 else ""
        print(f"  {k:15s}: {v:.5g}  (baseline {BASELINE[k]:.5g}){flag}")
    t0 = time.time()
    fin_conv, fin_F, fin_J = sweep_one(best_nested)
    print(f"\nFinal best:  conv_iter={fin_conv},  last_F={fin_F:.4e},  last_J={fin_J:.4e}"
          f"  ({time.time()-t0:.1f}s)")
