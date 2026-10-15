"""Conditioning sweep script — mirrors Exp_ToyProblem_CondSweep.ipynb.

IMPORTANT: wrap all execution in  if __name__ == "__main__":  so that
multiprocessing's spawn check does not re-execute the sweep when it
imports this module in a worker process.
"""
import os, sys, json, csv, time
import numpy as np

sys.path.insert(0, "experiments")
import toy_lib
from toy_lib import run_sweep, summarise, summary_table, theta_error, W_STAR


# =============================================================================
# Settings
# =============================================================================
C_VALUES = [0.1, 1.0, 10.0]
METHODS  = ["PABLO", "BOME", "TSP", "STABLE", "AAL", "PBGD-Free"]

HPARAMS_BASE = {
    "PABLO": dict(
        lr=0.003, alpha=0.9, eps=1e-3, rho=0.3,
        lr_lambda=0.01, tau=10.0, y_rate=0.5,
        grad_clip=10.0, lambda_bound=100.0, total_steps=None,
    ),
    "BOME": dict(
        lr=0.007, opt="adam", inner_steps=10, inner_lr=0.01,
        eta=0.1, barrier="grad_norm",
    ),
    "TSP": dict(
        gamma=0.5, delta=0.0, p=0.01, dual_tau=0.5,
        mu=0.5, theta=1.0, eta=0.2, step=0.1,
        omega=0.05, lambda_init=1.0,
    ),
    "STABLE": dict(
        lr_w=0.05, lr_theta=0.5, tau=1.0, mu_g=0.001, C_gxy=10.0,
    ),
    "AAL": dict(
        L_f=71.0, L_grad_g=71.0, L_H=60.0,
        gamma=1.0, rho0=0.0, alpha=1e-3,
        eps=1e-5, tau=1e-3, tau_decay=False,
    ),
    "PBGD-Free": dict(
        gamma=10.0, eta=1e-2, eta_gamma=1e-2, K=1,
    ),
}

# Step-size keys: only floats in this list are scaled during the lr search
STEP_KEYS = {
    "PABLO":     ["lr", "y_rate"],
    "BOME":      ["lr", "inner_lr"],
    "TSP":       ["step", "eta", "dual_tau"],
    "STABLE":    ["lr_w", "lr_theta"],
    "AAL":       ["alpha", "tau"],
    "PBGD-Free": ["eta", "eta_gamma"],
}

# PABLO-specific secondary search grids (applied AFTER the lr-scale search)
PABLO_LR_LAMBDA_GRID = [0.001, 0.005, 0.01, 0.05, 0.1]  # multiplier step (no zero)
PABLO_TAU_GRID       = [0.02, 0.2, 2.0, 10.0, 30.0]   # proximal weight
PABLO_RHO_GRID       = [0.03, 0.1, 0.3, 1.0, 3.0]     # AL penalty weight

LR_SCALES  = [0.1, 0.3, 1.0, 3.0, 10.0]

N_ITERS        = 2000
TUNE_N         = 4       # 4×4 = 16 starts
FULL_N         = 10      # 10×10 = 100 starts
GRID_LO, GRID_HI = -1.5, 1.5
SEED           = 0
LOSS_TOL       = 1e-9
N_JOBS         = 1       # forced serial (avoids spawn re-importing __main__)


def _median_loss(sweep_out, method):
    """Median F_T + J_T for runs that completed; inf if all diverged."""
    sm = summarise(sweep_out)[method]
    if sm.get("median_curve_UL") is None:
        return np.inf
    ul = float(sm["median_curve_UL"][-1])
    ll = float(sm["median_curve_LL"][-1])
    return (ul if np.isfinite(ul) else 1e30) + (ll if np.isfinite(ll) else 1e30)


def _run_tune(cfg, method, c):
    """Single tuning run on the 4×4 grid; returns median final loss."""
    try:
        sw = run_sweep({method: cfg}, c=c, grid_lo=GRID_LO, grid_hi=GRID_HI,
                       grid_n=TUNE_N, n_iters=N_ITERS, seed=SEED,
                       n_jobs=N_JOBS, progress=False)
        return _median_loss(sw, method), sw
    except Exception as exc:
        print(f"    ERROR: {exc}")
        return np.inf, None


def tune_pablo(c, base_cfg):
    """Four-phase search for PABLO at a given c.

    Phase 1 — lr scale  : scale lr and y_rate jointly.
    Phase 2 — lr_lambda : search multiplier step size (lr_lambda).
    Phase 3 — tau       : search proximal weight.
    Phase 4 — rho       : search augmented-Lagrangian penalty.

    Each phase fixes all parameters found in earlier phases.
    Returns the best cfg dict.
    """
    best_cfg  = base_cfg.copy()
    best_loss = np.inf

    # Phase 1: lr scale
    print("  PABLO  phase-1: lr scale")
    for s in LR_SCALES:
        cfg = {k: (v * s if k in STEP_KEYS["PABLO"] and isinstance(v, float) else v)
               for k, v in base_cfg.items()}
        loss, _ = _run_tune(cfg, "PABLO", c)
        tag = f"{loss:.3e}" if np.isfinite(loss) else "diverged"
        print(f"    lr_scale={s:.1f}x  loss={tag}")
        if loss < best_loss:
            best_loss, best_cfg = loss, cfg
    print(f"  -> phase-1 best: lr={best_cfg['lr']:.4g}  loss={best_loss:.3e}\n")

    # Phase 2: lr_lambda
    print("  PABLO  phase-2: lr_lambda")
    for lrl in PABLO_LR_LAMBDA_GRID:
        cfg = {**best_cfg, "lr_lambda": lrl}
        loss, _ = _run_tune(cfg, "PABLO", c)
        tag = f"{loss:.3e}" if np.isfinite(loss) else "diverged"
        print(f"    lr_lambda={lrl:.3g}  loss={tag}")
        if loss < best_loss:
            best_loss, best_cfg = loss, cfg
    print(f"  -> phase-2 best: lr_lambda={best_cfg['lr_lambda']:.3g}  loss={best_loss:.3e}\n")

    # Phase 3: tau
    print("  PABLO  phase-3: tau")
    for tau in PABLO_TAU_GRID:
        cfg = {**best_cfg, "tau": tau}
        loss, _ = _run_tune(cfg, "PABLO", c)
        tag = f"{loss:.3e}" if np.isfinite(loss) else "diverged"
        print(f"    tau={tau:.3g}  loss={tag}")
        if loss < best_loss:
            best_loss, best_cfg = loss, cfg
    print(f"  -> phase-3 best: tau={best_cfg['tau']:.3g}  loss={best_loss:.3e}\n")

    # Phase 4: rho
    print("  PABLO  phase-4: rho")
    for rho in PABLO_RHO_GRID:
        cfg = {**best_cfg, "rho": rho}
        loss, _ = _run_tune(cfg, "PABLO", c)
        tag = f"{loss:.3e}" if np.isfinite(loss) else "diverged"
        print(f"    rho={rho:.3g}  loss={tag}")
        if loss < best_loss:
            best_loss, best_cfg = loss, cfg
    print(f"  -> phase-4 best: rho={best_cfg['rho']:.3g}  loss={best_loss:.3e}\n")

    return best_cfg


if __name__ == "__main__":
    print("methods:", toy_lib.METHODS)
    print("cpu count:", os.cpu_count())
    n_pablo_runs = (len(LR_SCALES) + len(PABLO_LR_LAMBDA_GRID)
                    + len(PABLO_TAU_GRID) + len(PABLO_RHO_GRID)) * TUNE_N**2
    # (PABLO: lr-scale, lr_lambda, tau, rho — all non-zero grids)
    n_other_runs = (len(METHODS) - 1) * len(LR_SCALES) * TUNE_N**2
    print(f"Tuning per c:  PABLO {n_pablo_runs} runs  |  others {n_other_runs} runs")
    print(f"Full sweep:    {len(METHODS) * FULL_N**2} runs")
    print(f"c values: {C_VALUES}\n")

    # =========================================================================
    # Tuning phase
    # =========================================================================
    all_hparams = {}

    for c in C_VALUES:
        print(f"\n{'='*65}")
        print(f"  TUNING  c = {c:g}")
        print(f"{'='*65}")
        tuned = {}

        # ---- PABLO: 4-phase search ------------------------------------------
        tuned["PABLO"] = tune_pablo(c, HPARAMS_BASE["PABLO"])

        # ---- All other methods: lr-scale search -----------------------------
        for m in [m for m in METHODS if m != "PABLO"]:
            best_loss, best_scale, best_cfg = np.inf, 1.0, HPARAMS_BASE[m].copy()
            for s in LR_SCALES:
                cfg = {k: (v * s if k in STEP_KEYS[m] and isinstance(v, float) else v)
                       for k, v in HPARAMS_BASE[m].items()}
                loss, _ = _run_tune(cfg, m, c)
                tag = f"{loss:.3e}" if np.isfinite(loss) else "diverged"
                print(f"    {m:<12}  scale={s:.1f}x  loss={tag}")
                if loss < best_loss:
                    best_loss, best_scale, best_cfg = loss, s, cfg
            tuned[m] = best_cfg
            print(f"  -> {m}: best scale={best_scale:.1f}x  (loss={best_loss:.3e})\n")

        all_hparams[c] = tuned

    print("\nTuning complete.\n")

    # =========================================================================
    # Full sweep
    # =========================================================================
    all_sweep   = {}
    all_summary = {}

    for c in C_VALUES:
        print(f"\n{'='*65}")
        print(f"  FULL SWEEP  c = {c:g}")
        print(f"{'='*65}")
        t0 = time.time()
        sw = run_sweep(all_hparams[c], c=c, grid_lo=GRID_LO, grid_hi=GRID_HI,
                       grid_n=FULL_N, n_iters=N_ITERS, seed=SEED, n_jobs=N_JOBS)
        all_sweep[c]   = sw
        all_summary[c] = summarise(sw)
        print(f"  done in {time.time()-t0:.1f}s")
        print(summary_table(all_summary[c], order=METHODS))

    # =========================================================================
    # Convergence table
    # =========================================================================
    iters_to_conv = {}

    for c in C_VALUES:
        iters_to_conv[c] = {}
        for m in METHODS:
            sm = all_summary[c][m]
            ul = sm.get("median_curve_UL")
            ll = sm.get("median_curve_LL")
            if ul is None or ll is None:
                iters_to_conv[c][m] = np.nan
                continue
            hit = np.where((ul < LOSS_TOL) & (ll < LOSS_TOL))[0]
            iters_to_conv[c][m] = int(hit[0] + 1) if len(hit) else np.nan

    print(f"\nIterations to reach median F < {LOSS_TOL:g} AND J < {LOSS_TOL:g}"
          f"  (— = not reached)\n")
    header = f"{'algorithm':<15}" + "".join(f"  c={c:>6g}" for c in C_VALUES)
    print(header)
    print("-" * len(header))
    for m in METHODS:
        row = f"{m:<15}"
        for c in C_VALUES:
            v = iters_to_conv[c][m]
            row += f"  {'—' if np.isnan(v) else int(v):>8}"
        print(row)

    # =========================================================================
    # Save
    # =========================================================================
    os.makedirs("results", exist_ok=True)
    c_tag = "_".join(f"{c:g}" for c in C_VALUES)
    tag   = f"cond_sweep_c{c_tag}_n{FULL_N}_it{N_ITERS}"

    csv_path = f"results/{tag}_iters.csv"
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["algorithm"] + [f"c={c:g}" for c in C_VALUES])
        for m in METHODS:
            wr.writerow([m] + [
                ("" if np.isnan(iters_to_conv[c][m]) else int(iters_to_conv[c][m]))
                for c in C_VALUES
            ])
    print(f"\nSaved: {csv_path}")

    hp_path = f"results/{tag}_hparams.json"
    with open(hp_path, "w") as fh:
        json.dump({f"c={c:g}": {m: hp for m, hp in hps.items()}
                   for c, hps in all_hparams.items()},
                  fh, indent=2, default=str)
    print(f"Saved: {hp_path}")

    detail_path = f"results/{tag}_detail.csv"
    with open(detail_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["c", "algorithm", "w0", "theta0",
                     "final_ul_loss", "final_ll_loss",
                     "final_w_err", "final_theta_err"])
        for c in C_VALUES:
            for m in METHODS:
                d = all_sweep[c][m]
                for (w0, t0), w, th, ul, ll in zip(
                        d["starts"], d["w"], d["theta"], d["ul_loss"], d["ll_loss"]):
                    wr.writerow([c, m, w0, t0,
                                 float(ul[-1]), float(ll[-1]),
                                 float(abs(w[-1] - W_STAR)),
                                 float(theta_error(th[-1]))])
    print(f"Saved: {detail_path}")
    print("\nDone.")
