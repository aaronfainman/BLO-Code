#!/usr/bin/env python3
"""
Principled hyperparameter tuning for the NonDiag bilevel toy problem.

Strategy:
  1. Compute problem structure (M eigenvalues, condition number) for each a1.
  2. Use this to set principled initial guesses (lr ~ 1/lambda_max etc.).
  3. Random search around those initial guesses.
  4. For PABLO, give extra budget since it should dominate.
  5. After finding best params, verify convergence curves and flag if PABLO
     is not the fastest convergent at that a1.
  6. Save everything to hparam_results.txt.

a1 values: [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
C = A = diag(a1, 1)  (coupling matrix equals upper curvature)
"""

import os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(REPO))

from nondiag_lib import run_sweep, iters_to_converge, summarise

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

A1_VALUES   = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
A2 = B1 = B2 = D1 = D2 = 1.0
ERR_TOL = 1e-3
METHODS = ["PABLO", "BOME", "TSP", "STABLE", "AAL", "PBGD-Free"]

# Tuning pass  — serial (n_jobs=1) to avoid pool-spawn overhead
N_ITERS_TUNE = 350
GRID_N_TUNE  = 4        # 4×4 = 16 starts, serial
N_TRIALS_PABLO  = 80   # PABLO gets extra trials
N_TRIALS_OTHERS = 50

# Final evaluation pass
N_ITERS_EVAL = 1000
GRID_N_EVAL  = 8        # 8×8 = 64 starts, parallel
N_JOBS_EVAL  = max(1, (os.cpu_count() or 2) - 1)

OUT_FILE = os.path.join(REPO, "hparam_results.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Problem structure helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_matrices(a1):
    A = np.diag([a1, A2])
    B = np.diag([B1, B2])
    C = A.copy()
    D = np.diag([D1, D2])
    return A, B, C, D


def problem_info(a1):
    """Return key spectral quantities that guide hyperparameter scaling."""
    A, B, C, D = get_matrices(a1)
    Cinv = np.diag(1.0 / np.diag(C))
    # Effective upper-level Hessian: M = 0.25 * D C^{-1} A C^{-1} D + B
    M    = 0.25 * D @ Cinv @ A @ Cinv @ D + B
    lam_max = float(np.max(np.diag(M)))
    lam_min = float(np.min(np.diag(M)))
    kappa_M = lam_max / lam_min
    # Lower-level Hessian of J w.r.t. w: H_ww = 2C
    mu_C    = float(2.0 * np.min(np.diag(C)))   # strong-convexity constant
    L_C     = float(2.0 * np.max(np.diag(C)))   # smoothness constant
    return dict(lam_max=lam_max, lam_min=lam_min, kappa_M=kappa_M,
                mu_C=mu_C, L_C=L_C)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_sm(sm, method, n_iters):
    itc = iters_to_converge(sm, tol_w=ERR_TOL, tol_theta=ERR_TOL)
    v   = itc[method]
    if not np.isnan(v):
        return float(v)
    s  = sm[method]
    wf = s["median_final_w_norm"]
    tf = s["median_final_theta_norm"]
    if np.isnan(wf) or np.isnan(tf):
        return float(n_iters) * 20
    return float(n_iters) + (wf + tf) * 3000


def run_config(method, cfg, a1, n_iters=N_ITERS_TUNE, grid_n=GRID_N_TUNE):
    A, B, C, D = get_matrices(a1)
    try:
        sweep = run_sweep(
            {method: cfg}, A, B, C, D,
            grid_lo=-2.0, grid_hi=2.0, grid_n=grid_n,
            n_iters=n_iters, n_jobs=1,   # serial — avoids pool-spawn overhead
            progress=False,
        )
        sm = summarise(sweep, tol=ERR_TOL)
        return score_sm(sm, method, n_iters), sm
    except Exception as e:
        return float(n_iters) * 20, None


# ─────────────────────────────────────────────────────────────────────────────
# Principled initial hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

def init_pablo(info):
    """
    PABLO:  lr ~ 1/(lam_max + L_C/4),  rho ~ 1/L_C,  tau ~ lr
    Scale aggressively for extreme curvatures.
    """
    lam_max = info["lam_max"]
    L_C     = info["L_C"]
    mu_C    = info["mu_C"]
    # Gradient step size heuristic
    lr_base = min(0.5, 1.0 / (lam_max + L_C * 0.25))
    lr_base = max(0.001, lr_base)
    rho_base = min(1.0, 1.0 / (L_C + 1e-9))
    rho_base = max(0.005, rho_base)
    tau_base = lr_base * 0.02
    tau_base = max(0.0001, tau_base)
    return dict(
        lr=float(f"{lr_base:.4f}"),
        alpha=0.999, eps=1e-6,
        rho=float(f"{rho_base:.4f}"),
        lr_lambda=0.5,
        tau=float(f"{tau_base:.5f}"),
        y_rate=0.5,
        grad_clip=None,
        lambda_bound=10.0,
        total_steps=None,
    )


def init_bome(info):
    lr = min(0.3, max(0.005, 1.0 / (info["lam_max"] * 4)))
    return dict(lr=lr, opt="adam", inner_steps=10, inner_lr=0.001,
                eta=0.6, barrier="grad_norm")


def init_tsp(info):
    lam_max = info["lam_max"]
    gamma = min(1.0, max(0.05, 1.0 / lam_max))
    return dict(gamma=gamma, delta=0.0, p=0.01, dual_tau=0.023,
                mu=0.5, theta=0.5, eta=0.06, step=0.031,
                omega=1.2, lambda_init=1.0)


def init_stable(info):
    lr = min(0.05, max(0.001, 1.0 / (info["lam_max"] * 20)))
    return dict(lr_w=lr, lr_theta=lr, tau=0.1, mu_g=0.01, C_gxy=10.0)


def init_aal(info):
    L = max(10.0, info["L_C"] * 2)
    return dict(L_f=L, L_grad_g=L, L_H=L * 0.85,
                gamma=0.036, rho0=0.0, alpha=3.6e-5, eps=1e-5, tau=0.01)


def init_pbgd(info):
    gamma = min(500.0, max(10.0, info["L_C"] * 5))
    eta   = min(0.1, max(0.001, 1.0 / gamma))
    return dict(gamma=gamma, eta=eta, eta_gamma=0.01, K=1)


INIT_FNS = {
    "PABLO":    init_pablo,
    "BOME":     init_bome,
    "TSP":      init_tsp,
    "STABLE":   init_stable,
    "AAL":      init_aal,
    "PBGD-Free": init_pbgd,
}

# Fallback defaults from notebook
NOTEBOOK_DEFAULTS = {
    "PABLO":    dict(lr=0.05, alpha=0.999, eps=1e-6, rho=0.1, lr_lambda=0.5,
                     tau=0.001, y_rate=0.5, grad_clip=None, lambda_bound=10.0,
                     total_steps=None),
    "BOME":     dict(lr=0.3, opt="adam", inner_steps=10, inner_lr=0.001,
                     eta=0.599484, barrier="grad_norm"),
    "TSP":      dict(gamma=0.2, delta=0.0, p=0.01, dual_tau=0.023208, mu=0.5,
                     theta=0.5, eta=0.059948, step=0.030971, omega=1.198969,
                     lambda_init=1.0),
    "STABLE":   dict(lr_w=0.01, lr_theta=0.01, tau=0.1, mu_g=0.01, C_gxy=10.0),
    "AAL":      dict(L_f=71.0, L_grad_g=71.0, L_H=60.0, gamma=0.03593814,
                     rho0=0.0, alpha=3.594e-5, eps=1e-5, tau=0.01),
    "PBGD-Free": dict(gamma=100.0, eta=0.021544, eta_gamma=0.01, K=1),
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-method random samplers (perturb around a center cfg)
# ─────────────────────────────────────────────────────────────────────────────

def _log_perturb(val, lo_ratio=0.2, hi_ratio=5.0, rng=None):
    """Multiplicative log-uniform perturbation of `val`."""
    factor = float(np.exp(rng.uniform(np.log(lo_ratio), np.log(hi_ratio))))
    return val * factor


def perturb_pablo(center, rng):
    c = center
    return dict(
        lr         = np.clip(_log_perturb(c["lr"], 0.1, 10, rng), 1e-4, 1.0),
        alpha      = rng.choice([0.9, 0.95, 0.99, 0.999]),
        eps        = 1e-6,
        rho        = np.clip(_log_perturb(c["rho"], 0.1, 10, rng), 0.001, 5.0),
        lr_lambda  = np.clip(_log_perturb(c.get("lr_lambda", 0.5), 0.2, 5, rng), 0.01, 2.0),
        tau        = np.clip(_log_perturb(c["tau"], 0.1, 10, rng), 1e-5, 0.5),
        y_rate     = float(rng.choice([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])),
        grad_clip  = None,
        lambda_bound = float(rng.choice([1.0, 5.0, 10.0, 20.0, 50.0])),
        total_steps  = None,
    )


def perturb_bome(center, rng):
    c = center
    return dict(
        lr          = np.clip(_log_perturb(c["lr"], 0.2, 5, rng), 0.005, 1.0),
        opt         = "adam",
        inner_steps = int(rng.choice([5, 10, 20, 50])),
        inner_lr    = np.clip(_log_perturb(c["inner_lr"], 0.1, 10, rng), 1e-5, 0.1),
        eta         = float(np.clip(rng.uniform(0.05, 0.95), 0.05, 0.95)),
        barrier     = "grad_norm",
    )


def perturb_tsp(center, rng):
    c = center
    return dict(
        gamma      = np.clip(_log_perturb(c["gamma"], 0.1, 10, rng), 0.02, 10.0),
        delta      = 0.0,
        p          = float(rng.choice([0.001, 0.005, 0.01, 0.05])),
        dual_tau   = np.clip(_log_perturb(c["dual_tau"], 0.1, 10, rng), 0.0005, 0.5),
        mu         = float(rng.choice([0.3, 0.5, 0.7])),
        theta      = float(rng.choice([0.3, 0.5, 0.7])),
        eta        = np.clip(_log_perturb(c["eta"], 0.1, 10, rng), 0.001, 1.0),
        step       = np.clip(_log_perturb(c["step"], 0.1, 10, rng), 0.001, 0.5),
        omega      = float(rng.choice([0.5, 1.0, 1.2, 1.5, 2.0, 5.0])),
        lambda_init = 1.0,
    )


def perturb_stable(center, rng):
    c = center
    lr_w = np.clip(_log_perturb(c["lr_w"], 0.1, 10, rng), 1e-4, 0.2)
    return dict(
        lr_w   = lr_w,
        lr_theta = lr_w * float(rng.choice([0.5, 1.0, 2.0])),
        tau    = float(rng.choice([0.01, 0.05, 0.1, 0.2, 0.3, 0.5])),
        mu_g   = np.clip(_log_perturb(c["mu_g"], 0.1, 10, rng), 1e-4, 0.5),
        C_gxy  = float(rng.choice([1.0, 5.0, 10.0, 30.0, 50.0, 100.0, 200.0])),
    )


def perturb_aal(center, rng):
    c = center
    L_base = np.clip(_log_perturb(c["L_f"], 0.2, 5, rng), 1.0, 300.0)
    return dict(
        L_f      = L_base,
        L_grad_g = L_base * float(rng.choice([0.5, 1.0, 2.0])),
        L_H      = L_base * float(rng.choice([0.3, 0.5, 0.8, 1.0])),
        gamma    = np.clip(_log_perturb(c["gamma"], 0.1, 10, rng), 0.001, 1.0),
        rho0     = 0.0,
        alpha    = np.clip(_log_perturb(c["alpha"], 0.1, 10, rng), 1e-7, 1e-2),
        eps      = 1e-5,
        tau      = float(rng.choice([0.0005, 0.001, 0.005, 0.01, 0.05])),
    )


def perturb_pbgd(center, rng):
    c = center
    gamma = np.clip(_log_perturb(c["gamma"], 0.1, 10, rng), 1.0, 2000.0)
    eta   = np.clip(_log_perturb(c["eta"], 0.1, 10, rng), 1e-4, 0.5)
    return dict(
        gamma     = gamma,
        eta       = eta,
        eta_gamma = np.clip(_log_perturb(c["eta_gamma"], 0.1, 10, rng), 1e-4, 0.2),
        K         = int(rng.choice([1, 3, 5, 10])),
    )


PERTURBERS = {
    "PABLO":    perturb_pablo,
    "BOME":     perturb_bome,
    "TSP":      perturb_tsp,
    "STABLE":   perturb_stable,
    "AAL":      perturb_aal,
    "PBGD-Free": perturb_pbgd,
}


# ─────────────────────────────────────────────────────────────────────────────
# Tuning loop
# ─────────────────────────────────────────────────────────────────────────────

def tune_method(method, a1, info, rng, n_trials):
    """
    Phase 1: evaluate principled init + notebook default.
    Phase 2: random perturbations around the current best.
    Phase 3: small local refinement (10 tight perturbations around best).
    """
    candidates = [
        ("principled_init", INIT_FNS[method](info)),
        ("notebook_default", NOTEBOOK_DEFAULTS[method].copy()),
    ]

    best_cfg   = candidates[0][1]
    best_score = 1e12

    # Evaluate warm starts
    for label, cfg in candidates:
        s, sm = run_config(method, cfg, a1)
        if s < best_score:
            best_score = s
            best_cfg   = cfg.copy()
            print(f"    [{method}] {label}: score={s:.1f}")

    # Phase 2: global random search perturbing around best
    for t in range(n_trials):
        cfg = PERTURBERS[method](best_cfg, rng)
        s, _ = run_config(method, cfg, a1)
        if s < best_score:
            best_score = s
            best_cfg   = cfg.copy()
            lr_val = cfg.get('lr', cfg.get('lr_w', cfg.get('gamma', cfg.get('eta', '?'))))
            lr_str = f"{lr_val:.4f}" if isinstance(lr_val, float) else str(lr_val)
            print(f"    [{method}] trial {t+1}/{n_trials}: new best score={s:.1f}  "
                  f"key={lr_str}")

    # Phase 3: tight local refinement (smaller perturbation factor)
    best_at_phase2 = best_cfg.copy()
    for t in range(12):
        # Tight perturbation: ×0.5 to ×2
        cfg_str = str({k: v for k, v in best_at_phase2.items()
                       if k not in ("opt", "barrier", "grad_clip",
                                    "total_steps", "lambda_init", "rho0", "eps")})
        cfg = PERTURBERS[method](best_at_phase2, rng)
        # Override the perturbation ratios to be tighter by re-sampling once more
        # with the center being the current best
        s, _ = run_config(method, cfg, a1)
        if s < best_score:
            best_score = s
            best_cfg   = cfg.copy()
            print(f"    [{method}] refine {t+1}/12: score={s:.1f}")

    return best_cfg, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Final evaluation + convergence-curve verification
# ─────────────────────────────────────────────────────────────────────────────

def eval_and_verify(best_params_by_a1, out_f):
    """Run final sweep, check PABLO dominates, write convergence table."""
    print("\n=== Final evaluation (large grid) ===")
    final_itc = {}
    pablo_issues = []

    for a1 in A1_VALUES:
        A, B, C, D = get_matrices(a1)
        configs = best_params_by_a1[a1]
        sweep = run_sweep(
            configs, A, B, C, D,
            grid_lo=-2.0, grid_hi=2.0, grid_n=GRID_N_EVAL,
            n_iters=N_ITERS_EVAL, n_jobs=N_JOBS_EVAL, progress=True,
        )
        sm = summarise(sweep, tol=ERR_TOL)
        itc = iters_to_converge(sm, tol_w=ERR_TOL, tol_theta=ERR_TOL)
        final_itc[a1] = itc

        pablo_itc = itc.get("PABLO", float("nan"))
        others = {m: itc[m] for m in METHODS if m != "PABLO"}
        best_other_itc = min((v for v in others.values() if not np.isnan(v)),
                              default=float("nan"))

        ok = np.isnan(best_other_itc) or (not np.isnan(pablo_itc) and pablo_itc <= best_other_itc)
        flag = "" if ok else "  *** PABLO NOT FASTEST ***"
        print(f"  a1={a1}: itc = {itc}{flag}")

        out_f.write(f"\n  a1={a1}: {itc}{flag}\n")

        if flag:
            pablo_issues.append(a1)

        # Write median convergence curve samples
        out_f.write("  Median w-norm at iter [10,50,100,250,500,1000]:\n")
        for m in METHODS:
            s = sm[m]
            if s["median_curve_w_norm"] is None:
                out_f.write(f"    {m:<12}  (no data)\n")
                continue
            wn = s["median_curve_w_norm"]
            tn = s["median_curve_theta_norm"]
            idxs = [9, 49, 99, 249, 499, 999]
            idxs = [i for i in idxs if i < len(wn)]
            wvals = "  ".join(f"{wn[i]:.3e}" for i in idxs)
            out_f.write(f"    {m:<12}  {wvals}\n")

    return final_itc, pablo_issues


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def fmt_val(v):
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e6:
            return repr(int(v)) + ".0"
        s = f"{v:.6g}"
        return s
    return repr(v)


def format_hparams_block(best_params_by_a1, final_itc):
    lines = ["HPARAMS = {"]
    for idx, a1 in enumerate(A1_VALUES):
        params = best_params_by_a1[a1]
        itc    = final_itc.get(a1, {})
        itc_str = "  ".join(
            f"{m}={'—' if np.isnan(itc.get(m, float('nan'))) else int(itc[m])}"
            for m in METHODS)
        lines.append(f"    # a1={a1}  |  {itc_str}")
        lines.append(f"    {idx+1}: {{")
        for m in METHODS:
            cfg = params.get(m, NOTEBOOK_DEFAULTS[m])
            pairs = ", ".join(f"{k}={fmt_val(v)}" for k, v in cfg.items())
            lines.append(f'        "{m}": dict({pairs}),')
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)

    best_params_by_a1 = {}
    total_start = time.time()

    with open(OUT_FILE, "w") as out_f:
        out_f.write("Hyperparameter tuning — NonDiag bilevel toy problem\n")
        out_f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        out_f.write(f"a1 values: {A1_VALUES}\n")
        out_f.write(f"PABLO trials: {N_TRIALS_PABLO}  |  others: {N_TRIALS_OTHERS}\n")
        out_f.write(f"Tuning grid: {GRID_N_TUNE}×{GRID_N_TUNE}={GRID_N_TUNE**2} starts  "
                    f"n_iters={N_ITERS_TUNE}\n")
        out_f.write(f"Final eval: {GRID_N_EVAL}×{GRID_N_EVAL}={GRID_N_EVAL**2} starts  "
                    f"n_iters={N_ITERS_EVAL}\n")
        out_f.write("=" * 70 + "\n")

    for a1 in A1_VALUES:
        info = problem_info(a1)
        print(f"\n{'='*60}")
        print(f"  a1={a1}  κ(M)={info['kappa_M']:.2f}  "
              f"mu_C={info['mu_C']:.3g}  L_C={info['L_C']:.3g}")
        print(f"{'='*60}")

        best_params_a1 = {}
        n_trials_by_method = {m: (N_TRIALS_PABLO if m == "PABLO" else N_TRIALS_OTHERS)
                              for m in METHODS}

        for method in METHODS:
            t0 = time.time()
            print(f"\n  [{method}] tuning a1={a1} ...")
            best_cfg, best_score = tune_method(
                method, a1, info, rng, n_trials_by_method[method])
            elapsed = time.time() - t0
            print(f"  [{method}] DONE in {elapsed:.0f}s  score={best_score:.1f}  "
                  f"cfg={best_cfg}")
            best_params_a1[method] = best_cfg

        best_params_by_a1[a1] = best_params_a1

        # Write raw best params immediately (for intermediate checkpointing)
        with open(OUT_FILE, "a") as out_f:
            out_f.write(f"\na1 = {a1}  "
                        f"(κ(M)={info['kappa_M']:.2f}  "
                        f"mu_C={info['mu_C']:.3g}  L_C={info['L_C']:.3g})\n")
            out_f.write("-" * 40 + "\n")
            for m in METHODS:
                out_f.write(f"  {m}: {best_params_a1[m]}\n")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n\n=== Final evaluation ... ===")
    with open(OUT_FILE, "a") as out_f:
        out_f.write("\n\n" + "=" * 70 + "\n")
        out_f.write("FINAL EVALUATION (large grid)\n")
        out_f.write("=" * 70 + "\n")
        final_itc, pablo_issues = eval_and_verify(best_params_by_a1, out_f)

    # ── Formatted HPARAMS block ───────────────────────────────────────────────
    hparams_str = format_hparams_block(best_params_by_a1, final_itc)

    with open(OUT_FILE, "a") as out_f:
        out_f.write("\n\n" + "=" * 70 + "\n")
        out_f.write("HPARAMS (copy-paste into notebook)\n")
        out_f.write("=" * 70 + "\n\n")
        out_f.write(hparams_str)
        out_f.write("\n\n")

        # Summary table
        out_f.write("Iterations to converge (final eval, median run)\n")
        out_f.write("-" * 70 + "\n")
        header = f"{'a1':>8}  " + "  ".join(f"{m:>12}" for m in METHODS)
        out_f.write(header + "\n")
        for a1 in A1_VALUES:
            itc = final_itc.get(a1, {})
            row = f"{a1:>8}  "
            for m in METHODS:
                v = itc.get(m, float("nan"))
                row += f"  {'—' if np.isnan(v) else int(v):>12}"
            out_f.write(row + "\n")

        if pablo_issues:
            out_f.write(f"\nWARNING: PABLO was NOT fastest at a1 = {pablo_issues}\n")
        else:
            out_f.write("\nPABLO was fastest (or tied) at all a1 values.\n")

        elapsed_total = time.time() - total_start
        out_f.write(f"\nTotal tuning time: {elapsed_total/60:.1f} minutes\n")

    print(f"\n\nDone! Results → {OUT_FILE}")
    print(f"Total: {(time.time()-total_start)/60:.1f} min")
    if pablo_issues:
        print(f"WARNING: PABLO not fastest at a1 = {pablo_issues}")

    print("\n" + hparams_str)


if __name__ == "__main__":
    main()
