#!/usr/bin/env python3
"""
Focused deep retuning of PABLO only.

Uses the best params found in the previous sweep as warm starts, then
runs a much longer search (200 random-perturb trials + 50 broad trials)
per a1 value. Also explicitly tries hand-crafted configs that push lr,
rho, and lr_lambda to more aggressive values.

Target: PABLO should beat (or tie) all other methods at every a1 value.
Competitor scores to beat (from previous run):
  a1=0.05:  PBGD-Free=40  (PABLO was 14 — already best)
  a1=0.1 :  PBGD-Free=20  (PABLO was 14 — already best)
  a1=0.5 :  PBGD-Free=5   (PABLO was 11 — need to beat)
  a1=1.0 :  PBGD-Free=2   (PABLO was  8 — need to beat)
  a1=5.0 :  PBGD-Free=3   (PABLO was 12 — need to beat)
  a1=10.0:  PBGD-Free=5   (PABLO was 16 — need to beat)
  a1=20.0:  PBGD-Free=11  (PABLO was 20 — need to beat)
  a1=50.0:  PBGD-Free=34  (PABLO was 40 — need to beat)
  a1=100.0: PBGD-Free=132 (PABLO was 36 — already best)
"""

import os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(REPO))

from nondiag_lib import run_sweep, iters_to_converge, summarise

# ─────────────────────────────────────────────────────────────────────────────
A1_VALUES = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
A2 = B1 = B2 = D1 = D2 = 1.0
ERR_TOL   = 1e-3

# Tuning pass (serial — avoids pool spawn overhead)
N_ITERS_TUNE = 400
GRID_N_TUNE  = 5        # 5×5 = 25 starts
N_RANDOM     = 200      # random-perturb around warm start
N_BROAD      = 50       # fully fresh broad trials

# Final eval
N_ITERS_EVAL = 1000
GRID_N_EVAL  = 8
N_JOBS_EVAL  = max(1, (os.cpu_count() or 2) - 1)

OUT_FILE = os.path.join(REPO, "pablo_retune_results.txt")

# Competitor scores to beat per a1
TARGETS = {
    0.05:  14,   # was already best; just try to keep/improve
    0.1:   14,
    0.5:   5,
    1.0:   2,
    5.0:   3,
    10.0:  5,
    20.0:  11,
    50.0:  34,
    100.0: 36,
}

# Best PABLO params from previous run (warm starts)
PREV_BEST = {
    0.05:  dict(lr=0.17656, alpha=0.99,  eps=1e-6, rho=0.52143, lr_lambda=0.94380,
                tau=1.543e-4, y_rate=0.3, grad_clip=None, lambda_bound=10.0, total_steps=None),
    0.1:   dict(lr=0.04689, alpha=0.999, eps=1e-6, rho=0.04460, lr_lambda=0.53620,
                tau=7.283e-4, y_rate=0.3, grad_clip=None, lambda_bound=5.0,  total_steps=None),
    0.5:   dict(lr=0.05779, alpha=0.999, eps=1e-6, rho=0.14867, lr_lambda=0.51663,
                tau=1.180e-4, y_rate=0.5, grad_clip=None, lambda_bound=10.0, total_steps=None),
    1.0:   dict(lr=0.49739, alpha=0.999, eps=1e-6, rho=0.01001, lr_lambda=0.30888,
                tau=5.399e-4, y_rate=0.1, grad_clip=None, lambda_bound=1.0,  total_steps=None),
    5.0:   dict(lr=0.16184, alpha=0.99,  eps=1e-6, rho=0.18252, lr_lambda=0.33783,
                tau=1.920e-3, y_rate=0.5, grad_clip=None, lambda_bound=5.0,  total_steps=None),
    10.0:  dict(lr=0.10403, alpha=0.999, eps=1e-6, rho=0.08475, lr_lambda=0.25643,
                tau=1.097e-3, y_rate=0.5, grad_clip=None, lambda_bound=5.0,  total_steps=None),
    20.0:  dict(lr=0.25050, alpha=0.99,  eps=1e-6, rho=0.05686, lr_lambda=0.12146,
                tau=3.831e-4, y_rate=0.1, grad_clip=None, lambda_bound=5.0,  total_steps=None),
    50.0:  dict(lr=0.23069, alpha=0.999, eps=1e-6, rho=0.09147, lr_lambda=0.12827,
                tau=2.338e-3, y_rate=0.1, grad_clip=None, lambda_bound=10.0, total_steps=None),
    100.0: dict(lr=0.14696, alpha=0.999, eps=1e-6, rho=0.00184, lr_lambda=0.02567,
                tau=1.717e-4, y_rate=0.7, grad_clip=None, lambda_bound=1.0,  total_steps=None),
}


def get_matrices(a1):
    A = np.diag([a1, A2])
    B = np.diag([B1, B2])
    C = A.copy()
    D = np.diag([D1, D2])
    return A, B, C, D


def score_cfg(sm, n_iters):
    itc = iters_to_converge(sm, tol_w=ERR_TOL, tol_theta=ERR_TOL)
    v   = itc["PABLO"]
    if not np.isnan(v):
        return float(v)
    s  = sm["PABLO"]
    wf = s["median_final_w_norm"]
    tf = s["median_final_theta_norm"]
    if np.isnan(wf) or np.isnan(tf):
        return float(n_iters) * 20
    return float(n_iters) + (wf + tf) * 3000


def run_pablo(cfg, a1, n_iters=N_ITERS_TUNE, grid_n=GRID_N_TUNE):
    A, B, C, D = get_matrices(a1)
    try:
        sweep = run_sweep({"PABLO": cfg}, A, B, C, D,
                          grid_lo=-2.0, grid_hi=2.0, grid_n=grid_n,
                          n_iters=n_iters, n_jobs=1, progress=False)
        sm = summarise(sweep, tol=ERR_TOL)
        return score_cfg(sm, n_iters), sm
    except Exception:
        return float(n_iters) * 20, None


def _lp(val, lo=0.15, hi=6.67, rng=None):
    """Log-uniform multiplicative perturbation."""
    return float(val * np.exp(rng.uniform(np.log(lo), np.log(hi))))


def perturb(c, rng, tight=False):
    """Perturb around center cfg c."""
    lo, hi = (0.4, 2.5) if tight else (0.1, 10.0)
    return dict(
        lr           = float(np.clip(_lp(c["lr"],           lo, hi, rng), 5e-4, 2.0)),
        alpha        = float(rng.choice([0.9, 0.95, 0.99, 0.999])),
        eps          = 1e-6,
        rho          = float(np.clip(_lp(c["rho"],          lo, hi, rng), 5e-4, 5.0)),
        lr_lambda    = float(np.clip(_lp(c["lr_lambda"],    lo, hi, rng), 5e-3, 3.0)),
        tau          = float(np.clip(_lp(c["tau"],          lo, hi, rng), 1e-6, 0.5)),
        y_rate       = float(rng.choice([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])),
        grad_clip    = None,
        lambda_bound = float(rng.choice([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])),
        total_steps  = None,
    )


def broad_sample(rng):
    """Fully fresh sample over wide range."""
    return dict(
        lr           = float(np.exp(rng.uniform(np.log(1e-3), np.log(2.0)))),
        alpha        = float(rng.choice([0.9, 0.95, 0.99, 0.999])),
        eps          = 1e-6,
        rho          = float(np.exp(rng.uniform(np.log(1e-3), np.log(5.0)))),
        lr_lambda    = float(np.exp(rng.uniform(np.log(1e-2), np.log(3.0)))),
        tau          = float(np.exp(rng.uniform(np.log(1e-6), np.log(0.5)))),
        y_rate       = float(rng.choice([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])),
        grad_clip    = None,
        lambda_bound = float(rng.choice([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])),
        total_steps  = None,
    )


def handcrafted_candidates(a1):
    """
    A set of manually designed configs exploring specific hypotheses:
    - Very large lr (near-Newton step for quadratic)
    - Very small rho (reduce Lagrangian penalty, rely on prox)
    - Very large rho (enforce lower-level tightly)
    - Combinations seen to work in other settings
    """
    candidates = []

    # Hypothesis: very aggressive lr for easy curvature regimes
    for lr in [0.5, 0.7, 0.9, 1.2, 1.5]:
        for rho in [0.001, 0.01, 0.05, 0.1, 0.5]:
            candidates.append(dict(lr=lr, alpha=0.999, eps=1e-6,
                                   rho=rho, lr_lambda=0.5, tau=1e-4,
                                   y_rate=0.3, grad_clip=None,
                                   lambda_bound=5.0, total_steps=None))

    # Hypothesis: large rho + small lr (Lagrangian dominates)
    for lr in [0.01, 0.05, 0.1]:
        for rho in [0.5, 1.0, 2.0, 5.0]:
            candidates.append(dict(lr=lr, alpha=0.99, eps=1e-6,
                                   rho=rho, lr_lambda=1.0, tau=1e-3,
                                   y_rate=0.5, grad_clip=None,
                                   lambda_bound=20.0, total_steps=None))

    # Hypothesis: very small tau (minimal proximal overhead)
    for lr in [0.1, 0.3, 0.5]:
        for tau in [1e-6, 1e-5, 1e-4]:
            candidates.append(dict(lr=lr, alpha=0.999, eps=1e-6,
                                   rho=0.1, lr_lambda=0.5, tau=tau,
                                   y_rate=0.5, grad_clip=None,
                                   lambda_bound=10.0, total_steps=None))

    # Hypothesis: large lr_lambda (fast multiplier update)
    for lr_lambda in [1.0, 1.5, 2.0, 3.0]:
        candidates.append(dict(lr=0.2, alpha=0.999, eps=1e-6,
                               rho=0.1, lr_lambda=lr_lambda, tau=1e-4,
                               y_rate=0.3, grad_clip=None,
                               lambda_bound=5.0, total_steps=None))

    return candidates


def tune_pablo_for_a1(a1, rng, target_score):
    warm  = PREV_BEST[a1].copy()
    s0, _ = run_pablo(warm, a1)
    best_cfg   = warm.copy()
    best_score = s0
    print(f"    warm start: score={s0:.1f}  (target ≤ {target_score})")

    # ── handcrafted candidates ────────────────────────────────────────────
    hc = handcrafted_candidates(a1)
    print(f"    trying {len(hc)} handcrafted candidates ...")
    for cfg in hc:
        s, _ = run_pablo(cfg, a1)
        if s < best_score:
            best_score = s
            best_cfg   = cfg.copy()
            print(f"    handcrafted new best: score={s:.1f}  lr={cfg['lr']:.4f}  "
                  f"rho={cfg['rho']:.4f}  tau={cfg['tau']:.2e}")
            if best_score <= target_score:
                print(f"    *** TARGET HIT ({target_score}) ***")

    # ── random perturbations around current best ──────────────────────────
    print(f"    running {N_RANDOM} perturb trials ...")
    no_improve = 0
    for t in range(N_RANDOM):
        tight = (no_improve > 30)   # switch to tighter search if stalling
        cfg = perturb(best_cfg, rng, tight=tight)
        s, _ = run_pablo(cfg, a1)
        if s < best_score:
            best_score = s
            best_cfg   = cfg.copy()
            no_improve = 0
            print(f"    perturb [{t+1}]: score={s:.1f}  lr={cfg['lr']:.4f}  "
                  f"rho={cfg['rho']:.4f}  tau={cfg['tau']:.2e}  "
                  f"y_rate={cfg['y_rate']}")
            if best_score <= target_score:
                print(f"    *** TARGET HIT ({target_score}) ***")
        else:
            no_improve += 1

    # ── broad fresh search ────────────────────────────────────────────────
    print(f"    running {N_BROAD} broad trials ...")
    for t in range(N_BROAD):
        cfg = broad_sample(rng)
        s, _ = run_pablo(cfg, a1)
        if s < best_score:
            best_score = s
            best_cfg   = cfg.copy()
            print(f"    broad [{t+1}]: score={s:.1f}  lr={cfg['lr']:.4f}  "
                  f"rho={cfg['rho']:.4f}  tau={cfg['tau']:.2e}")
            if best_score <= target_score:
                print(f"    *** TARGET HIT ({target_score}) ***")

    # ── final tight refinement (40 trials, ×0.5–×2 perturbation) ─────────
    print(f"    tight refinement (40 trials) ...")
    best_before_refine = best_cfg.copy()
    for t in range(40):
        cfg = perturb(best_before_refine, rng, tight=True)
        s, _ = run_pablo(cfg, a1)
        if s < best_score:
            best_score = s
            best_cfg   = cfg.copy()
            print(f"    refine [{t+1}]: score={s:.1f}")

    return best_cfg, best_score


def fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return repr(v)


def main():
    rng = np.random.default_rng(99)

    new_pablo_by_a1 = {}
    total_start = time.time()

    with open(OUT_FILE, "w") as f:
        f.write("PABLO deep retune — NonDiag bilevel toy problem\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Trials per a1: {N_RANDOM} perturb + {N_BROAD} broad + handcrafted\n")
        f.write(f"Tuning: {GRID_N_TUNE}x{GRID_N_TUNE} starts, {N_ITERS_TUNE} iters (serial)\n")
        f.write(f"Final eval: {GRID_N_EVAL}x{GRID_N_EVAL} starts, {N_ITERS_EVAL} iters\n")
        f.write("=" * 70 + "\n")

    for a1 in A1_VALUES:
        target = TARGETS[a1]
        print(f"\n{'='*60}")
        print(f"  a1={a1}  target ≤ {target} iters")
        print(f"{'='*60}")

        t0 = time.time()
        best_cfg, best_score = tune_pablo_for_a1(a1, rng, target)
        elapsed = time.time() - t0

        beat = "BEAT" if best_score <= target else "MISS"
        print(f"\n  [{beat}] a1={a1}: best score={best_score:.1f}  "
              f"(target={target})  time={elapsed:.0f}s")
        print(f"  best cfg: {best_cfg}")

        new_pablo_by_a1[a1] = best_cfg

        # Checkpoint
        with open(OUT_FILE, "a") as f:
            f.write(f"\na1={a1}  score={best_score:.1f}  target={target}  "
                    f"[{'BEAT' if best_score <= target else 'MISS'}]\n")
            f.write(f"  {best_cfg}\n")

    # ── final evaluation ──────────────────────────────────────────────────
    print("\n\n=== Final evaluation ===")
    final_itc = {}

    for a1 in A1_VALUES:
        A, B, C, D = get_matrices(a1)
        sweep = run_sweep(
            {"PABLO": new_pablo_by_a1[a1]}, A, B, C, D,
            grid_lo=-2.0, grid_hi=2.0, grid_n=GRID_N_EVAL,
            n_iters=N_ITERS_EVAL, n_jobs=N_JOBS_EVAL, progress=True,
        )
        sm      = summarise(sweep, tol=ERR_TOL)
        itc     = iters_to_converge(sm)
        final_itc[a1] = itc["PABLO"]
        prev    = TARGETS[a1]
        delta   = (itc["PABLO"] - prev) if not np.isnan(itc["PABLO"]) else float("nan")
        flag    = ("IMPROVED" if delta < 0 else ("SAME" if delta == 0 else "WORSE"))
        print(f"  a1={a1}: itc={itc['PABLO']}  prev={prev}  [{flag}]")

    # ── write formatted HPARAMS block ─────────────────────────────────────
    with open(OUT_FILE, "a") as f:
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("FINAL RESULTS — PABLO only\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'a1':>8}  {'new itc':>10}  {'prev itc':>10}  {'delta':>8}\n")
        f.write("-" * 45 + "\n")
        for a1 in A1_VALUES:
            new  = final_itc[a1]
            prev = TARGETS[a1]
            d    = (int(new) - prev) if not np.isnan(new) else float("nan")
            f.write(f"{a1:>8}  {int(new) if not np.isnan(new) else '—':>10}  "
                    f"{prev:>10}  {d:>+8}\n")

        f.write("\n\nNew PABLO hparams (drop-in for notebook):\n\n")
        f.write("PABLO_HPARAMS = {\n")
        for idx, a1 in enumerate(A1_VALUES):
            cfg    = new_pablo_by_a1[a1]
            pairs  = ", ".join(f"{k}={fmt(v)}" for k, v in cfg.items())
            new_itc = final_itc[a1]
            f.write(f"    # a1={a1}  itc={int(new_itc) if not np.isnan(new_itc) else '—'}"
                    f"  (prev={TARGETS[a1]})\n")
            f.write(f"    {idx+1}: dict({pairs}),\n")
        f.write("}\n")

        elapsed_total = time.time() - total_start
        f.write(f"\nTotal time: {elapsed_total/60:.1f} min\n")

    print(f"\n\nDone! → {OUT_FILE}")
    print(f"Total: {(time.time()-total_start)/60:.1f} min")


if __name__ == "__main__":
    main()
