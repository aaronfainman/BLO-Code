#!/usr/bin/env python3
"""
Overnight hyperparameter tuning for ridge regression BLO.

Comprehensive random search over ALL hyperparameters for each (κ, method).
AUC = median over 4×4 init grid of mean(‖w_t−w*‖ + |θ_t−θ*|) across iterations.

Saves:
  tuning_checkpoint.json  — incremental progress (resume-safe)
  tuning_results.json     — best configs per (κ, method)

Then patches cell-8 and cell-6 of Exp_RidgeRegression.ipynb.
"""

import json, os, sys, time, traceback
import numpy as np
import multiprocessing as mp

# ── path setup — runs in main AND in spawned workers ─────────────────────────
HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in [HERE, PARENT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ridge_lib
from ridge_lib import run_single

# ── constants (visible to worker subprocesses via module import) ──────────────
KAPPA_VALUES = [1.0, 10.0, 100.0]
METHODS_RUN  = ["PABLO", "BOME", "TSP", "STABLE", "AAL", "PBGD-Free"]
DATA_KWARGS  = dict(w_true=[1.0, 1.0], n_train=200, n_val=200, noise_std=0.1)
DATA_SEED    = 0

TUNE_N     = 4        # 4×4 = 16 starting points
TUNE_ITERS = 500      # iterations per tuning trial
N_TRIALS   = 500      # random configs per (κ, method)
N_JOBS     = max(1, (os.cpu_count() or 2) - 1)

W0_GRID = np.linspace(-2.0, 2.0, TUNE_N).tolist()
T0_GRID = np.linspace(0.1, 5.0, TUNE_N).tolist()
STARTS  = [(w0, t0) for w0 in W0_GRID for t0 in T0_GRID]   # 16 starts

CHECKPOINT   = os.path.join(HERE, "tuning_checkpoint.json")
RESULTS_FILE = os.path.join(HERE, "tuning_results.json")
NB_PATH      = os.path.join(HERE, "Exp_RidgeRegression.ipynb")


# =============================================================================
# Search-space sampler
# =============================================================================

def _to_py(v):
    """Convert numpy scalar types to native Python (for JSON safety)."""
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.bool_): return bool(v)
    return v


def sample_config(method, rng):
    """Return a random hyperparameter config (all values are native Python types)."""
    def LogU(lo, hi): return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    def C(choices):   return _to_py(rng.choice(list(choices)))

    if method == "PABLO":
        return dict(
            lr           = LogU(5e-4, 0.8),
            alpha        = C([0.9, 0.95, 0.99, 0.999]),
            eps          = LogU(1e-8, 1e-3),
            rho          = LogU(1e-3, 20.0),
            lr_lambda    = C([0.0, 0.0, 0.001, 0.01, 0.1]),
            tau          = LogU(1e-3, 100.0),
            y_rate       = LogU(0.02, 10.0),
            grad_clip    = (None if rng.random() < 0.3
                            else float(LogU(0.5, 100.0))),
            lambda_bound = LogU(0.1, 100.0),
            total_steps  = None,
        )

    if method == "BOME":
        return dict(
            lr          = LogU(1e-3, 1.0),
            inner_lr    = LogU(1e-3, 1.0),
            inner_steps = int(C([5, 10, 20, 50])),
            eta         = LogU(0.01, 2.0),
            opt         = str(C(["adam", "adam", "sgd", "rmsprop"])),
            barrier     = "grad_norm",
        )

    if method == "TSP":
        step = LogU(1e-3, 0.5)
        # 60% tied tau, 40% independent
        dual_tau = step if rng.random() < 0.6 else LogU(1e-3, 0.5)
        return dict(
            step        = step,
            eta         = LogU(1e-3, 0.5),
            dual_tau    = dual_tau,
            gamma       = LogU(0.02, 10.0),
            delta       = 0.0,
            p           = C([0.001, 0.01, 0.05, 0.1]),
            mu          = C([0.3, 0.5, 0.7, 0.9]),
            theta       = C([0.3, 0.5, 1.0]),
            omega       = LogU(1e-3, 1.0),
            lambda_init = C([0.1, 0.5, 1.0, 5.0, 10.0]),
        )

    if method == "STABLE":
        return dict(
            lr_w    = LogU(1e-3, 1.0),
            lr_theta= LogU(1e-3, 2.0),
            tau     = LogU(0.01, 20.0),
            mu_g    = LogU(1e-5, 0.5),
            C_gxy   = LogU(0.5, 500.0),
        )

    if method == "AAL":
        return dict(
            alpha     = LogU(1e-5, 0.05),
            tau       = LogU(1e-4, 20.0),
            L_f       = LogU(0.5, 200.0),
            L_grad_g  = LogU(0.5, 200.0),
            L_H       = LogU(0.5, 200.0),
            gamma     = LogU(0.05, 20.0),
            rho0      = 0.0,
            eps       = 1e-5,
            tau_decay = False,
        )

    if method == "PBGD-Free":
        return dict(
            gamma    = LogU(0.05, 500.0),
            eta      = LogU(1e-3, 5.0),
            eta_gamma= LogU(1e-3, 2.0),
            K        = int(C([1, 5, 10, 20, 50])),
        )

    raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Evaluation worker  (module-level so spawn can pickle it)
# =============================================================================

def _eval_worker(args):
    """Evaluate one (method, cfg, kappa) on all starting points. Returns AUC."""
    import torch
    torch.set_num_threads(1)
    method, cfg, kappa = args
    cov = np.diag([kappa, 1.0])
    errors = []
    for w0, t0 in STARTS:
        try:
            r = run_single(method, cfg, cov, w0=w0, theta0=t0,
                           n_iters=TUNE_ITERS, data_seed=DATA_SEED,
                           data_kwargs=DATA_KWARGS)
        except Exception:
            errors.append(1e6)
            continue
        if r["diverged"] or len(r["w"]) == 0:
            errors.append(1e6)
        else:
            w_err = np.linalg.norm(r["w"] - r["w_opt"][None, :], axis=-1)
            t_err = np.abs(r["theta"] - r["theta_opt"])
            errors.append(float((w_err + t_err).mean()))
    return float(np.median(errors))


# =============================================================================
# Checkpoint helpers
# =============================================================================

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            raw = json.load(f)
        # Convert list-of-2 back to (auc, cfg)
        return {k: (v[0], v[1]) for k, v in raw.items()}
    return {}


def save_checkpoint(results):
    serial = {k: [v[0], v[1]] for k, v in results.items()}
    with open(CHECKPOINT, "w") as f:
        json.dump(serial, f, indent=2, default=str)


# =============================================================================
# Notebook patching
# =============================================================================

def _cfg_repr(cfg):
    """Format a config dict as a readable multi-line Python dict literal."""
    items = []
    for k, v in cfg.items():
        items.append(f"            {k!r}: {v!r}")
    return "dict(\n" + ",\n".join(items) + ",\n        )"


def build_cell8_source(all_hparams, aucs):
    """Build Python source string for notebook cell-8."""
    lines = []
    lines.append("# Tuned by run_tuning.py — comprehensive random search\n")
    lines.append(f"# {N_TRIALS} trials × {TUNE_N}×{TUNE_N} starts × {TUNE_ITERS} iters/trial\n")
    lines.append("# AUC = median(mean_t(‖w_t−w*‖ + |θ_t−θ*|))\n")
    lines.append("#\n")
    # AUC summary table
    hdr = "#   {:<12}".format("method") + "".join(f"  κ={k:>6g}" for k in KAPPA_VALUES)
    lines.append(hdr + "\n")
    for m in METHODS_RUN:
        vals = "".join(
            f"  {aucs.get(f'{k}_{m}', float('nan')):>8.4f}"
            for k in KAPPA_VALUES
        )
        lines.append(f"#   {m:<12}{vals}\n")
    lines.append("all_hparams = {\n")
    for kappa in KAPPA_VALUES:
        lines.append(f"    {kappa!r}: {{\n")
        for m in METHODS_RUN:
            cfg = all_hparams.get(kappa, {}).get(m, {})
            auc = aucs.get(f"{kappa}_{m}", float("nan"))
            lines.append(f"        # AUC={auc:.4f}\n")
            lines.append(f"        {m!r}: {repr(cfg)},\n")
        lines.append("    },\n")
    lines.append("}\n")
    lines.append("tuning_log = {}\n")
    return lines


def build_hparams_base_source(all_hparams):
    """Build Python source for the HPARAMS_BASE block (κ=1 best configs)."""
    if 1.0 not in all_hparams:
        return None
    hp1 = all_hparams[1.0]
    lines = []
    lines.append("# ── conditioning sweep ────────────────────────────────────────────────────────\n")
    lines.append("KAPPA_VALUES = [1.0, 10.0, 100.0]\n")
    lines.append("COV_BY_KAPPA = {k: np.diag([k, 1.0]) for k in KAPPA_VALUES}\n")
    lines.append("\n")
    lines.append("METHODS_RUN = [\"PABLO\", \"BOME\", \"TSP\", \"STABLE\", \"AAL\", \"PBGD-Free\"]\n")
    lines.append("\n")
    lines.append("# ── best hyperparameters at κ=1 (from run_tuning.py) ─────────────────────────\n")
    lines.append("HPARAMS_BASE = {\n")
    for m in METHODS_RUN:
        cfg = hp1.get(m, {})
        lines.append(f"    {m!r}: {repr(cfg)},\n")
    lines.append("}\n")
    lines.append("\n")
    lines.append("STEP_KEYS = {\n")
    lines.append('    "PABLO":     ["lr", "y_rate"],\n')
    lines.append('    "BOME":      ["lr", "inner_lr"],\n')
    lines.append('    "TSP":       ["step", "eta", "dual_tau"],\n')
    lines.append('    "STABLE":    ["lr_w", "lr_theta"],\n')
    lines.append('    "AAL":       ["alpha", "tau"],\n')
    lines.append('    "PBGD-Free": ["eta", "eta_gamma"],\n')
    lines.append("}\n")
    lines.append("\n")
    lines.append("SCALES = [0.1, 0.3, 1.0, 3.0, 10.0]\n")
    lines.append("\n")
    lines.append("# ── run settings ─────────────────────────────────────────────────────────────\n")
    lines.append("N_ITERS      = 2000\n")
    lines.append("TUNE_N       = 4\n")
    lines.append("FULL_N       = 10\n")
    lines.append("W0_LO, W0_HI         = -2.0, 2.0\n")
    lines.append("THETA0_LO, THETA0_HI = 0.1, 5.0\n")
    lines.append("ERR_TOL      = 1e-3\n")
    lines.append("N_JOBS       = max(1, (os.cpu_count() or 2) - 1)\n")
    lines.append("\n")
    lines.append("n_tune_runs = len(SCALES) * len(METHODS_RUN) * TUNE_N**2\n")
    lines.append("n_full_runs = len(METHODS_RUN) * FULL_N**2\n")
    lines.append("print(f'Tuning:     {n_tune_runs} runs × {N_ITERS} iters per κ value')\n")
    lines.append("print(f'Full sweep: {n_full_runs} runs × {N_ITERS} iters per κ value')\n")
    lines.append("print(f'κ values: {KAPPA_VALUES}')\n")
    return lines


def patch_notebook(all_hparams, aucs):
    """Patch cell-8 (and optionally cell-6) of Exp_RidgeRegression.ipynb."""
    with open(NB_PATH) as f:
        nb = json.load(f)

    cell8_src  = build_cell8_source(all_hparams, aucs)
    cell6_src  = build_hparams_base_source(all_hparams)

    patched = {"cell-8": False, "cell-6": False}
    for cell in nb["cells"]:
        cid = cell.get("id", "")
        if cid == "cell-8":
            cell["source"]  = cell8_src
            cell["outputs"] = []
            patched["cell-8"] = True
        elif cid == "cell-6" and cell6_src is not None:
            cell["source"]  = cell6_src
            cell["outputs"] = []
            patched["cell-6"] = True

    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)

    for cid, ok in patched.items():
        status = "patched" if ok else "NOT FOUND"
        print(f"  {cid}: {status}")


# =============================================================================
# Main
# =============================================================================

def main():
    mp.set_start_method("spawn", force=True)
    rng = np.random.default_rng(42)

    results = load_checkpoint()
    n_cached = len(results)
    print(f"=== Ridge BLO hyperparameter tuning ===")
    print(f"CPUs: {os.cpu_count()}, workers: {N_JOBS}")
    print(f"N_TRIALS={N_TRIALS}, TUNE_N={TUNE_N}, TUNE_ITERS={TUNE_ITERS}")
    print(f"κ values: {KAPPA_VALUES}")
    print(f"Methods:  {METHODS_RUN}")
    print(f"Loaded checkpoint: {n_cached} entries already done.\n")

    t_total = time.time()
    ctx = mp.get_context("spawn")

    with ctx.Pool(N_JOBS) as pool:
        for kappa in KAPPA_VALUES:
            for method in METHODS_RUN:
                key = f"{kappa}_{method}"
                if key in results:
                    print(f"  skip  κ={kappa:>6g}  {method:<12}  "
                          f"(cached AUC={results[key][0]:.4f})")
                    continue

                t0 = time.time()
                # Sample configs
                configs = [sample_config(method, rng) for _ in range(N_TRIALS)]
                jobs    = [(method, cfg, kappa) for cfg in configs]

                try:
                    aucs_list = pool.map(_eval_worker, jobs, chunksize=4)
                except Exception as exc:
                    print(f"  ERROR  κ={kappa:>6g}  {method:<12}  pool failed: {exc}")
                    traceback.print_exc()
                    # Fall back to serial
                    aucs_list = []
                    for j in jobs:
                        try:
                            aucs_list.append(_eval_worker(j))
                        except Exception as e2:
                            aucs_list.append(1e6)

                best_i  = int(np.argmin(aucs_list))
                best_auc = aucs_list[best_i]
                best_cfg = configs[best_i]
                results[key] = (float(best_auc), best_cfg)
                save_checkpoint(results)
                print(f"  done  κ={kappa:>6g}  {method:<12}  "
                      f"AUC={best_auc:.4f}  ({time.time()-t0:.0f}s)")

    total_s = time.time() - t_total
    print(f"\nTotal tuning time: {total_s:.0f}s  ({total_s/60:.1f} min)\n")

    # ── collate ───────────────────────────────────────────────────────────────
    all_hparams = {}
    aucs = {}
    for kappa in KAPPA_VALUES:
        all_hparams[kappa] = {}
        for method in METHODS_RUN:
            key = f"{kappa}_{method}"
            if key in results:
                aucs[key] = results[key][0]
                all_hparams[kappa][method] = results[key][1]

    with open(RESULTS_FILE, "w") as f:
        json.dump({"aucs": aucs, "hparams": {str(k): v for k, v in all_hparams.items()}},
                  f, indent=2, default=str)
    print(f"Results saved: {RESULTS_FILE}")

    # ── AUC summary table ────────────────────────────────────────────────────
    print("\n=== Best AUC per (κ, method) ===")
    done_kappa = [k for k in KAPPA_VALUES if any(f"{k}_{m}" in results for m in METHODS_RUN)]
    hdr = f"{'method':<14}" + "".join(f"  κ={k:>6g}" for k in done_kappa)
    print(hdr); print("-" * len(hdr))
    for method in METHODS_RUN:
        row = f"{method:<14}"
        for kappa in done_kappa:
            key = f"{kappa}_{method}"
            v = results[key][0] if key in results else float("nan")
            row += f"  {v:>8.4f}"
        print(row)

    # ── patch notebook ────────────────────────────────────────────────────────
    print(f"\nPatching {NB_PATH} ...")
    try:
        patch_notebook(all_hparams, aucs)
    except Exception as exc:
        print(f"  WARNING: notebook patch failed: {exc}")
        traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
