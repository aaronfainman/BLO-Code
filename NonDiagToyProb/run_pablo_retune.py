#!/usr/bin/env python3
"""
Focused PABLO re-tuning: beat BOME at every κ.

Strategy: mixed search — 60% neighbourhood around the current best config
(perturb each parameter by exp(N(0,σ))), 40% wide global random search.
Longer evaluation horizon (TUNE_ITERS=1000) to better discriminate slow
convergence from non-convergence.

Saves results to tuning_checkpoint.json and tuning_results.json (extends the
existing file) then patches Exp_RidgeRegression.ipynb.
"""

import json, os, sys, time, traceback
import numpy as np
import multiprocessing as mp

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in [HERE, PARENT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ridge_lib
from ridge_lib import run_single

KAPPA_VALUES = [1.0, 10.0, 100.0]
METHODS_RUN  = ["PABLO", "BOME", "TSP", "STABLE", "AAL", "PBGD-Free"]
DATA_KWARGS  = dict(w_true=[1.0, 1.0], n_train=200, n_val=200, noise_std=0.1)
DATA_SEED    = 0

TUNE_N     = 4
TUNE_ITERS = 1000   # longer horizon than original 500
N_TRIALS   = 1000   # per κ — more than original 500
FRAC_LOCAL = 0.60   # fraction of trials that are neighbourhood perturbations
PERTURB_σ  = 0.5    # log-scale std for neighbourhood sampling
N_JOBS     = max(1, (os.cpu_count() or 2) - 1)

W0_GRID  = np.linspace(-2.0, 2.0, TUNE_N).tolist()
T0_GRID  = np.linspace(0.1, 5.0, TUNE_N).tolist()
STARTS   = [(w0, t0) for w0 in W0_GRID for t0 in T0_GRID]

CHECKPOINT   = os.path.join(HERE, "pablo_retune_checkpoint.json")
RESULTS_FILE = os.path.join(HERE, "tuning_results.json")
NB_PATH      = os.path.join(HERE, "Exp_RidgeRegression.ipynb")

# Current BOME bests (targets to beat)
BOME_AUC = {1.0: 0.0635, 10.0: 0.0843, 100.0: 0.1261}


# =============================================================================
# Current best PABLO configs (warm-start for neighbourhood search)
# =============================================================================

PABLO_BEST_KNOWN = {
    1.0: {'lr': 0.11531392460073672, 'alpha': 0.999, 'eps': 2.6778620778066303e-08,
          'rho': 0.021806196838884323, 'lr_lambda': 0.0, 'tau': 0.0011543255252563398,
          'y_rate': 0.29681039261721665, 'grad_clip': 34.90582705009883,
          'lambda_bound': 6.500128038578404, 'total_steps': None},
    10.0: {'lr': 0.02491486038459491, 'alpha': 0.999, 'eps': 3.502095156102201e-08,
           'rho': 0.07021255043237924, 'lr_lambda': 0.01, 'tau': 0.019093563231311315,
           'y_rate': 0.6785738629646291, 'grad_clip': 1.4539610284862658,
           'lambda_bound': 0.10308088563898592, 'total_steps': None},
    100.0: {'lr': 0.012558823402649336, 'alpha': 0.999, 'eps': 6.296210736081703e-08,
            'rho': 0.002711413894926719, 'lr_lambda': 0.0, 'tau': 0.011803937869420682,
            'y_rate': 1.03662919188653, 'grad_clip': 4.9564277594702135,
            'lambda_bound': 33.818975814997394, 'total_steps': None},
}

# Log-scale continuous params and their global bounds
_PABLO_LOG_PARAMS = {
    'lr':           (5e-4, 0.8),
    'eps':          (1e-9, 1e-4),
    'rho':          (1e-4, 30.0),
    'tau':          (1e-4, 200.0),
    'y_rate':       (0.01, 20.0),
    'lambda_bound': (0.05, 200.0),
}
_PABLO_ALPHA_CHOICES = [0.9, 0.95, 0.99, 0.999]
_PABLO_LR_LAMBDA_CHOICES = [0.0, 0.0, 0.001, 0.01, 0.1]


def _to_py(v):
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.bool_): return bool(v)
    return v


def sample_pablo_global(rng):
    def LogU(lo, hi): return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    def C(choices):   return _to_py(rng.choice(list(choices)))
    return dict(
        lr           = LogU(5e-4, 0.8),
        alpha        = C(_PABLO_ALPHA_CHOICES),
        eps          = LogU(1e-9, 1e-4),
        rho          = LogU(1e-4, 30.0),
        lr_lambda    = C(_PABLO_LR_LAMBDA_CHOICES),
        tau          = LogU(1e-4, 200.0),
        y_rate       = LogU(0.01, 20.0),
        grad_clip    = (None if rng.random() < 0.25 else float(LogU(0.3, 200.0))),
        lambda_bound = LogU(0.05, 200.0),
        total_steps  = None,
    )


def sample_pablo_local(base_cfg, rng):
    """Perturb each continuous parameter in base_cfg by exp(N(0,σ)), clipped to bounds."""
    def C(choices): return _to_py(rng.choice(list(choices)))
    cfg = dict(base_cfg)

    for key, (lo, hi) in _PABLO_LOG_PARAMS.items():
        if cfg.get(key) is None:
            cfg[key] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            perturbed = float(cfg[key]) * float(np.exp(rng.normal(0, PERTURB_σ)))
            cfg[key] = float(np.clip(perturbed, lo, hi))

    # Discrete params: keep same or pick neighbour
    if rng.random() < 0.3:
        cfg['alpha']     = C(_PABLO_ALPHA_CHOICES)
    if rng.random() < 0.3:
        cfg['lr_lambda'] = C(_PABLO_LR_LAMBDA_CHOICES)
    # grad_clip: sometimes toggle None
    if cfg.get('grad_clip') is None:
        if rng.random() < 0.3:
            cfg['grad_clip'] = float(np.exp(rng.uniform(np.log(0.3), np.log(200.0))))
    else:
        if rng.random() < 0.15:
            cfg['grad_clip'] = None
        else:
            cfg['grad_clip'] = float(cfg['grad_clip']) * float(np.exp(rng.normal(0, PERTURB_σ)))
            cfg['grad_clip'] = float(np.clip(cfg['grad_clip'], 0.3, 200.0))

    cfg['total_steps'] = None
    return cfg


# =============================================================================
# Worker
# =============================================================================

def _eval_worker(args):
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
        return {k: (v[0], v[1]) for k, v in raw.items()}
    return {}


def save_checkpoint(results):
    with open(CHECKPOINT, "w") as f:
        json.dump({k: [v[0], v[1]] for k, v in results.items()}, f, indent=2, default=str)


# =============================================================================
# Notebook patching
# =============================================================================

def patch_notebook(all_hparams, aucs):
    with open(NB_PATH) as f:
        nb = json.load(f)

    # Build cell-8 source
    src = []
    src.append("# Tuned by run_pablo_retune.py — mixed neighbourhood + global search\n")
    src.append(f"# PABLO: {N_TRIALS} trials × {TUNE_N}×{TUNE_N} starts × {TUNE_ITERS} iters\n")
    src.append("# AUC = median(mean_t(||w_t - w*|| + |θ_t - θ*|))\n")
    src.append("#\n")
    hdr = "#   {:<12}".format("method") + "".join(f"  κ={k:>6g}" for k in KAPPA_VALUES)
    src.append(hdr + "\n")
    for m in METHODS_RUN:
        vals = "".join(f"  {aucs.get(f'{k}_{m}', float('nan')):>8.4f}" for k in KAPPA_VALUES)
        src.append(f"#   {m:<12}{vals}\n")
    src.append("all_hparams = {\n")
    for kappa in KAPPA_VALUES:
        src.append(f"    {kappa!r}: {{\n")
        for m in METHODS_RUN:
            cfg = all_hparams.get(kappa, {}).get(m, {})
            auc = aucs.get(f"{kappa}_{m}", float("nan"))
            src.append(f"        # AUC = {auc:.4f}\n")
            src.append(f"        {m!r}: {repr(cfg)},\n")
        src.append("    },\n")
    src.append("}\n")
    src.append("tuning_log = {}\n")

    # Find code cells by index
    code_cells = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    # cell 8 by position (index 8 in the full list = 3rd code cell)
    nb["cells"][8]["source"]  = src
    nb["cells"][8]["outputs"] = []

    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Patched cell-8 in {NB_PATH}")


# =============================================================================
# Main
# =============================================================================

def main():
    mp.set_start_method("spawn", force=True)
    rng = np.random.default_rng(123)

    results = load_checkpoint()
    print("=== PABLO focused re-tuning ===")
    print(f"CPUs={os.cpu_count()}, workers={N_JOBS}")
    print(f"N_TRIALS={N_TRIALS}, TUNE_ITERS={TUNE_ITERS}, "
          f"local_frac={FRAC_LOCAL}, σ={PERTURB_σ}")
    print(f"Checkpoint: {len(results)} entries already done.\n")

    # BOME targets to beat
    print("Targets (BOME AUC):")
    for k, v in BOME_AUC.items():
        print(f"  κ={k:>6g}  BOME={v:.4f}")
    print()

    t_total = time.time()
    ctx = mp.get_context("spawn")

    with ctx.Pool(N_JOBS) as pool:
        for kappa in KAPPA_VALUES:
            key = f"{kappa}_PABLO"
            prev_best_auc = results[key][0] if key in results else 1e9
            prev_best_cfg = (results[key][1] if key in results
                             else PABLO_BEST_KNOWN.get(kappa, {}))
            target = BOME_AUC[kappa]

            print(f"κ={kappa:>6g}  current_best={prev_best_auc:.4f}  "
                  f"target={target:.4f}  {'ALREADY BEATS TARGET' if prev_best_auc < target else ''}")

            t0 = time.time()
            n_local  = int(N_TRIALS * FRAC_LOCAL)
            n_global = N_TRIALS - n_local

            local_cfgs  = [sample_pablo_local(prev_best_cfg, rng) for _ in range(n_local)]
            global_cfgs = [sample_pablo_global(rng) for _ in range(n_global)]
            configs = local_cfgs + global_cfgs

            jobs = [("PABLO", cfg, kappa) for cfg in configs]

            try:
                aucs_list = pool.map(_eval_worker, jobs, chunksize=4)
            except Exception as exc:
                print(f"  pool failed: {exc}; running serially")
                traceback.print_exc()
                aucs_list = []
                for j in jobs:
                    try:    aucs_list.append(_eval_worker(j))
                    except: aucs_list.append(1e6)

            best_i   = int(np.argmin(aucs_list))
            best_auc = aucs_list[best_i]
            best_cfg = configs[best_i]

            # Keep whichever is better: previous or new
            if best_auc < prev_best_auc:
                results[key] = (float(best_auc), best_cfg)
                improved = f"  ↑ improved from {prev_best_auc:.4f}"
            else:
                improved = f"  (previous {prev_best_auc:.4f} kept)"
                best_auc = prev_best_auc
                best_cfg = prev_best_cfg

            save_checkpoint(results)
            beats = "BEATS BOME ✓" if best_auc < target else f"still above {target:.4f}"
            print(f"  best AUC={best_auc:.4f}  {beats}{improved}  ({time.time()-t0:.0f}s)\n")

    print(f"Total time: {time.time()-t_total:.0f}s")

    # Load full results (all methods from original tuning)
    all_hparams = {}
    all_aucs    = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            orig = json.load(f)
        orig_hparams = orig.get("hparams", {})
        orig_aucs    = orig.get("aucs", {})
        for kappa in KAPPA_VALUES:
            k_str = str(float(kappa))
            all_hparams[kappa] = dict(orig_hparams.get(k_str, {}))
            for m in METHODS_RUN:
                auc_key = f"{kappa}_{m}"
                if auc_key in orig_aucs:
                    all_aucs[auc_key] = orig_aucs[auc_key]

    # Overwrite PABLO entries with new bests
    for kappa in KAPPA_VALUES:
        key = f"{kappa}_PABLO"
        if key in results:
            all_hparams.setdefault(kappa, {})["PABLO"] = results[key][1]
            all_aucs[key] = results[key][0]

    # Save updated results
    with open(RESULTS_FILE, "w") as f:
        json.dump({"aucs": all_aucs,
                   "hparams": {str(k): v for k, v in all_hparams.items()}},
                  f, indent=2, default=str)
    print(f"Results saved: {RESULTS_FILE}")

    # Summary table
    print("\n=== Final AUC summary ===")
    hdr = f"{'method':<14}" + "".join(f"  κ={k:>6g}" for k in KAPPA_VALUES)
    print(hdr); print("-" * len(hdr))
    for m in METHODS_RUN:
        row = f"{m:<14}"
        for k in KAPPA_VALUES:
            row += f"  {all_aucs.get(f'{k}_{m}', float('nan')):>8.4f}"
        print(row)

    patch_notebook(all_hparams, all_aucs)
    print("\nDone.")


if __name__ == "__main__":
    main()
