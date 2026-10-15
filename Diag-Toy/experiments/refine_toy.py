"""Coordinate-descent refinement on top of tune_toy.py's random search.

Applied identically to every method: start from that method's best configuration,
sweep each hyperparameter over its (extended) list with the others held fixed,
keep any improvement, repeat for a few passes. This removes the random-search
variance that otherwise decides which method 'wins'.

Run:  python experiments/refine_toy.py
Updates results/toy_tuned_params.json in place (adds a "refined" entry).
"""
import csv
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tune_toy import FINAL_ITERS, GRIDS, evaluate

PASSES = 3

# extra values the random search never had, added for every method
EXTRA = {
    "PABLO": {"lr": [0.7, 0.2, 0.15, 0.07, 0.02, 0.007, 0.001],
              "eps": [1e-10, 1e-7, 1e-5, 3e-2, 0.3],
              "alpha": [0.95, 0.995, 0.9995, 0.999995],
              "tau": [0.05, 0.5, 1.0, 5.0, 20.0, 50.0],
              "y_rate": [0.02, 0.1, 0.3, 0.7],
              "grad_clip": [0.3, 0.5, 2.0, 5.0, 30.0, 300.0],
              "rho": [3e-4, 3e-3, 0.03, 0.3, 3.0],
              "lr_lambda": [3e-3, 3e-2, 0.3]},
    "BOME": {"lr": [0.7, 0.2, 0.15, 0.07, 0.02, 0.007, 0.001],
             "inner_lr": [0.5, 0.2, 0.05, 0.02, 0.005, 0.001],
             "inner_steps": [2, 3, 30, 100],
             "eta": [0.02, 0.03, 0.2, 0.3, 0.7, 5.0]},
    "TSP": {"gamma": [0.3, 0.7, 1.5, 3.0, 5.0],
            "eta": [0.02, 0.03, 0.07, 0.15, 0.3, 0.5],
            "step": [0.7, 0.2, 0.15, 0.05, 0.02, 0.007],
            "p": [3e-4, 3e-3, 0.03, 0.3, 3.0],
            "dual_tau": [0.05, 0.2, 1.0, 2.0, 10.0, 50.0],
            "mu": [0.05, 0.2, 0.7, 2.0],
            "theta": [0.05, 0.2, 0.7, 2.0],
            "omega": [0.02, 0.1, 0.3, 0.7],
            "lambda_init": [0.0, 0.1, 2.0, 5.0]},
    "STABLE": {"lr_w": [0.7, 0.2, 0.05, 0.02, 0.007, 0.001],
               "lr_theta": [0.7, 0.2, 0.05, 0.02, 0.007, 0.001],
               "tau": [0.2, 0.4, 0.8, 0.95],
               "mu_g": [1e-5, 1e-4, 3e-3, 0.02, 0.1, 0.5, 2.0],
               "C_gxy": [1.0, 3.0, 100.0, 1e4, 1e6]},
}


def values_for(method, key):
    vals = list(GRIDS[method][key]) + list(EXTRA.get(method, {}).get(key, []))
    out, seen = [], set()
    for v in vals:
        if str(v) not in seen:
            seen.add(str(v))
            out.append(v)
    return out


def _worker(args):
    method, cfg = args
    score, hit, _ = evaluate(method, cfg, FINAL_ITERS)
    return score, hit, cfg


def refine(method, cfg, pool):
    best_score, best_hit, _ = evaluate(method, cfg, FINAL_ITERS)
    best_cfg = dict(cfg)
    print(f"  {method}: start {best_score:.3e} (iters to 1e-6 = {best_hit})")
    for p in range(PASSES):
        improved = False
        for key in GRIDS[method]:
            cands = []
            for v in values_for(method, key):
                if str(v) == str(best_cfg[key]):
                    continue
                trial = dict(best_cfg)
                trial[key] = v
                cands.append((method, trial))
            if not cands:
                continue
            for score, hit, trial in pool.map(_worker, cands, chunksize=2):
                key_now = (best_score, best_hit if best_hit is not None else 10 ** 9)
                key_new = (score, hit if hit is not None else 10 ** 9)
                if key_new < key_now:
                    best_score, best_hit, best_cfg = score, hit, trial
                    improved = True
        print(f"    pass {p+1}: {best_score:.3e} (iters to 1e-6 = {best_hit})")
        if not improved:
            break
    return best_score, best_hit, best_cfg


def main():
    with open("results/toy_tuned_params.json") as fh:
        tuned = json.load(fh)

    # multi-start: the top-N distinct configurations from the random-search log,
    # so coordinate descent is not trapped in whichever basin it happened to land in
    log = {}
    with open("results/toy_tuning_log.csv") as fh:
        for row in csv.DictReader(fh):
            try:
                sc = float(row["score"])
            except ValueError:
                continue
            if not np.isfinite(sc):
                continue
            log.setdefault(row["method"], []).append((sc, json.loads(row["config"])))
    for m in log:
        log[m].sort(key=lambda r: r[0])

    N_STARTS = 5
    refined = {}
    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for method, entry in tuned["best"].items():
            starts = [entry["accurate"]["config"]]
            if entry.get("fast"):
                starts.append(entry["fast"]["config"])
            for _, cfg in log.get(method, [])[:N_STARTS]:
                if cfg not in starts:
                    starts.append(cfg)
            runs = [refine(method, s, pool) for s in starts]
            runs.sort(key=lambda r: (r[0], r[1] if r[1] is not None else 10 ** 9))
            score, hit, cfg = runs[0]
            # also keep whichever refined run is fastest while still <= 1e-8
            fast = sorted([r for r in runs if r[0] <= 1e-8 and r[1] is not None],
                          key=lambda r: (r[1], r[0]))
            refined[method] = {
                "score": score, "iters_to_1e-6": hit, "config": cfg,
                "fast": ({"score": fast[0][0], "iters_to_1e-6": fast[0][1],
                          "config": fast[0][2]} if fast else None),
            }

    tuned["refined"] = refined
    with open("results/toy_tuned_params.json", "w") as fh:
        json.dump(tuned, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"{'method':<8} {'median |w-w*|':>16} {'iters to 1e-6':>15}")
    for m, r in refined.items():
        print(f"{m:<8} {r['score']:>16.3e} {str(r['iters_to_1e-6']):>15}")
    print("=" * 78)
    for m, r in refined.items():
        print(f"{m}: {json.dumps(r['config'], default=str)}")


if __name__ == "__main__":
    main()
