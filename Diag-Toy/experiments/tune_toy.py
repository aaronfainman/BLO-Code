"""Per-method hyperparameter search on the conditioned toy bilevel problem.

    F(w, theta) = (w+1)^4 - 2(w+1)^2 + (1/c)(theta - theta^2)^2 + 1
    J(w, theta) = (w-1)^4 - 2(w-1)^2 + (1/(2c)) theta^2 + 1

argmin_w J = {0, 2} for every theta, and F(0,.) = -1 against F(2,.) = 63, so the
bilevel solution is w* = 0. c controls the conditioning.

Each method gets its OWN random search over its own grid; every method is then
reported at its best setting. Score = median |w - w*| over the last 10% of
iterations (median rather than the final value so a method that oscillates is not
flattered by where it happens to stop).

PABLO runs with NO decaying step-size schedule. Instead the search covers the
knobs that address the small-gradient wander: the RMSProp smoothing beta
(`alpha`), the RMSProp epsilon, and a clamp on ||grad_Z K||.

Run:  python experiments/tune_toy.py
Writes results/toy_tuned_params.json and results/toy_tuning_log.csv
"""
import csv
import functools
import json
import multiprocessing as mp
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blo_optimisers import (BLO_optimiser_BOME, BLO_optimiser_PABLO_Prox_Bound,
                            BLO_optimiser_STABLE, BLO_optimiser_TSP_Paper)

C = 10.0                 # conditioning constant
W_STAR = 0.0
W_INIT, THETA_INIT = 0.5, 0.5
SEARCH_ITERS = 1000      # iterations used while searching
FINAL_ITERS = 2000       # iterations used to re-score the shortlist
LAMBDA_BOUND = 100.0     # radius R of the ball u is projected onto
N_SAMPLES = {"PABLO": 1400, "BOME": 500, "TSP": 800, "STABLE": 500}


class ToyLower(nn.Module):
    def __init__(self):
        super().__init__()
        self.omega = nn.Parameter(torch.tensor(W_INIT))


class ToyUpper(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(THETA_INIT))


def toy_lower_loss(ll_model, data, ul_model, stochastic, seed, **kwargs):
    w, theta = ll_model.omega, ul_model.theta
    return (w - 1) ** 4 - 2 * (w - 1) ** 2 + theta ** 2 / (2 * C) + 1


def toy_upper_loss(ll_model, data, ul_model, stochastic, seed, **kwargs):
    w, theta = ll_model.omega, ul_model.theta
    return (w + 1) ** 4 - 2 * (w + 1) ** 2 + (theta - theta ** 2) ** 2 / C + 1


SGD_MOMENTUM = functools.partial(torch.optim.SGD, momentum=0.9)
OPTIMISERS = {"sgd": torch.optim.SGD, "sgdm": SGD_MOMENTUM,
              "adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop}

GRIDS = {
    "PABLO": {
        "lr":         [1.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.003],
        "eps":        [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1],
        "alpha":      [0.9, 0.99, 0.999, 0.9999, 0.99999],   # the paper's beta
        "rho":        [0.001, 0.01, 0.1, 1.0],
        "lr_lambda":  [0.0, 1e-4, 1e-3, 1e-2, 1e-1],
        "tau":        [0.002, 0.02, 0.2, 2.0, 10.0],
        "y_rate":     [0.01, 0.05, 0.2, 0.5, 1.0],           # = gamma * tau
        "grad_clip":  [None, 1.0, 3.0, 10.0, 100.0],
    },
    "BOME": {
        "lr":          [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.003],
        "inner_lr":    [0.3, 0.1, 0.03, 0.01, 0.003],
        "inner_steps": [5, 10, 20, 50],
        "eta":         [0.05, 0.1, 0.5, 1.0, 2.0],
        "opt":         ["sgd", "sgdm", "adam", "rmsprop"],
    },
    "TSP": {
        "gamma":       [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        "eta":         [0.005, 0.01, 0.05, 0.1, 0.2],
        "step":        [0.3, 0.1, 0.03, 0.01, 0.003],        # alpha = beta
        "p":           [0.0, 1e-3, 1e-2, 1e-1, 1.0],
        "dual_tau":    [0.005, 0.05, 0.5, 5.0],
        "mu":          [0.1, 0.5, 1.0],
        "theta":       [0.1, 0.5, 1.0],
        "omega":       [0.05, 0.2, 0.5],
        "lambda_init": [0.5, 1.0],
    },
    "STABLE": {
        "lr_w":     [0.5, 0.3, 0.1, 0.03, 0.01, 0.003],
        "lr_theta": [0.5, 0.3, 0.1, 0.03, 0.01, 0.003],
        "tau":      [0.05, 0.1, 0.3, 0.6, 0.9, 1.0],
        "mu_g":     [1e-3, 1e-2, 0.05, 0.2, 1.0],
        "C_gxy":    [10.0, 50.0, 1000.0],
    },
}


def build(method, cfg, ll, ul):
    common = dict(lower_loss_fn=toy_lower_loss, upper_loss_fn=toy_upper_loss)
    if method == "PABLO":
        opt_kwargs = {"alpha": cfg["alpha"], "eps": cfg["eps"]}
        return BLO_optimiser_PABLO_Prox_Bound(
            ll_model=ll, ul_model=ul, **common,
            w_optimiser=torch.optim.RMSprop, theta_optimiser=torch.optim.RMSprop,
            w_optimiser_kwargs=opt_kwargs, theta_optimiser_kwargs=opt_kwargs,
            lr_w=cfg["lr"], lr_theta=cfg["lr"], lr_lambda=cfg["lr_lambda"], rho=cfg["rho"],
            tau=cfg["tau"], gamma=cfg["y_rate"] / cfg["tau"],
            lambda_bound=LAMBDA_BOUND, grad_clip=cfg["grad_clip"])   # no schedule
    if method == "BOME":
        o = OPTIMISERS[cfg["opt"]]
        return BLO_optimiser_BOME(ll_model=ll, ul_model=ul, **common,
                                  inner_steps=cfg["inner_steps"], inner_lr=cfg["inner_lr"],
                                  eta=cfg["eta"], lr_w=cfg["lr"], lr_theta=cfg["lr"],
                                  w_optimiser=o, theta_optimiser=o)
    if method == "TSP":
        return BLO_optimiser_TSP_Paper(ll_model=ll, ul_model=ul, **common,
                                       gamma=cfg["gamma"], delta=0.0, p=cfg["p"],
                                       tau=cfg["dual_tau"], mu=cfg["mu"], theta=cfg["theta"],
                                       eta=cfg["eta"], alpha=cfg["step"], beta=cfg["step"],
                                       omega=cfg["omega"], lambda_init=cfg["lambda_init"])
    if method == "STABLE":
        return BLO_optimiser_STABLE(ll_model=ll, ul_model=ul, **common,
                                    lr_w=cfg["lr_w"], lr_theta=cfg["lr_theta"],
                                    tau=cfg["tau"], mu_g=cfg["mu_g"], C_gxy=cfg["C_gxy"])
    raise ValueError(method)


def evaluate(method, cfg, n_iters, want_trace=False):
    """Returns (score, iters_to_1e-6, trace or None). score = inf on divergence."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    ll, ul = ToyLower(), ToyUpper()
    pass_seed = method in ("BOME", "TSP")
    try:
        opt = build(method, cfg, ll, ul)
    except Exception:
        return float("inf"), None, None

    errs = []
    hit = None
    for i in range(n_iters):
        try:
            opt.step(None, seed=i) if pass_seed else opt.step(None)
        except Exception:
            return float("inf"), None, None
        w = ll.omega.item()
        if not np.isfinite(w):
            return float("inf"), None, None
        e = abs(w - W_STAR)
        errs.append(e)
        if hit is None and e <= 1e-6:
            hit = i + 1
    tail = errs[int(0.9 * len(errs)):]
    score = float(np.median(tail))
    if not np.isfinite(score):
        return float("inf"), None, None
    return score, hit, (errs if want_trace else None)


def _worker(args):
    method, cfg, n_iters = args
    score, hit, _ = evaluate(method, cfg, n_iters)
    return method, cfg, score, hit


def sample_configs(method, n, rng):
    grid = GRIDS[method]
    keys = list(grid)
    seen, out = set(), []
    full = int(np.prod([len(grid[k]) for k in keys]))
    for _ in range(min(n, full) * 40):
        if len(out) >= min(n, full):
            break
        cfg = {k: rng.choice(grid[k]) for k in keys}
        sig = tuple(cfg[k] for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cfg)
    return out


def main():
    rng = random.Random(0)
    os.makedirs("results", exist_ok=True)

    jobs = []
    for method, n in N_SAMPLES.items():
        for cfg in sample_configs(method, n, rng):
            jobs.append((method, cfg, SEARCH_ITERS))
    print(f"searching {len(jobs)} configurations at c = {C} "
          f"({SEARCH_ITERS} iters each), then re-scoring the top 5 per method "
          f"at {FINAL_ITERS}\n")

    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        results = pool.map(_worker, jobs, chunksize=4)

    with open("results/toy_tuning_log.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["method", "score", "iters_to_1e-6", "config"])
        for method, cfg, score, hit in results:
            writer.writerow([method, score, hit, json.dumps(cfg, default=str)])

    best = {}
    for method in N_SAMPLES:
        mine = [r for r in results if r[0] == method]
        n_ok = sum(1 for r in mine if np.isfinite(r[2]))

        def rescore(cands):
            out = []
            for _, cfg, _, _ in cands:
                sc, ht, _ = evaluate(method, cfg, FINAL_ITERS)
                out.append((sc, ht, cfg))
            return out

        # most accurate
        acc = rescore(sorted(mine, key=lambda r: r[2])[:6])
        acc.sort(key=lambda r: (r[0], r[1] if r[1] is not None else 10 ** 9))

        # fastest to 1e-6, among configs that are also accurate to <= 1e-8
        fast_pool = [r for r in mine if r[3] is not None and r[2] <= 1e-8]
        fast = rescore(sorted(fast_pool, key=lambda r: r[3])[:6]) if fast_pool else []
        fast = [r for r in fast if r[1] is not None and r[0] <= 1e-8]
        fast.sort(key=lambda r: (r[1], r[0]))

        best[method] = {
            "accurate": {"score": acc[0][0], "iters_to_1e-6": acc[0][1], "config": acc[0][2]},
            "fast": ({"score": fast[0][0], "iters_to_1e-6": fast[0][1], "config": fast[0][2]}
                     if fast else None),
            "searched": len(mine), "finite": n_ok,
        }
        f = best[method]["fast"]
        print(f"{method:<8} accuracy-best  median |w-w*| = {acc[0][0]:.3e}  "
              f"iters to 1e-6 = {acc[0][1]}")
        print(f"         {json.dumps(acc[0][2], default=str)}")
        if f:
            print(f"         speed-best     median |w-w*| = {f['score']:.3e}  "
                  f"iters to 1e-6 = {f['iters_to_1e-6']}")
            print(f"         {json.dumps(f['config'], default=str)}")
        print(f"         ({n_ok}/{len(mine)} configs did not diverge)\n")

    with open("results/toy_tuned_params.json", "w") as fh:
        json.dump({"c": C, "w_star": W_STAR, "search_iters": SEARCH_ITERS,
                   "final_iters": FINAL_ITERS, "best": best}, fh, indent=2, default=str)
    print("wrote results/toy_tuned_params.json and results/toy_tuning_log.csv")


if __name__ == "__main__":
    main()
