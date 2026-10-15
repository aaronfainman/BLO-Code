"""Toy bilevel comparison at conditioning c, each method at its own tuned setting.

    F(w, theta) = (w+1)^4 - 2(w+1)^2 + (1/c)(theta - theta^2)^2 + 1
    J(w, theta) = (w-1)^4 - 2(w-1)^2 + (1/(2c)) theta^2 + 1

argmin_w J = {0, 2} for every theta, and F(0,.) = -1 against F(2,.) = 63, so the
bilevel solution is w* = 0.

Hyperparameters come from results/toy_tuned_params.json, produced by
experiments/tune_toy.py (random search) and experiments/refine_toy.py
(coordinate descent). Nothing here is hand-set.

Run:  python experiments/toy_comparison.py
Writes results/toy_comparison.{png,csv} and results/toy_comparison_params.txt
"""
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tune_toy import (C, GRIDS, LAMBDA_BOUND, N_SAMPLES, SEARCH_ITERS, THETA_INIT,
                      W_INIT, W_STAR, ToyLower, ToyUpper, build, toy_lower_loss,
                      toy_upper_loss)

N_ITERS = 2000
ORDER = ["PABLO", "BOME", "TSP", "STABLE"]
# reference palette, fixed slot order (blue, orange, aqua, yellow)
COLOURS = {"PABLO": "#2a78d6", "BOME": "#eb6834", "TSP": "#1baf7a", "STABLE": "#eda100"}


def grad_w_J(w):
    """d/dw of the lower level; independent of theta."""
    return np.abs(4 * (w - 1) ** 3 - 4 * (w - 1))


def run(method, cfg):
    torch.manual_seed(0)
    ll, ul = ToyLower(), ToyUpper()
    opt = build(method, cfg, ll, ul)
    pass_seed = method in ("BOME", "TSP")
    hist = {"w": [], "theta": [], "ll_loss": [], "ul_loss": []}
    for i in range(N_ITERS):
        stats = opt.step(None, seed=i) if pass_seed else opt.step(None)
        hist["w"].append(ll.omega.item())
        hist["theta"].append(ul.theta.item())
        hist["ll_loss"].append(stats["ll_loss"])
        hist["ul_loss"].append(stats["ul_loss"])
    return hist


def _place_labels(ax, entries):
    """Greedy de-collision of end-of-line labels in log space."""
    lo, hi = np.log10(ax.get_ylim()[0]), np.log10(ax.get_ylim()[1])
    gap = (hi - lo) * 0.055
    placed, prev = [], -np.inf
    for name, y, colour in sorted(entries, key=lambda e: e[1]):
        ly = max(np.log10(max(y, 10 ** lo)), prev + gap)
        placed.append((name, 10 ** ly, colour))
        prev = ly
    return placed


def plot(hists, path="results/toy_comparison.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.grid(True, which="major", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")

    ends = [[], []]
    for name in ORDER:
        w = np.array(hists[name]["w"])
        it = np.arange(1, len(w) + 1)
        series = (np.maximum(np.abs(w - W_STAR), 1e-13),
                  np.maximum(grad_w_J(w), 1e-13))
        for ax, s, bucket in zip(axes, series, ends):
            ax.plot(it, s, color=COLOURS[name], linewidth=2, label=name)
            bucket.append((name, s[-1], COLOURS[name]))

    axes[0].set_title(r"distance to the bilevel solution  $|w - w^*|$", fontsize=10)
    axes[1].set_title(r"lower-level stationarity  $|\nabla_w J(w,\theta)|$", fontsize=10)
    for ax, bucket in zip(axes, ends):
        ax.set_xlim(1, N_ITERS * 1.34)
        for label, y, colour in _place_labels(ax, bucket):
            ax.annotate(label, xy=(N_ITERS * 1.04, y), fontsize=7.5,
                        color="#3d3d38", va="center")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
    fig.suptitle(f"Toy bilevel problem, conditioning $c={C:g}$ — each method at its own "
                 f"tuned setting, no LR schedule", fontsize=11, color="#3d3d38")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def write_params(tuned, hists, path="results/toy_comparison_params.txt"):
    L = []
    A = L.append
    A("=" * 78)
    A("Toy bilevel comparison - exact parameters to reproduce results/toy_comparison.png")
    A("=" * 78)
    A("")
    A("PROBLEM")
    A("  F(w,theta) = (w+1)^4 - 2(w+1)^2 + (1/c)(theta - theta^2)^2 + 1     [upper]")
    A("  J(w,theta) = (w-1)^4 - 2(w-1)^2 + (1/(2c)) theta^2 + 1             [lower]")
    A(f"  c            = {C:g}")
    A(f"  w_0          = {W_INIT}")
    A(f"  theta_0      = {THETA_INIT}")
    A(f"  w*           = {W_STAR}   (argmin_w J = {{0, 2}}; F(0,.) = -1 vs F(2,.) = 63)")
    A(f"  iterations   = {N_ITERS}, deterministic (stochastic=False), float32, CPU")
    A(f"  torch seed   = 0 before each run")
    A("")
    A("  NOTE: dF/dtheta = (2/c)(theta - theta^2)(1 - 2theta) is exactly 0 at theta_0 = 0.5,")
    A("  and d2J/dw dtheta = 0, so every method's theta-direction vanishes. theta does not")
    A("  move in any run; the comparison is effectively one-dimensional in w.")
    A("")
    A("TUNING PROTOCOL")
    A("  Each method was tuned separately, on this problem, against the same objective:")
    A("    score = median |w - w*| over the last 10% of iterations (median, not the final")
    A("            value, so an oscillating method is not flattered by where it stops)")
    A("    tie-break = first iteration at which |w - w*| <= 1e-6")
    A(f"  Stage 1: random search, {SEARCH_ITERS} iterations per configuration")
    for m in ORDER:
        A(f"           {m:<8} {N_SAMPLES[m]:>5} configurations sampled")
    A("  Stage 2: coordinate descent from the best configurations, re-scored over")
    A(f"           {N_ITERS} iterations (experiments/refine_toy.py)")
    A("  Grids searched are in experiments/tune_toy.py (GRIDS) and")
    A("  experiments/refine_toy.py (EXTRA). Full log: results/toy_tuning_log.csv")
    A("")
    A("RESULTS")
    A(f"  {'method':<8} {'median |w-w*|':>16} {'iters to 1e-6':>15} {'final |w-w*|':>15}")
    for m in ORDER:
        r = tuned["refined"][m]
        final = abs(hists[m]["w"][-1] - W_STAR)
        A(f"  {m:<8} {r['score']:>16.3e} {str(r['iters_to_1e-6']):>15} {final:>15.3e}")
    A("")
    A("TUNED HYPERPARAMETERS")
    A("")
    A("  PABLO  -- BLO_optimiser_PABLO_Prox_Bound (Algorithm 1: proximal + projection)")
    c = tuned["refined"]["PABLO"]["config"]
    A(f"    w_optimiser = theta_optimiser = torch.optim.RMSprop")
    A(f"      lr    (mu)      = {c['lr']}")
    A(f"      alpha (beta)    = {c['alpha']}")
    A(f"      eps             = {c['eps']}")
    A(f"    rho               = {c['rho']}")
    A(f"    lr_lambda (eta)   = {c['lr_lambda']}")
    A(f"    tau               = {c['tau']}          (prox coefficient; loss term is (tau/2)||Z-y||^2)")
    A(f"    gamma             = {c['y_rate'] / c['tau']:g}          (y <- y + gamma*tau (Z - y); gamma*tau = {c['y_rate']})")
    A(f"    lambda_bound (R)  = {LAMBDA_BOUND}")
    A(f"    grad_clip         = {c['grad_clip']}          (clips ||grad_Z K||, joint over W and theta)")
    A(f"    total_steps       = None         (NO decaying step-size schedule)")
    A(f"    lambda_init       = None -> u_0 = 0;  y_init = None -> y_0 = Z_0")
    A("")
    A(f"    WARNING: lr_lambda = {c['lr_lambda']}. The tuner drove the multiplier step to zero, so u stays")
    A("    at 0 and K = F + (rho/2)||grad_W J||^2 + (tau/2)||Z-y||^2. That is a quadratic")
    A("    PENALTY method, not an augmented Lagrangian. With lr_lambda > 0 the best")
    A("    achievable score on this problem was ~1.0e-08, about two orders of magnitude")
    A("    worse, and neither eps, alpha nor grad_clip recovered it (all within 40% of")
    A("    each other). See ALGORITHM_REVIEW.md.")
    A("")
    A("  BOME  -- BLO_optimiser_BOME")
    c = tuned["refined"]["BOME"]["config"]
    A(f"    lr_w = lr_theta   = {c['lr']}")
    A(f"    w_optimiser = theta_optimiser = torch.optim.{ {'sgd':'SGD','sgdm':'SGD(momentum=0.9)','adam':'Adam','rmsprop':'RMSprop'}[c['opt']] }")
    A(f"    inner_steps (T)   = {c['inner_steps']}")
    A(f"    inner_lr          = {c['inner_lr']}")
    A(f"    eta               = {c['eta']}")
    A(f"    barrier           = 'grad_norm'  (default; phi = eta ||grad q||^2)")
    A("")
    A("  TSP  -- BLO_optimiser_TSP_Paper")
    c = tuned["refined"]["TSP"]["config"]
    A(f"    gamma  (Moreau)   = {c['gamma']}")
    A(f"    delta             = 0.0")
    A(f"    p      (anchor)   = {c['p']}")
    A(f"    tau    (dual step)= {c['dual_tau']}")
    A(f"    mu     (lambda EMA)= {c['mu']}")
    A(f"    theta  (h EMA)    = {c['theta']}")
    A(f"    eta    (z step)   = {c['eta']}")
    A(f"    alpha = beta      = {c['step']}")
    A(f"    omega  (anchor EMA)= {c['omega']}")
    A(f"    lambda_init       = {c['lambda_init']}")
    A("")
    A("  STABLE  -- BLO_optimiser_STABLE")
    c = tuned["refined"]["STABLE"]["config"]
    A(f"    lr_w   (beta)     = {c['lr_w']}")
    A(f"    lr_theta (alpha)  = {c['lr_theta']}")
    A(f"    tau               = {c['tau']}")
    A(f"    mu_g              = {c['mu_g']}          (H_yy projected onto {{X >= mu_g I}})")
    A(f"    C_gxy             = {c['C_gxy']}          (H_xy projected onto {{||X|| <= C_gxy}})")
    A(f"    x_projection      = None")
    A("")
    A("REPRODUCE")
    A("  python experiments/tune_toy.py       # random search   -> results/toy_tuned_params.json")
    A("  python experiments/refine_toy.py     # coordinate desc -> adds 'refined' to that file")
    A("  python experiments/toy_comparison.py # -> this file, the plot and the csv")
    A("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    print(f"wrote {path}")


def main():
    with open("results/toy_tuned_params.json") as fh:
        tuned = json.load(fh)

    print(f"Toy problem, c = {C:g}, {N_ITERS} deterministic iterations, "
          f"w_0 = {W_INIT}, theta_0 = {THETA_INIT}, w* = {W_STAR}\n")
    hists = {}
    for m in ORDER:
        cfg = tuned["refined"][m]["config"]
        hists[m] = run(m, cfg)
        h = hists[m]
        print(f"  {m:<8} w={h['w'][-1]:+.3e}  theta={h['theta'][-1]:.6f}  "
              f"F={h['ul_loss'][-1]:+.8f}  J={h['ll_loss'][-1]:+.8f}  "
              f"|w-w*|={abs(h['w'][-1] - W_STAR):.3e}")

    os.makedirs("results", exist_ok=True)
    with open("results/toy_comparison.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["algorithm", "iteration", "w", "theta", "ll_loss", "ul_loss"])
        for m in ORDER:
            h = hists[m]
            for i in range(N_ITERS):
                wr.writerow([m, i + 1, h["w"][i], h["theta"][i],
                             h["ll_loss"][i], h["ul_loss"][i]])
    print("\nwrote results/toy_comparison.csv")
    plot(hists)
    write_params(tuned, hists)


if __name__ == "__main__":
    main()
