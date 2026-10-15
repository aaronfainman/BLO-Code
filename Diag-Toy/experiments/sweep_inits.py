"""Sweep initial conditions (w_0, theta_0) over a grid and average per algorithm.

Each method uses the configuration tuned at the single start w_0 = theta_0 = 0.5
(results/toy_tuned_params.json). The sweep therefore measures how well that tuned
setting generalises across the basin structure, not a per-start retune.

Solution structure of the toy problem at conditioning c:
    argmin_w J = {0, 2} for every theta, with a barrier at w = 1.
    F(0,.) = -1 against F(2,.) = 63, so the bilevel solution is w* = 0.
    Starts with w_0 > 1 begin in the WRONG lower-level basin: J pulls them to
    w = 2 while F pulls them to w = 0.
    theta* minimises (theta - theta^2)^2, so theta* in {0, 1}; the theta error is
    min(|theta|, |theta - 1|).

Run:  python experiments/sweep_inits.py
Writes results/sweep_inits.{png,csv} and appends to results/toy_comparison_params.txt
"""
import csv
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tune_toy
from tune_toy import C, W_STAR, build, toy_lower_loss, toy_upper_loss

GRID_LO, GRID_HI, GRID_N = -1.5, 1.5, 20
N_ITERS = 2000
TOL = 1e-6
ORDER = ["PABLO", "BOME", "TSP", "STABLE"]
COLOURS = {"PABLO": "#2a78d6", "BOME": "#eb6834", "TSP": "#1baf7a", "STABLE": "#eda100"}


def theta_error(theta):
    return np.minimum(np.abs(theta), np.abs(theta - 1.0))


def _worker(args):
    method, cfg, w0, t0 = args
    torch.set_num_threads(1)
    torch.manual_seed(0)
    ll, ul = tune_toy.ToyLower(), tune_toy.ToyUpper()
    with torch.no_grad():
        ll.omega.fill_(w0)
        ul.theta.fill_(t0)
    pass_seed = method in ("BOME", "TSP")
    try:
        opt = build(method, cfg, ll, ul)
    except Exception:
        return method, w0, t0, None, None
    ws = np.empty(N_ITERS, dtype=np.float32)
    ts = np.empty(N_ITERS, dtype=np.float32)
    for i in range(N_ITERS):
        try:
            opt.step(None, seed=i) if pass_seed else opt.step(None)
        except Exception:
            return method, w0, t0, None, None
        ws[i] = ll.omega.item()
        ts[i] = ul.theta.item()
    if not (np.all(np.isfinite(ws)) and np.all(np.isfinite(ts))):
        return method, w0, t0, None, None
    return method, w0, t0, ws, ts


def main():
    with open("results/toy_tuned_params.json") as fh:
        tuned = json.load(fh)

    grid = np.linspace(GRID_LO, GRID_HI, GRID_N)
    jobs = [(m, tuned["refined"][m]["config"], float(w0), float(t0))
            for m in ORDER for w0 in grid for t0 in grid]
    print(f"c = {C:g};  {GRID_N}x{GRID_N} = {GRID_N**2} starts on "
          f"[{GRID_LO}, {GRID_HI}]^2, {N_ITERS} iterations, {len(jobs)} runs\n")

    with mp.Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        raw = pool.map(_worker, jobs, chunksize=4)

    data = {m: {"w": [], "t": [], "starts": [], "diverged": 0} for m in ORDER}
    for method, w0, t0, ws, ts in raw:
        if ws is None:
            data[method]["diverged"] += 1
            continue
        data[method]["w"].append(ws)
        data[method]["t"].append(ts)
        data[method]["starts"].append((w0, t0))

    summary = {}
    for m in ORDER:
        W = np.abs(np.array(data[m]["w"]) - W_STAR)          # (runs, iters)
        T = theta_error(np.array(data[m]["t"]))
        n = W.shape[0]
        total = GRID_N ** 2                      # ALL starts, diverged ones included
        final = W[:, -1]
        solved = final <= TOL
        wrong_basin = np.abs(np.array(data[m]["w"])[:, -1] - 2.0) < 1e-3
        summary[m] = {
            "runs": n,
            "total": total,
            "diverged": data[m]["diverged"],
            # medians/geomeans are over SURVIVING runs only -- see frac_completed
            "median_final_w_err": float(np.median(final)),
            "geomean_final_w_err": float(np.exp(np.mean(np.log(np.maximum(final, 1e-13))))),
            "median_final_theta_err": float(np.median(T[:, -1])),
            # fractions are over ALL starts: a diverged run is neither solved nor
            # in the wrong basin, it simply failed
            "frac_completed": n / total,
            "frac_solved": float(solved.sum()) / total,
            "frac_wrong_basin": float(wrong_basin.sum()) / total,
            "frac_diverged": data[m]["diverged"] / total,
            "median_curve_w": np.median(W, axis=0),
            "median_curve_t": np.median(T, axis=0),
            "success_curve": (W <= TOL).sum(axis=0) / total,
        }
        s = summary[m]
        print(f"  {m:<8} solved {s['frac_solved']*100:5.1f}%   "
              f"wrong basin (w->2) {s['frac_wrong_basin']*100:5.1f}%   "
              f"diverged {s['frac_diverged']*100:5.1f}%   "
              f"| median final |w-w*| = {s['median_final_w_err']:.3e}   "
              f"geomean = {s['geomean_final_w_err']:.3e}  "
              f"(over the {s['runs']}/{s['total']} starts that completed)")

    os.makedirs("results", exist_ok=True)
    with open("results/sweep_inits.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["algorithm", "w0", "theta0", "final_w", "final_theta",
                     "final_w_err", "final_theta_err", "solved"])
        for m in ORDER:
            for (w0, t0), ws, ts in zip(data[m]["starts"], data[m]["w"], data[m]["t"]):
                we = abs(float(ws[-1]) - W_STAR)
                wr.writerow([m, w0, t0, float(ws[-1]), float(ts[-1]), we,
                             float(theta_error(ts[-1])), int(we <= TOL)])
    print("\nwrote results/sweep_inits.csv")
    plot(summary)
    append_params(tuned, summary)


def _place_labels(ax, entries):
    lo, hi = np.log10(ax.get_ylim()[0]), np.log10(ax.get_ylim()[1])
    gap = (hi - lo) * 0.06
    placed, prev = [], -np.inf
    for name, y, colour in sorted(entries, key=lambda e: e[1]):
        ly = max(np.log10(max(y, 10 ** lo)), prev + gap)
        placed.append((name, 10 ** ly, colour))
        prev = ly
    return placed


def plot(summary, path="results/sweep_inits.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(True, which="major", color="#e6e6e3", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    it = np.arange(1, N_ITERS + 1)
    ends = [[], []]
    for m in ORDER:
        s = summary[m]
        lbl = m if s["frac_completed"] > 0.999 else f"{m} ({s['frac_completed']*100:.0f}% of starts)"
        axes[0].plot(it, np.maximum(s["median_curve_w"], 1e-13), color=COLOURS[m],
                     linewidth=2, label=lbl)
        axes[1].plot(it, np.maximum(s["median_curve_t"], 1e-13), color=COLOURS[m],
                     linewidth=2, label=m)
        axes[2].plot(it, s["success_curve"], color=COLOURS[m], linewidth=2, label=m)
        ends[0].append((m, max(s["median_curve_w"][-1], 1e-13), COLOURS[m]))
        ends[1].append((m, max(s["median_curve_t"][-1], 1e-13), COLOURS[m]))

    axes[0].set_title(r"median $|w - w^*|$ across starts", fontsize=10)
    axes[1].set_title(r"median $\min(|\theta|, |\theta-1|)$ across starts", fontsize=10)
    axes[2].set_title(rf"fraction of starts with $|w-w^*| \leq$ {TOL:g}", fontsize=10)
    axes[2].set_ylim(-0.03, 1.03)

    for ax, bucket in zip(axes[:2], ends):
        ax.set_xlim(1, N_ITERS * 1.34)
        for label, y, colour in _place_labels(ax, bucket):
            ax.annotate(label, xy=(N_ITERS * 1.04, y), fontsize=7.5,
                        color="#3d3d38", va="center")
    axes[2].set_xlim(1, N_ITERS)
    for m in ORDER:
        axes[2].annotate(m, xy=(N_ITERS * 0.62, summary[m]["success_curve"][-1]),
                         xytext=(0, 4), textcoords="offset points",
                         fontsize=7.5, color="#3d3d38")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
    fig.suptitle(f"Toy bilevel problem, $c={C:g}$ — averaged over a "
                 f"{GRID_N}$\\times${GRID_N} grid of starts "
                 f"$(w_0, \\theta_0) \\in [{GRID_LO}, {GRID_HI}]^2$",
                 fontsize=11, color="#3d3d38")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def append_params(tuned, summary):
    path = "results/toy_comparison_params.txt"
    L = ["", "=" * 78,
         f"INITIAL-CONDITION SWEEP  (results/sweep_inits.png, results/sweep_inits.csv)",
         "=" * 78, "",
         f"  grid        = {GRID_N} x {GRID_N} starts, w_0 and theta_0 each "
         f"linspace({GRID_LO}, {GRID_HI}, {GRID_N})",
         f"  iterations  = {N_ITERS} per start, deterministic, float32, torch seed 0",
         f"  tolerance   = {TOL:g}",
         "  hyperparameters: unchanged from the single-start tuning above (each method",
         "  uses its own tuned configuration); this measures how that setting generalises",
         "",
         "  theta* in {0, 1}, so the theta error is min(|theta|, |theta - 1|).",
         "  Starts with w_0 > 1 lie in the WRONG lower-level basin (J pulls to w = 2,",
         "  F pulls to w = 0), so 'wrong basin' below counts runs that finished at w = 2.",
         "",
         "  Fractions are over ALL starts. Medians and geomeans are over the starts that",
         "  COMPLETED, so they are not comparable across methods with different",
         "  divergence rates - read them next to the 'diverged' column.",
         "",
         f"  {'method':<8} {'solved':>8} {'wrong basin':>12} {'diverged':>9} "
         f"{'median |w-w*|':>15} {'geomean':>12}"]
    for m in ORDER:
        s = summary[m]
        L.append(f"  {m:<8} {s['frac_solved']*100:>7.1f}% "
                 f"{s['frac_wrong_basin']*100:>11.1f}% {s['frac_diverged']*100:>8.1f}% "
                 f"{s['median_final_w_err']:>15.3e} {s['geomean_final_w_err']:>12.3e}")
    L += ["",
          "  CAVEAT - the single-start tuning does not transfer:",
          "    The tuning run had theta frozen at the stationary point theta_0 = 0.5, and",
          "    the tuning objective only measured |w - w*|. A large proximal weight tau",
          "    therefore cost nothing there, and the tuner picked tau = 10 for PABLO.",
          "    Once theta can move, that tau pins the theta error at 1.36e-03 on EVERY one",
          "    of the 400 starts (min 1.344e-03, max 1.365e-03), while BOME, TSP and STABLE",
          "    reach 1e-18 or below. Cause confirmed by ablation, holding all else fixed:",
          "",
          "        tau = 0.02  ->  theta_err = 4.4e-43        eps       : no effect",
          "        tau = 0.2   ->  theta_err = 1.7e-43        grad_clip : no effect",
          "        tau = 2.0   ->  theta_err = 8.5e-04        lr        : scales with lr",
          "        tau = 10.0  ->  theta_err = 1.4e-03",
          "",
          "    The reference y is an EMA of Z, so it lags during geometric decay; with tau",
          "    large the lag term tau(Z - y) dominates dF/dtheta and holds theta away from",
          "    theta*. Setting y_rate = gamma*tau = 1 (y = Z each step, no lag) also removes",
          "    the floor. PABLO's numbers here should be re-tuned against a non-degenerate",
          "    objective before being used.",
          "",
          "  REPRODUCE: python experiments/sweep_inits.py", ""]
    with open(path, "a") as fh:
        fh.write("\n".join(L))
    print(f"appended sweep section to {path}")


if __name__ == "__main__":
    main()
