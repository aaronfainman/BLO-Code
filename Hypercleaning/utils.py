import inspect
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import get_batch, make_batch_iterators


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_classifier(classifier, loader, device=torch.device("cpu"), num_classes=10):
    """Returns (loss, accuracy, macro-F1 over classes, confusion matrix [true, pred])."""
    was_training = classifier.training
    classifier.eval()

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_loss, total_examples = 0.0, 0

    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        logits = classifier(images)
        predictions = logits.argmax(dim=1)

        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        total_examples += labels.numel()
        confusion += torch.bincount(labels.cpu() * num_classes + predictions.cpu(),
                                    minlength=num_classes ** 2).reshape(num_classes, num_classes)

    tp = confusion.diag().float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp
    macro_f1 = (2 * tp / (2 * tp + fp + fn).clamp_min(1)).mean().item()
    accuracy = tp.sum().item() / max(total_examples, 1)

    if was_training:
        classifier.train()

    return total_loss / max(total_examples, 1), accuracy, macro_f1, confusion


@torch.no_grad()
def sample_weight_stats(weight_model, corrupted_mask, threshold=0.5):
    """Mean clean/corrupted weights and the data-cleaning F1 score.

    A training sample is flagged as corrupted when its weight is below `threshold`;
    the F1 score measures how well the flagged set matches the true corrupted set.
    """
    n_train = len(corrupted_mask)
    device = next(weight_model.parameters()).device
    weights = weight_model(torch.arange(n_train, device=device)).detach().cpu()
    mask = corrupted_mask.cpu().bool()

    flagged = weights < threshold
    tp = (flagged & mask).sum().item()
    fp = (flagged & ~mask).sum().item()
    fn = (~flagged & mask).sum().item()

    return {
        "clean_weight":     weights[~mask].mean().item(),
        "corrupted_weight": weights[mask].mean().item(),
        "cleaning_f1":      2 * tp / max(2 * tp + fp + fn, 1),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

LOGGED_KEYS = ["ll_loss", "ul_loss"]
EVAL_KEYS = ["test_loss", "test_accuracy", "test_macro_f1",
             "clean_weight", "corrupted_weight", "cleaning_f1"]


def run_hypercleaning(optimiser, classifier, weight_model, datasets, corrupted_mask,
                      num_iters=1000, batch_size=100, seed=0, eval_every=10, print_every=100,
                      divergence_threshold=1e3, device=torch.device("cpu"), name=""):
    """Run one BLO optimiser on the hypercleaning problem.

    Every optimiser in blo_optimisers.py exposes step(data, **kwargs). BOME and TSP
    additionally take an explicit `seed`; it is only passed to optimisers whose step()
    names it, because the PABLO family forwards **kwargs straight into the loss functions.

    Returns a history dict: every key of the step() results per iteration, plus the
    evaluation metrics (EVAL_KEYS) at `eval_iterations`.
    """
    torch.manual_seed(seed)
    train_iter, val_iter = make_batch_iterators(datasets["train"], datasets["val"],
                                                batch_size=batch_size, seed=seed)
    test_loader = DataLoader(datasets["test"], batch_size=1000, shuffle=False)
    takes_seed = "seed" in inspect.signature(optimiser.step).parameters

    history = {"eval_iterations": [], **{k: [] for k in EVAL_KEYS}, "valid": True}
    t0 = time.time()

    def evaluate(iteration):
        test_loss, test_accuracy, test_macro_f1, _ = evaluate_classifier(classifier, test_loader, device)
        stats = sample_weight_stats(weight_model, corrupted_mask)
        history["eval_iterations"].append(iteration)
        for key, value in zip(EVAL_KEYS, [test_loss, test_accuracy, test_macro_f1, *stats.values()]):
            history[key].append(value)

    evaluate(0)

    for iteration in range(1, num_iters + 1):
        data = get_batch(train_iter, val_iter, device)
        results = optimiser.step(data, seed=seed + iteration) if takes_seed else optimiser.step(data)
        with torch.no_grad():
            weight_model.raw_theta.data.clamp_(min=-10,max=10)
            # weight_model[weight_model>5] = 5

        for key, value in results.items():
            history.setdefault(key, []).append(float(value))

        checked = [results["ll_loss"], results["ul_loss"]]
        if not all(np.isfinite(v) for v in checked) or any(abs(v) > divergence_threshold for v in checked):
            print(f"{name} diverged at iteration {iteration}: LL={checked[0]:.3e} UL={checked[1]:.3e}")
            history["valid"] = False
            break

        if iteration % eval_every == 0:
            evaluate(iteration)

        if print_every and iteration % print_every == 0:
            print(f"{name:10s} {iteration:5d} | LL={results['ll_loss']:.4f} | UL={results['ul_loss']:.4f} | "
                  f"test acc={100 * history['test_accuracy'][-1]:.2f}% | "
                  f"clean w={history['clean_weight'][-1]:.3f} | corrupt w={history['corrupted_weight'][-1]:.3f} | "
                  f"cleaning F1={100 * history['cleaning_f1'][-1]:.2f}% | {time.time() - t0:.0f}s")

    return history


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_metric(metric:list, plot_type:str='None', logy:bool=True,logx=False, plot_step:int=1, marker:str='-k', linewidth:float=None,start:int=0,ax:matplotlib.axes.Axes=None,mean_window:int=None):
    if metric is None or len(metric)==0:
        return
    st1,st2,st3 = [start,len(metric),plot_step]
    if 'run' in plot_type.lower():
        metric_sum = np.cumsum(metric)
        plt_metric = [ metric_sum[idx]/(idx+1) for idx in range(len(metric)) ]
    elif 'mov' in plot_type.lower():
        mean_window = mean_window if mean_window is not None else 5
        filter = np.array([1/mean_window]*mean_window)
        plt_metric = np.convolve(metric,filter,mode='valid').tolist()
        plt_metric = [metric[0], *plt_metric]
        if len(plt_metric)<(st2-st1):
            st2 = len(plt_metric)
    else:
        plt_metric = metric

    if ax:
        ax.plot(np.arange(st1,st2,st3), plt_metric[st1:st2:st3], marker,linewidth=linewidth)
        if logy:
            ax.set_yscale('log')
        if logx:
            ax.set_xscale('log')
    else:
        plt.plot(np.arange(st1,st2,st3), plt_metric[st1:st2:st3], marker,linewidth=linewidth)
        if logy:
            plt.yscale('log')
        if logx:
            plt.xscale('log')


def plot_confusion_matrix(confusion, title="Normalised confusion matrix", ax=None):
    normalised = (confusion.float() / confusion.sum(dim=1, keepdim=True).clamp_min(1)).numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    for i in range(normalised.shape[0]):
        for j in range(normalised.shape[1]):
            ax.text(j, i, f"{normalised[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if normalised[i, j] > 0.5 else "black")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit"); ax.set_ylabel("True digit"); ax.set_title(title)
    plt.colorbar(image, ax=ax)
    return ax
