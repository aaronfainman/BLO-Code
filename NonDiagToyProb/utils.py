
from dataclasses import dataclass, field
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Any


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

DIFFICULTY_NAMES = ["easy", "medium", "hard"]


def get_difficulty_weights(data: dict, ll_model: nn.Module, 
                           ul_model: nn.Module,
                           device=torch.device('cpu'),) -> dict[str, float]:
    """Return the mean auxiliary-network weight for each difficulty tier.

    data     : dict as returned by data_utils.get_batch()
    ll_model : lower-level model (CIFARSqueezeNet) — used to extract features
    ul_model : upper-level / auxiliary model — maps features → sample weights

    Returns a dict  {'easy': mean_w, 'medium': mean_w, 'hard': mean_w}
    """
    images            = data['features_train'].to(device)
    difficulty_labels = data['difficulty_train'].to(device)

    ll_model.eval()
    ul_model.eval()
    # with torch.no_grad():
    #     pred_logits, _ = ll_model(images, return_features = True)
    #     per_sample_logit_loss = F.cross_entropy(pred_logits,data['labels_train'].to(device), reduction="none")
    #     weights = ul_model(per_sample_logit_loss)
    with torch.no_grad():
        _, features = ll_model(images, return_features=True)
        weights = ul_model(features).squeeze(-1)   # shape [B]

    ll_model.train()
    ul_model.train()

    result = {}
    for label, name in enumerate(DIFFICULTY_NAMES):
        mask = difficulty_labels == label
        result[name] = weights[mask].mean().item() if mask.any() else float('nan')
    return result



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
        # metric = list(metric)
        # metric.append(metric[-1]*mean_window)
        # metric=[metric[0]*mean_window, *metric, metric[-1]*mean_window]
        plt_metric = np.convolve(metric,filter,mode='valid').tolist()
        plt_metric = [metric[0], *plt_metric]
        # plt_metric = plt_metric[(mean_window-1):]
        if len(plt_metric)<(st2-st1):
            st2 = len(plt_metric)
        # print(len(plt_metric))
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

@dataclass
class BLOMetrics:

    ll_loss: list[float]
    ul_loss: list[float]
    test_loss: list[float]
    grad_w_L_norm: list[float]
    grad_theta_L_norm: list[float]
    lr: list[float]
    other: dict[float, list[Any]]

    def __init__(self,
                 ll_loss: float | list[float] | None = None,
                 ul_loss: float | list[float] | None = None,
                 test_loss: float | list[float] | None = None,
                 grad_w_L_norm: float | list[float] | None = None,
                 grad_theta_L_norm: float | list[float] | None = None,
                 lr: float | list[float] | None = None,
                 other: dict[float, Any | list[Any]] | None = None):

        def _to_list(v):
            if v is None:        return []
            if isinstance(v, list): return v
            return [v]

        self.ll_loss = _to_list(ll_loss)
        self.ul_loss = _to_list(ul_loss)
        self.test_loss = _to_list(test_loss)
        self.grad_w_L_norm = _to_list(grad_w_L_norm)
        self.grad_theta_L_norm = _to_list(grad_theta_L_norm)
        self.lr = _to_list(lr)
        self.other: dict[float, list[Any]] = {}
        if other:
            for k, L in other.items():
                self.other[k]  = _to_list(L)

    def update(self, results: dict, lr: float | None = None, other: dict[float, Any] | None = None):
        """Append one step's results dict (as returned by optimiser.step())."""
        if 'll_loss' in results: self.ll_loss.append(results['ll_loss'])
        if 'ul_loss' in results: self.ul_loss.append(results['ul_loss'])
        if 'grad_w_L_norm' in results: self.grad_w_L_norm.append(results['grad_w_L_norm'])
        if 'grad_theta_L_norm' in results: self.grad_theta_L_norm.append(results['grad_theta_L_norm'])
        if 'test_loss' in results: self.test_loss.append(results['test_loss'])
        if lr is not None: self.lr.append(lr)
        if other is not None:
            for k, v in other.items():
                if k not in self.other:
                    self.other[k] = []
                self.other[k].append(v)



    def __len__(self):
        return len(self.ll_loss)

    def __repr__(self):
        n = len(self)
        ll  = f"{self.ll_loss[-1]:.4f}"  if self.ll_loss  else "—"
        ul  = f"{self.ul_loss[-1]:.4f}"  if self.ul_loss  else "—"
        tst = f"{self.test_loss[-1]:.4f}" if self.test_loss else "—"
        gwL = f"{self.grad_w_L_norm[-1]:.4f}" if self.grad_w_L_norm else "—"
        gtL = f"{self.grad_theta_L_norm[-1]:.4f}" if self.grad_w_L_norm else "—"
        return f"BLOMetrics(steps={n}, ll={ll}, ul={ul}, test={tst}, grad_w_L={gwL}, grad_theta_L={gtL})"


@dataclass
class CL_BLOMetrics(BLOMetrics):
    easy_weights: list[float]
    medium_weights: list[float]
    hard_weights: list[float]

    def __init__(self,
                 ll_loss: float | list[float] | None = None,
                 ul_loss: float | list[float] | None = None,
                 test_loss: float | list[float] | None = None,
                 grad_w_L_norm: float | list[float] | None = None,
                 grad_theta_L_norm: float | list[float] | None = None,
                 lr: float | list[float] | None = None,
                 easy_weights: float | list[float] | None = None,
                 medium_weights: float | list[float] | None = None,
                 hard_weights: float | list[float] | None = None, 
                 other: dict[float, Any | list[Any]] | None = None):
        
        super().__init__(ll_loss, ul_loss, test_loss, grad_w_L_norm, grad_theta_L_norm,
                         lr)
        def _to_list(v):
            if v is None:        return []
            if isinstance(v, list): return v
            return [v]
        
        self.easy_weights = _to_list(easy_weights)
        self.medium_weights = _to_list(medium_weights)
        self.hard_weights = _to_list(hard_weights)

    def update(self, results: dict|None = {}, 
               lr = None, 
               difficulty_weights:dict[str,float]|None = None,
               other: dict[float, Any | list[Any]] | None = None):

        super().update(results, lr, other)

        if difficulty_weights:
            self.easy_weights.append(difficulty_weights['easy'])
            self.medium_weights.append(difficulty_weights['medium'])
            self.hard_weights.append(difficulty_weights['hard'])