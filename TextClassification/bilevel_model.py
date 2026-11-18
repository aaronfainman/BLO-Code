"""Bilevel model for text classification with a small causal LM.

    frozen trunk (embeddings + layers 0..L-2)            -- run ONCE, outputs cached
      -> last decoder layer          [UL params theta]   -- trained on validation CE
      -> final RMSNorm               [UL params theta, unless train_norm=False]
      -> pooling (last token / mean)
      -> classification head         [LL params W]       -- trained on training CE
      -> 5 class logits

Because layers 0..L-2 are frozen, their output never changes. `encode_trunk` computes it
once per dataset; every loss evaluation afterwards runs a single decoder layer.

The decoder layer is called directly with a hand-built additive attention mask, rather
than via transformers' create_causal_mask, whose signature differs between 4.x and 5.x.
The additive float mask is accepted by both the "eager" and "sdpa" attention backends.

Second-order terms: the BLO methods differentiate grad_W J (W = head) w.r.t. theta. Since
the head sits downstream of the layer, that only needs a double backward through the head
(linear + cross-entropy) and a SINGLE backward through the attention layer, so SDPA's
missing double-backward support is not a problem.
"""
import copy

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- helpers

def additive_mask(attn: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """attn [B, T] (1 = token, 0 = pad) -> additive mask [B, 1, T, T]: causal AND key is a token."""
    B, T = attn.shape
    causal = torch.ones(T, T, dtype=torch.bool, device=attn.device).tril()
    allowed = causal[None, None] & attn[:, None, None, :].bool()
    mask = torch.zeros(B, 1, T, T, dtype=dtype, device=attn.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


def run_layers(layers, h: torch.Tensor, attn: torch.Tensor, rotary: nn.Module) -> torch.Tensor:
    """Apply decoder layers to hidden states h [B, T, d] with right padding mask attn [B, T]."""
    pos = torch.arange(h.shape[1], device=h.device)[None].expand(h.shape[0], -1)
    pos_emb = rotary(h, pos)
    mask = additive_mask(attn, h.dtype)
    for layer in layers:
        out = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pos_emb)
        h = out[0] if isinstance(out, tuple) else out   # transformers < 4.54 returns a tuple
    return h


def pool(h: torch.Tensor, attn: torch.Tensor, how: str) -> torch.Tensor:
    """h [B, T, d] -> [B, d]. Assumes RIGHT padding."""
    if how == "last":
        idx = attn.sum(1) - 1
        return h[torch.arange(h.shape[0], device=h.device), idx]
    if how == "mean":
        m = attn.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1)
    raise ValueError(f"pooling must be 'last' or 'mean', got {how!r}")


def trim(h: torch.Tensor, attn: torch.Tensor):
    """Drop trailing all-padding columns of a right-padded batch (saves attention compute)."""
    T = int(attn.sum(1).max())
    return h[:, :T], attn[:, :T]


# ----------------------------------------------------------------------------- models

class Representation(nn.Module):
    """UL model (theta): last decoder layer -> final RMSNorm -> pooling. Output [B, d].

    Holds deep COPIES of the base model's last layer, norm and rotary embedding, so each
    run starts from the pretrained weights and the base model is never modified.
    The layer and the norm are both trainable by default; train_norm=False freezes the norm
    (frozen parameters are excluded by the BLO optimisers' requires_grad filter).
    """

    def __init__(self, base_lm, pooling: str = "last", train_norm: bool = True):
        super().__init__()
        inner = base_lm.model
        self.layer = copy.deepcopy(inner.layers[-1]).float()
        self.norm = copy.deepcopy(inner.norm).float()
        self.rotary = copy.deepcopy(inner.rotary_emb)   # buffers only, no parameters
        self.pooling = pooling
        self.layer.requires_grad_(True)
        self.norm.requires_grad_(train_norm)

    def forward(self, h: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        h, attn = trim(h, attn)
        h = run_layers([self.layer], h.float(), attn, self.rotary)
        return pool(self.norm(h), attn, self.pooling)


class ClassificationHead(nn.Module):
    """LL model (W): linear map from pooled features to class logits."""

    def __init__(self, d_model: int, n_classes: int = 5):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def build_bilevel_models(base_lm, n_classes: int = 5, pooling: str = "last",
                         train_norm: bool = True, seed: int = 0, device=None):
    """Fresh (representation, head) pair. `seed` fixes the head's random initialisation."""
    device = device or next(base_lm.parameters()).device
    rep = Representation(base_lm, pooling=pooling, train_norm=train_norm).to(device)
    with torch.random.fork_rng(devices=[]):   # seeded head init without touching the global RNG
        torch.manual_seed(seed)
        head = ClassificationHead(base_lm.config.hidden_size, n_classes)
    return rep, head.to(device)


def classify(rep: Representation, head: ClassificationHead, h, attn) -> torch.Tensor:
    """Full bilevel forward on cached trunk states: logits [B, n_classes]."""
    return head(rep(h, attn))


# ----------------------------------------------------------------------------- trunk cache

@torch.no_grad()
def encode_trunk(base_lm, tok, prompts, batch_size: int = 32, store_device="cpu",
                 store_dtype=torch.float32):
    """Run the frozen trunk (embeddings + layers 0..L-2) once over `prompts`.

    Returns (h [N, T, d], attn [N, T]): the hidden states entering the last layer, right
    padded to the longest prompt. Kept on `store_device` (move batches to the GPU as needed,
    or pass the GPU here if the whole cache fits).
    """
    inner = base_lm.model
    device = next(base_lm.parameters()).device
    enc = [tok(p).input_ids for p in prompts]
    T = max(map(len, enc))
    pad = tok.pad_token_id
    H = torch.empty(len(enc), T, base_lm.config.hidden_size, dtype=store_dtype, device=store_device)
    A = torch.zeros(len(enc), T, dtype=torch.long, device=store_device)

    for i in range(0, len(enc), batch_size):
        chunk = enc[i:i + batch_size]
        Tb = max(map(len, chunk))
        ids = torch.full((len(chunk), Tb), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), Tb), dtype=torch.long)
        for k, s in enumerate(chunk):                          # RIGHT padding
            ids[k, :len(s)] = torch.tensor(s)
            att[k, :len(s)] = 1
        ids, att = ids.to(device), att.to(device)
        h = inner.embed_tokens(ids)
        h = run_layers(inner.layers[:-1], h, att, inner.rotary_emb)
        H[i:i + len(chunk), :Tb] = h.to(store_device, store_dtype)
        H[i:i + len(chunk), Tb:] = 0
        A[i:i + len(chunk), :Tb] = att.to(store_device)
    return H, A
