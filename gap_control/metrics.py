"""Evaluation metrics (handbook 3.5).

Grouped exactly as the handbook's metric system:
  * control     -- attribute success / mean reward
  * perturbation-- KL to base, top-k overlap, selected-token base rank
  * quality     -- perplexity (under the base model), distinct-n
  * efficiency  -- ms/token (collected during decode)
  * diagnostics -- bias-advantage correlation (the key "did it learn advantage" check)
"""
from __future__ import annotations

from typing import List

import torch


# ---- control ----
def control_success(rewards: torch.Tensor, threshold: float = 0.5) -> float:
    return (rewards >= threshold).float().mean().item()


def mean_reward(rewards: torch.Tensor) -> float:
    return rewards.mean().item()


# ---- quality ----
@torch.no_grad()
def perplexity(base, texts: List[str]) -> float:
    """Mean per-token PPL of `texts` under the base model."""
    import math
    nlls, ntok = [], 0
    for t in texts:
        ids = base.tokenizer(t, return_tensors="pt").input_ids.to(base.device)
        if ids.size(1) < 2:
            continue
        out = base.model(ids, labels=ids)
        n = ids.size(1) - 1
        nlls.append(out.loss.item() * n)
        ntok += n
    if ntok == 0:
        return float("nan")
    return math.exp(sum(nlls) / ntok)


@torch.no_grad()
def conditional_perplexity(base, prompts: List[str], gens: List[str]) -> float:
    """Mean PPL of the GENERATED text only, conditioned on the prompt (the prompt is used
    as context but excluded from the average — standard CTG fluency, prompt not counted)."""
    import math
    nlls, ntok = [], 0
    for prompt, gen in zip(prompts, gens):
        if not gen.strip():
            continue
        p_ids = base.tokenizer(prompt, return_tensors="pt").input_ids.to(base.device)
        f_ids = base.tokenizer(prompt + gen, return_tensors="pt").input_ids.to(base.device)
        n_p = p_ids.size(1)
        if f_ids.size(1) - n_p < 1:
            continue
        logits = base.model(f_ids).logits
        # predict token t from position t-1; score only generated positions [n_p, end)
        tgt = f_ids[0, n_p:]
        lp = logits[0, n_p - 1:-1].log_softmax(-1)
        nll = -lp[range(tgt.size(0)), tgt].sum().item()
        nlls.append(nll); ntok += tgt.size(0)
    return math.exp(sum(nlls) / ntok) if ntok else float("nan")


def distinct_n(texts: List[str], n: int = 2) -> float:
    grams, total = set(), 0
    for t in texts:
        toks = t.split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i:i + n]))
            total += 1
    return len(grams) / max(total, 1)


# ---- diagnostics ----
def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors."""
    if a.numel() < 2:
        return float("nan")
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return (ra @ rb / denom).item()


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).item()


def topk_overlap(p_ctrl_ids: List[int], p_base_ids: List[int]) -> float:
    s1, s2 = set(p_ctrl_ids), set(p_base_ids)
    return len(s1 & s2) / max(len(s1 | s2), 1)
