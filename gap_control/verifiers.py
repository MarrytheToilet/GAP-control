"""Hard-attribute verifiers (Family H): exact, programmatic, training-free rewards.

Each verifier maps text -> reward in [0, 1] given a parameter spec. They are
language-agnostic except length, which counts tokens with a provided tokenizer so it works
for English and Chinese alike (handbook 3.2 group D).

These are the "hard constraint" axis that classifier-based CTG baselines handle poorly,
while GAP-Control treats them with the same advantage-residual mechanism.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

import torch


# ----------------------------- length -----------------------------
LENGTH_BUCKETS = {"short": (0, 25), "medium": (26, 60), "long": (61, 10_000)}


def _tok_len(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text.split())


def length_reward(texts: List[str], *, bucket: Optional[str] = None,
                  target: Optional[int] = None, tolerance: int = 8,
                  tokenizer=None) -> torch.Tensor:
    """Reward 1 inside the target bucket / within tolerance of target, else exp decay.

    Provide either `bucket` (short/medium/long) or a numeric `target` length.
    """
    out = []
    for t in texts:
        n = _tok_len(t, tokenizer)
        if target is not None:
            d = abs(n - target)
            r = 1.0 if d <= tolerance else float(torch.exp(torch.tensor(-(d - tolerance) / 20.0)))
        elif bucket is not None:
            lo, hi = LENGTH_BUCKETS[bucket]
            if lo <= n <= hi:
                r = 1.0
            else:
                d = lo - n if n < lo else n - hi
                r = float(torch.exp(torch.tensor(-d / 20.0)))
        else:
            raise ValueError("length_reward needs `bucket` or `target`")
        out.append(r)
    return torch.tensor(out)


# ----------------------------- keyword -----------------------------
def keyword_reward(texts: List[str], *, keywords: List[str],
                   mode: str = "substring") -> torch.Tensor:
    """Fraction of required keywords present. mode: 'substring' (default) or 'word'."""
    out = []
    kws = [k.lower() for k in keywords]
    for t in texts:
        low = t.lower()
        if mode == "word":
            toks = set(re.findall(r"\w+", low))
            hit = sum(1 for k in kws if k in toks)
        else:
            hit = sum(1 for k in kws if k in low)
        out.append(hit / max(len(kws), 1))
    return torch.tensor(out)


# ----------------------------- structure -----------------------------
_LIST_RE = re.compile(r"(^|\n)\s*(\d+[\.\)]|[-*•])\s+", re.MULTILINE)
_ENUM_WORDS_RE = re.compile(r"\b(first|second|third|finally|next|then)\b", re.IGNORECASE)
_QUOTE_RE = re.compile(r"[\"“”‘’'].+?[\"“”‘’']")


def _is_interrogative(t: str) -> float:
    t = t.strip()
    return 1.0 if t.endswith("?") or t.endswith("？") else 0.0


def _is_exclamatory(t: str) -> float:
    t = t.strip()
    return 1.0 if t.endswith("!") or t.endswith("！") else 0.0


def _is_enumeration(t: str) -> float:
    if _LIST_RE.search(t):
        return 1.0
    # soft credit for >=2 sequence-marker words
    return 1.0 if len(_ENUM_WORDS_RE.findall(t)) >= 2 else 0.0


def _is_dialogue(t: str) -> float:
    return 1.0 if _QUOTE_RE.search(t) else 0.0


STRUCTURE_FNS: dict[str, Callable[[str], float]] = {
    "interrogative": _is_interrogative,
    "exclamatory": _is_exclamatory,
    "enumeration": _is_enumeration,
    "dialogue": _is_dialogue,
}


def structure_reward(texts: List[str], *, pattern: str) -> torch.Tensor:
    fn = STRUCTURE_FNS[pattern]
    return torch.tensor([fn(t) for t in texts])
