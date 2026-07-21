"""Attribute registry, control conditions, and the reward mixture R_c (handbook 2.3).

A control condition is a list of components, each component = (dim, value, alpha [, params]).
The same condition object yields:
  * a control vector for the encoder  -> `to_alpha(num_slots)` (+ optional keyword embedding)
  * the scalar reward R_c(y)          -> `reward(texts, ...)`

Soft dims (sentiment/emotion/style) need a classifier bank (duck-typed: `bank.prob(dim,
value, texts) -> [N]`). Hard dims (length/keyword/structure) use the exact verifiers.

This is what makes single-attribute, continuous-intensity, and multi-attribute composition
all the *same* code path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from . import verifiers


# ----------------------- canonical taxonomy -----------------------
# Attribute-set presets. Select with GAPCTRL_ATTRS (default = the paper's library);
# CompMCTG presets mirror the benchmark's aspects (Zhong et al. 2024), lowercased.
_ATTR_PRESETS = {
    "default": {
        "soft": {
            "sentiment": ["positive", "negative", "neutral"],
            "emotion": ["joy", "anger", "sadness", "fear"],
            "style": ["formal", "informal", "literary"],
        },
        "intensity": {"sentiment": ("negative", "positive"),
                      "style": ("informal", "formal")},
    },
    "yelp": {  # CompMCTG-Yelp: 3 aspects, 8 combinations
        "soft": {
            "sentiment": ["positive", "negative"],
            "pronoun": ["singular", "plural"],
            "tense": ["present", "past"],
        },
        "intensity": {"sentiment": ("negative", "positive")},
    },
    "fyelp": {  # CompMCTG-Fyelp: 4 aspects, 40 combinations
        "soft": {
            "sentiment": ["positive", "negative"],
            "gender": ["male", "female"],
            "cuisine": ["asian", "american", "mexican", "bar", "dessert"],
            "tense": ["present", "past"],
        },
        "intensity": {"sentiment": ("negative", "positive")},
    },
    "amazon": {  # CompMCTG-Amazon: 2 aspects, 12 combinations
        "soft": {
            "sentiment": ["positive", "negative"],
            "topic": ["books", "clothing", "music", "electronics", "movies", "sports"],
        },
        "intensity": {"sentiment": ("negative", "positive")},
    },
}
ATTR_SET = os.environ.get("GAPCTRL_ATTRS", "default")
SOFT_DIMS = _ATTR_PRESETS[ATTR_SET]["soft"]
HARD_CATEGORICAL = {
    "length": ["short", "medium", "long"],
    "structure": ["interrogative", "exclamatory", "enumeration", "dialogue"],
}
# keyword is parametric (no fixed class) -> a single "constraint active" indicator slot
PARAMETRIC_DIMS = ["keyword"]

# continuous-intensity axes (handbook 3.2 group B): (dim, low_class, high_class)
INTENSITY_AXES = _ATTR_PRESETS[ATTR_SET]["intensity"]


def _build_slots():
    """Stable, ordered slot index for every categorical class + keyword indicator."""
    slots, idx = {}, 0
    for dim, classes in {**SOFT_DIMS, **HARD_CATEGORICAL}.items():
        for c in classes:
            slots[(dim, c)] = idx
            idx += 1
    slots[("keyword", "present")] = idx
    idx += 1
    return slots, idx


SLOTS, NUM_SLOTS = _build_slots()
SOFT_DIM_SET = set(SOFT_DIMS)
ALL_DIMS = list(SOFT_DIMS) + list(HARD_CATEGORICAL) + PARAMETRIC_DIMS


def dim_of(value_or_dim: str):
    for dim, classes in {**SOFT_DIMS, **HARD_CATEGORICAL}.items():
        if value_or_dim in classes:
            return dim
    return None


@dataclass
class Component:
    dim: str                              # sentiment/emotion/style/length/keyword/structure
    value: Optional[str] = None           # class / bucket / structure pattern
    alpha: float = 1.0                    # mixture weight (also intensity for soft dims)
    length_target: Optional[int] = None   # for dim == length (overrides bucket)
    tolerance: int = 8
    keywords: Optional[List[str]] = None  # for dim == keyword

    def reward(self, texts: List[str], *, bank=None, tokenizer=None) -> torch.Tensor:
        if self.dim in SOFT_DIM_SET:
            if bank is None:
                raise ValueError(f"soft dim {self.dim} needs a classifier bank")
            return bank.prob(self.dim, self.value, texts)
        if self.dim == "length":
            return verifiers.length_reward(
                texts, bucket=self.value if self.length_target is None else None,
                target=self.length_target, tolerance=self.tolerance, tokenizer=tokenizer)
        if self.dim == "keyword":
            return verifiers.keyword_reward(texts, keywords=self.keywords or [])
        if self.dim == "structure":
            return verifiers.structure_reward(texts, pattern=self.value)
        raise ValueError(f"unknown dim {self.dim}")


@dataclass
class ControlCondition:
    components: List[Component] = field(default_factory=list)
    aggregate: str = "mean"   # "mean" (weighted), "min" (joint/harsh), "gmean"
    conflict_mu: float = 0.0  # penalty on spread between component rewards

    # ---------- reward R_c(y) ----------
    def reward(self, texts: List[str], *, bank=None, tokenizer=None) -> torch.Tensor:
        rs = torch.stack([c.reward(texts, bank=bank, tokenizer=tokenizer)
                          for c in self.components])           # [C, N]
        w = torch.tensor([c.alpha for c in self.components]).clamp_min(1e-6).unsqueeze(1)
        if self.aggregate == "mean":
            agg = (w * rs).sum(0) / w.sum()
        elif self.aggregate == "min":
            agg = rs.min(0).values
        elif self.aggregate == "gmean":
            agg = (rs.clamp_min(1e-6).log() * w).sum(0) / w.sum()
            agg = agg.exp()
        else:
            raise ValueError(self.aggregate)
        if self.conflict_mu > 0 and rs.size(0) > 1:
            agg = agg - self.conflict_mu * (rs.max(0).values - rs.min(0).values)
        return agg.clamp(0.0, 1.0)

    def joint_success(self, texts: List[str], *, bank=None, tokenizer=None,
                      thresh: float = 0.5) -> torch.Tensor:
        """All components individually satisfied (handbook 3.8 joint success)."""
        rs = torch.stack([c.reward(texts, bank=bank, tokenizer=tokenizer)
                          for c in self.components])
        return (rs >= thresh).all(0).float()

    # ---------- control vector for the encoder ----------
    def to_alpha(self, num_slots: int = NUM_SLOTS) -> torch.Tensor:
        a = torch.zeros(num_slots)
        for c in self.components:
            if c.dim == "keyword":
                a[SLOTS[("keyword", "present")]] = c.alpha
                continue
            if c.dim == "length" and c.length_target is not None:
                # map numeric target to nearest bucket slot for encoding
                bucket = next(b for b, (lo, hi) in verifiers.LENGTH_BUCKETS.items()
                              if lo <= c.length_target <= hi)
                a[SLOTS[("length", bucket)]] = c.alpha
                continue
            key = (c.dim, c.value)
            if key in SLOTS:
                a[SLOTS[key]] = c.alpha
        return a

    def keyword_tokens(self) -> Optional[List[str]]:
        for c in self.components:
            if c.dim == "keyword" and c.keywords:
                return c.keywords
        return None

    # ---- structure (interrogative/exclamatory) as a forced-marker push ----
    _STRUCT_MARK = {"interrogative": "?", "exclamatory": "!"}

    def structure_marks(self) -> List[str]:
        return [self._STRUCT_MARK[c.value] for c in self.components
                if c.dim == "structure" and c.value in self._STRUCT_MARK]

    def structure_active_ids(self, tokenizer) -> List[int]:
        """Token ids of required structural markers ('?', '!') so the residual can
        surface them in S_t, exactly as for a required keyword."""
        ids = set()
        for m in self.structure_marks():
            for variant in (m, " " + m):
                toks = tokenizer.encode(variant, add_special_tokens=False)
                if toks:
                    ids.add(toks[-1])
        return sorted(ids)

    def length_spec(self):
        """Return (lo, hi) target token range for a length component, else None.
        Used to bias the EOS logit (a one-step residual cannot count tokens)."""
        for c in self.components:
            if c.dim == "length":
                if c.length_target is not None:
                    t = c.length_target
                    return (max(1, int(0.75 * t)), int(1.25 * t))
                if c.value in verifiers.LENGTH_BUCKETS:
                    return verifiers.LENGTH_BUCKETS[c.value]
        return None

    def keyword_active_ids(self, tokenizer) -> List[int]:
        """Token ids that must be added to the active set so the residual can surface a
        required keyword (handbook: the residual only acts within S_t = TopK(pi_0); a rare
        keyword is otherwise out of reach). Returns the leading sub-token of each surface
        variant (' kw', 'kw', capitalized)."""
        kws = self.keyword_tokens()
        if not kws:
            return []
        ids = set()
        for kw in kws:
            for variant in (" " + kw, kw, " " + kw.capitalize(), kw.capitalize()):
                toks = tokenizer.encode(variant, add_special_tokens=False)
                if toks:
                    ids.add(toks[0])
        return sorted(ids)


# ---------- natural-language phrasing (for prompt / CFG / PREADD baselines) ----------
PHRASES = {
    ("sentiment", "positive"): "positive in sentiment",
    ("sentiment", "negative"): "negative in sentiment",
    ("sentiment", "neutral"): "neutral and objective",
    ("emotion", "joy"): "joyful", ("emotion", "anger"): "angry",
    ("emotion", "sadness"): "sad", ("emotion", "fear"): "fearful and anxious",
    ("style", "formal"): "formal and professional",
    ("style", "informal"): "casual and colloquial",
    ("style", "literary"): "vivid and literary",
    ("length", "short"): "very short", ("length", "medium"): "medium-length",
    ("length", "long"): "long and detailed",
    ("structure", "interrogative"): "phrased as a question",
    ("structure", "exclamatory"): "an excited exclamation",
    ("structure", "enumeration"): "a list of points",
    ("structure", "dialogue"): "a line of quoted dialogue",
}
# opposites used to build PREADD anti-prompts (None -> falls back to plain prompt)
OPPOSITE = {("sentiment", "positive"): ("sentiment", "negative"),
            ("sentiment", "negative"): ("sentiment", "positive"),
            ("style", "formal"): ("style", "informal"),
            ("style", "informal"): ("style", "formal")}


def _phrase(c: "Component") -> str:
    if c.dim == "keyword":
        return f"mentioning '{', '.join(c.keywords or [])}'"
    return PHRASES.get((c.dim, c.value), f"{c.value} {c.dim}")


def describe(cond: "ControlCondition") -> str:
    """e.g. 'positive in sentiment, formal and professional, mentioning ocean'."""
    return ", ".join(_phrase(c) for c in cond.components)


def describe_anti(cond: "ControlCondition") -> Optional[str]:
    """Opposite-attribute description for PREADD; None if no opposite is defined."""
    parts = []
    for c in cond.components:
        opp = OPPOSITE.get((c.dim, c.value))
        if opp:
            parts.append(PHRASES[opp])
    return ", ".join(parts) if parts else None


# ---------- convenience constructors ----------
def single(value: str, alpha: float = 1.0, **kw) -> ControlCondition:
    """One categorical attribute by its class name (dim inferred)."""
    dim = dim_of(value)
    if dim is None:
        raise ValueError(f"unknown attribute value {value!r}")
    return ControlCondition([Component(dim=dim, value=value, alpha=alpha, **kw)])


def compose(*specs, aggregate: str = "mean", conflict_mu: float = 0.0) -> ControlCondition:
    """Compose components. Each spec is a Component or a class-name string."""
    comps = []
    for s in specs:
        comps.append(s if isinstance(s, Component)
                     else Component(dim=dim_of(s), value=s))
    return ControlCondition(comps, aggregate=aggregate, conflict_mu=conflict_mu)
