"""Per-step running state: the signal that makes control *dynamic* (handbook 2.5 idea
"隐向量反映当前偏好满足情况").

At every generation step the controller is told how far the constraint is *already*
satisfied by the text generated so far. This closes the loop: the value head and residual
budget adapt to the live state instead of applying a static bias.

Two functions, both computed exactly from the generated tokens (no extra model):
  * running_state(...)          -> fixed-dim feature fed to the controller
  * observed_satisfaction(...)  -> a lower bound on final reward for *monotone* constraints
                                   (keyword/contains): once satisfied it stays satisfied,
                                   so the gap can be provably driven to 0 -> residual off.
"""
from __future__ import annotations

import torch

from . import verifiers
from .attributes import ControlCondition

STATE_DIM = 4   # [norm_length, length_progress, keyword_coverage, structure_satisfied]

_BUCKET_REF = {"short": 25, "medium": 60, "long": 100}


def _length_target(condition: ControlCondition):
    for c in condition.components:
        if c.dim == "length":
            if c.length_target is not None:
                return c.length_target
            return _BUCKET_REF.get(c.value, 60)
    return None


def running_state(text: str, n_tokens: int, condition: ControlCondition,
                  *, max_new: int) -> torch.Tensor:
    """Fixed [STATE_DIM] feature from the continuation generated so far.

    text: decoded continuation. n_tokens: number of generated tokens.
    """
    feat = torch.zeros(STATE_DIM)
    feat[0] = min(n_tokens / max(max_new, 1), 1.5)                 # how far through budget
    tgt = _length_target(condition)
    if tgt:
        feat[1] = min(n_tokens / max(tgt, 1), 2.0)                 # length progress vs target
    kw = condition.keyword_tokens()
    if kw:
        feat[2] = verifiers.keyword_reward([text], keywords=kw)[0].item()  # coverage so far
    for c in condition.components:
        if c.dim == "structure":
            feat[3] = verifiers.structure_reward([text], pattern=c.value)[0].item()
            break
    return feat


def observed_satisfaction(text: str, condition: ControlCondition) -> float:
    """Lower bound on final R_c from monotone constraints already met by `text`.

    Only keyword/"contains" are monotone (a present keyword cannot disappear). For terminal
    constraints (length/structure) mid-stream satisfaction is NOT a guarantee, so they
    contribute 0 here. Returns the keyword coverage weighted into [0,1] over the condition;
    used to clamp the value gap so the residual switches off once the keyword is in.
    """
    kw = condition.keyword_tokens()
    if not kw:
        return 0.0
    cov = verifiers.keyword_reward([text], keywords=kw)[0].item()
    # if keyword is the only component, coverage is the full observed satisfaction;
    # otherwise it's a partial floor proportional to its mixture share.
    n = len(condition.components)
    return cov if n == 1 else cov / n
