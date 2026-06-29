"""LLM-as-judge evaluation (recent CTG practice; see LLMs-as-Judges survey 2024).

Scores generated text on three standard CTG dimensions (Air-Decoding / human-eval style):
  * relevance  -- does the text exhibit the target attribute? (the control metric)
  * fluency    -- is it grammatical / natural?
  * topicality -- is it a coherent continuation of the prompt?

Uses the configured judge model (distinct from the synthesis generators), via the
OpenAI-compatible API. Returns mean scores in [0,1] (rescaled from 1-5).
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List

from .env import get_client, synth_settings

_RUBRIC = ("Rate the TEXT on three axes, each an integer 1-5:\n"
           "- relevance: how strongly the text is {attribute} (5 = clearly {attribute}).\n"
           "- fluency: how grammatical and natural the text reads.\n"
           "- topicality: how coherently it continues the prompt.\n"
           'Reply ONLY as JSON: {{"relevance":x,"fluency":y,"topicality":z}}.')


def _score_one(model, attribute, prompt, text):
    msg = (f"Prompt: {prompt!r}\nText: {text!r}\n\n" +
           _RUBRIC.format(attribute=attribute))
    try:
        r = get_client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": msg}],
            temperature=0.0, max_tokens=40)
        out = r.choices[0].message.content
        d = {k: int(v) for k, v in re.findall(r'"(relevance|fluency|topicality)"\s*:\s*(\d)', out)}
        if len(d) == 3:
            return {k: (v - 1) / 4.0 for k, v in d.items()}  # 1-5 -> 0..1
    except Exception:
        pass
    return None


def judge(attribute: str, prompts: List[str], texts: List[str], model: str = None) -> dict:
    """Return mean {relevance, fluency, topicality} over the batch (skipping failures)."""
    s = synth_settings()
    model = model or s["judge_model"]
    with ThreadPoolExecutor(max_workers=s["concurrency"]) as ex:
        res = list(ex.map(lambda pt: _score_one(model, attribute, pt[0], pt[1]),
                          zip(prompts, texts)))
    res = [r for r in res if r]
    if not res:
        return {"relevance": float("nan"), "fluency": float("nan"),
                "topicality": float("nan"), "n": 0}
    agg = {k: sum(r[k] for r in res) / len(res) for k in ("relevance", "fluency", "topicality")}
    agg["n"] = len(res)
    return agg
