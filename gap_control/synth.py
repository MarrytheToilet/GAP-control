"""Synthetic data generation + LLM-judge filtering for soft-attribute classifiers.

Role separation (so no model grades its own output):
  * generation : a rotated pool (GAPCTRL_GEN_MODELS) for stylistic diversity
  * filtering  : a distinct model (GAPCTRL_FILTER_MODEL) confirms each example's label
Each example is conditioned on a balanced seed cell (genre x topic x length x register; see
seeds.py) so a single class spans many genres/registers and the data is not skewed.

The judge confidence (0..1) doubles as a graded label for intensity calibration (handbook 5.3).
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from .env import get_client, synth_settings

# human-readable description of each (dim, class) used in prompts and the judge question
DESCRIPTIONS = {
    ("sentiment", "positive"): "expressing clearly positive sentiment",
    ("sentiment", "negative"): "expressing clearly negative sentiment",
    ("sentiment", "neutral"): "emotionally neutral and factual, neither positive nor negative",
    ("emotion", "joy"): "conveying joy or happiness",
    ("emotion", "anger"): "conveying anger or irritation",
    ("emotion", "sadness"): "conveying sadness or disappointment",
    ("emotion", "fear"): "conveying fear, worry, or anxiety",
    ("style", "formal"): "written in a formal, professional register",
    ("style", "informal"): "written in a casual, colloquial register",
    ("style", "literary"): "written in a vivid, literary, figurative style",
}


def _chat(model: str, messages, *, temperature: float, max_tokens: int,
          retries: int = 4) -> str:
    client = get_client()
    last = None
    for k in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens)
            return r.choices[0].message.content or ""
        except Exception as e:  # transient rate-limit / network
            last = e
            time.sleep(2 ** k)
    raise RuntimeError(f"chat failed for {model}: {last}")


def _extract_list(text: str) -> List[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    lines = [re.sub(r"^\s*[-*\d.\)]+\s*", "", l).strip() for l in text.splitlines()]
    return [l.strip(' "') for l in lines if len(l.strip()) > 3]


def generate_examples(dim: str, label: str, n: int, *, cell: dict, model: str,
                      temperature: float, max_tokens: int) -> List[str]:
    """Generate n examples for (dim,label) conditioned on a seed cell."""
    desc = DESCRIPTIONS.get((dim, label), f"{label} {dim}")
    reg = f" {cell['register']}." if cell.get("register") else "."
    user = (
        f"Write {n} diverse English text snippets, each {cell['length_desc']} long, "
        f"in the form of {cell['genre']} about {cell['topic']}{reg} "
        f"Every snippet must be {desc}. Make them genuinely different from each other "
        f"(different specifics, openings, vocabulary). Do not mention the label or the "
        f"word '{label}'. Return ONLY a JSON array of strings."
    )
    out = _chat(model,
                [{"role": "system", "content": "You generate diverse labeled text data for training text classifiers."},
                 {"role": "user", "content": user}],
                temperature=temperature, max_tokens=max_tokens)
    return _extract_list(out)[:n]


def judge_label(dim: str, label: str, text: str, *, model: str) -> float:
    """Filter-model confidence in [0,1] that `text` matches (dim, label)."""
    desc = DESCRIPTIONS.get((dim, label), f"{label} {dim}")
    user = (f"Text: {text!r}\nIs this text {desc}? "
            f"Answer with a single integer 0 (definitely not) to 100 (definitely yes). "
            f"Reply with only the number.")
    out = _chat(model, [{"role": "user", "content": user}],
                temperature=0.0, max_tokens=8)
    m = re.search(r"\d+", out)
    return min(max(int(m.group()) / 100.0, 0.0), 1.0) if m else 0.0


def filter_examples(dim: str, label: str, items: List[dict], *,
                    filter_model: str, keep_threshold: float, concurrency: int) -> List[dict]:
    """Judge each item['text'] with the filter model; keep above threshold. Concurrent."""
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        scores = list(ex.map(lambda it: judge_label(dim, label, it["text"], model=filter_model), items))
    kept = []
    for it, sc in zip(items, scores):
        if sc >= keep_threshold:
            it["judge"] = sc
            kept.append(it)
    return kept
