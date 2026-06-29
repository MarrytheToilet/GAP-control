"""Environment / .env loading and a thin OpenAI-compatible client factory.

Keeps API credentials out of code and configs. `load_env()` reads `.env` from the repo
root (idempotent); `get_client()` returns a configured OpenAI client for synthesis.
"""
from __future__ import annotations

import os
from functools import lru_cache

_LOADED = False


def load_env() -> None:
    global _LOADED
    if _LOADED:
        return
    try:
        from dotenv import load_dotenv
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_dotenv(os.path.join(root, ".env"))
    except Exception:
        pass
    # default to offline HF so we never accidentally hit the hub
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _LOADED = True


def models_dir() -> str:
    load_env()
    return os.environ.get("GAPCTRL_MODELS_DIR", "/home/hanyu/models")


def resolve_model(name_or_path: str) -> str:
    """If `name_or_path` is a bare model name and exists under GAPCTRL_MODELS_DIR, use the
    local copy; otherwise return it unchanged (HF id / absolute path)."""
    load_env()
    if os.path.isabs(name_or_path) or os.path.exists(name_or_path):
        return name_or_path
    local = os.path.join(models_dir(), name_or_path)
    return local if os.path.exists(local) else name_or_path


@lru_cache(maxsize=4)
def get_client(base_url: str | None = None, api_key: str | None = None):
    load_env()
    from openai import OpenAI
    return OpenAI(
        api_key=api_key or os.environ.get("GAPCTRL_API_KEY", "EMPTY"),
        base_url=base_url or os.environ.get("GAPCTRL_BASE_URL", "https://api.openai.com/v1"),
    )


def synth_settings() -> dict:
    """Role-separated model assignment (no model grades its own output):
      * gen_models  -- pool of generators for classifier/control data (rotated for diversity)
      * filter_model-- judges fidelity of *generated training data* (distinct from generators)
      * judge_model -- LLM-as-judge reward + evaluation (distinct again)
    """
    load_env()
    default = os.environ.get("GAPCTRL_MODEL", "") or "gpt-4o-mini"
    gen = os.environ.get("GAPCTRL_GEN_MODELS", "").strip()
    gen_models = [m.strip() for m in gen.split(",") if m.strip()] or [default]
    return {
        "gen_models": gen_models,
        "filter_model": os.environ.get("GAPCTRL_FILTER_MODEL", "").strip() or default,
        "judge_model": os.environ.get("GAPCTRL_JUDGE_MODEL", "").strip() or default,
        "temperature": float(os.environ.get("GAPCTRL_SYNTH_TEMPERATURE", "1.0")),
        "max_tokens": int(os.environ.get("GAPCTRL_SYNTH_MAX_TOKENS", "256")),
        "concurrency": int(os.environ.get("GAPCTRL_SYNTH_CONCURRENCY", "8")),
    }
