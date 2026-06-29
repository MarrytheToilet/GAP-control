"""Control rewards R_c(y).

The teacher advantage is only as stable as the reward (handbook 5.1 / 5.3), so the
reward is a first-class, pluggable component returning a scalar in [0, 1] per text.

Two backends:
  * HFClassifierReward  -- a sentiment text-classification model (default).
                           Downloads `reward_model` on first use.
  * LLMJudgeReward      -- fully offline fallback using a cached instruct model that
                           emits a yes/no judgement; reward = P(yes).

`build_reward(cfg)` picks one from the config. Both expose `score(texts) -> [N] tensor`.
For continuous-intensity work later, return calibrated probabilities, not hard labels.
"""
from __future__ import annotations

from typing import List

import torch


class Reward:
    """Interface: map a list of texts to a [N] float tensor of rewards in [0, 1]."""

    name: str = "reward"

    def score(self, texts: List[str]) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class HFClassifierReward(Reward):
    def __init__(self, model_name: str, positive_label: str = "POSITIVE",
                 device: str = "cuda", batch_size: int = 64):
        from transformers import (AutoModelForSequenceClassification,
                                   AutoTokenizer)
        self.name = f"hf:{model_name}"
        self.device = device
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        # map the configured positive label to a column index
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        label2id = {v.upper(): k for k, v in id2label.items()}
        if positive_label.upper() in label2id:
            self.pos_idx = label2id[positive_label.upper()]
        else:  # binary fallback: assume the higher index is "positive"
            self.pos_idx = max(id2label)

    @torch.no_grad()
    def score(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0)
        out = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [t if t.strip() else " " for t in texts[i:i + self.batch_size]]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to(self.device)
            probs = self.model(**enc).logits.softmax(-1)
            out.append(probs[:, self.pos_idx].float().cpu())
        return torch.cat(out)


class LLMJudgeReward(Reward):
    """Offline reward: P(the judge answers 'Yes' to 'is this <attribute>?')."""

    def __init__(self, model_name: str, attribute: str = "positive in sentiment",
                 device: str = "cuda", batch_size: int = 16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = f"judge:{model_name}"
        self.device = device
        self.batch_size = batch_size
        self.attribute = attribute
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16).to(device)
        self.model.eval()
        # token ids for " Yes" / " No" (first subword); robust across tokenizers
        self.yes_id = self.tok(" Yes", add_special_tokens=False).input_ids[0]
        self.no_id = self.tok(" No", add_special_tokens=False).input_ids[0]

    def _prompt(self, text: str) -> str:
        msg = (f"Text: {text!r}\n"
               f"Question: Is this text {self.attribute}? Answer Yes or No.")
        chat = [{"role": "user", "content": msg}]
        return self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def score(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0)
        out = []
        for i in range(0, len(texts), self.batch_size):
            prompts = [self._prompt(t if t.strip() else " ") for t in texts[i:i + self.batch_size]]
            enc = self.tok(prompts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            logits = self.model(**enc).logits[:, -1, :]
            yn = logits[:, [self.yes_id, self.no_id]].softmax(-1)
            out.append(yn[:, 0].float().cpu())
        return torch.cat(out)


class JudgeBank:
    """Adapter exposing `prob(dim, value, texts)` via offline LLM-judge, so soft attributes
    work without trained classifiers (slower; for bootstrap / fully-offline runs)."""

    def __init__(self, model_name: str, device: str = "cuda"):
        from .attributes import SOFT_DIMS  # noqa
        from .synth import DESCRIPTIONS
        self.device = device
        self.model_name = model_name
        self.desc = DESCRIPTIONS
        self._cache = {}

    def _judge(self, dim, value):
        key = (dim, value)
        if key not in self._cache:
            attr = self.desc.get(key, f"{value} in {dim}")
            self._cache[key] = LLMJudgeReward(self.model_name, attribute=attr,
                                              device=self.device)
        return self._cache[key]

    def prob(self, dim: str, value: str, texts):
        return self._judge(dim, value).score(texts)


def build_condition_reward(cfg, tokenizer=None):
    """Return (reward_fn, condition, bank) for the configured control condition.

    reward_fn(texts) -> [N] reward in [0,1] = R_c(y) (handbook 2.3). Soft dims use the
    trained classifier bank, or an offline LLM-judge fallback; hard dims use exact verifiers.
    """
    condition = cfg.condition()
    soft_dims = cfg.soft_dims()
    bank = None
    if soft_dims:
        if cfg.soft_reward_backend == "classifier":
            from .classifiers import SoftClassifierBank
            bank = SoftClassifierBank(cfg.classifier_dir, cfg.backbone, cfg.device)
            missing = [d for d in soft_dims if d not in bank.available()]
            if missing:
                print(f"[rewards] no trained classifier for {missing}; "
                      f"falling back to llm_judge")
                bank = JudgeBank(cfg.reward_judge_model, cfg.device)
        else:
            bank = JudgeBank(cfg.reward_judge_model, cfg.device)

    def reward_fn(texts):
        return condition.reward(texts, bank=bank, tokenizer=tokenizer)

    return reward_fn, condition, bank


def build_reward(cfg) -> Reward:
    if getattr(cfg, "reward_backend", None) == "hf_classifier":
        try:
            return HFClassifierReward(cfg.reward_model, cfg.reward_positive_label, cfg.device)
        except Exception as e:  # offline / download failure -> fall back to judge
            print(f"[rewards] hf_classifier unavailable ({e!r}); falling back to llm_judge")
            return LLMJudgeReward(cfg.reward_judge_model,
                                  attribute=f"{cfg.target_attribute} in sentiment",
                                  device=cfg.device)
    if cfg.reward_backend == "llm_judge":
        return LLMJudgeReward(cfg.reward_judge_model,
                              attribute=f"{cfg.target_attribute} in sentiment",
                              device=cfg.device)
    raise ValueError(f"Unknown reward_backend: {cfg.reward_backend}")
