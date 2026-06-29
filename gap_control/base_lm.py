"""Frozen base language model wrapper.

Exposes exactly what GAP-Control needs from pi_0:
  * `step(...)`  -> last-layer hidden state h_t and logits l0_t at the final position
  * `lm_head`    -> the tied output projection W_LM, reused by the controller so the
                    residual lives in the base model's semantic output space (handbook 2.4)
  * `generate(...)` -> plain sampling, used both for prefix construction and rollouts

The base model is never trained; everything runs under torch.no_grad in eval mode.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env import resolve_model


_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class BaseLM:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "float16"):
        self.device = device
        model_name = resolve_model(model_name)   # use local copy if present
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # left padding so the "last real token" is always at position -1 for batched steps
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=_DTYPES.get(dtype, torch.float16)
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def lm_head(self) -> torch.nn.Module:
        """The tied output projection W_LM (frozen). b_raw = lm_head(r_t)."""
        return self.model.get_output_embeddings()

    def _pick_hidden(self, hidden_states, fracs):
        """Concat last-token hidden over the given layer fractions (1.0 = last layer)."""
        N = len(hidden_states) - 1                            # index 0 = embeddings
        parts = [hidden_states[max(1, round(f * N))][:, -1, :] for f in fracs]
        return torch.cat(parts, dim=-1).float()              # [B, H*len(fracs)]

    @torch.no_grad()
    def step(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
             hidden_fracs=(1.0,)):
        """One forward pass. Returns (hidden, logits_last) at the final position.
        hidden = concat of last-token hidden over `hidden_fracs` layers (fp32)."""
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h = self._pick_hidden(out.hidden_states, hidden_fracs)
        logits_last = out.logits[:, -1, :].float()            # [B, V]
        return h, logits_last

    @torch.no_grad()
    def step_cached(self, input_ids: torch.Tensor, past=None, need_hidden: bool = True,
                    hidden_fracs=(1.0,), attention_mask: Optional[torch.Tensor] = None):
        """KV-cached forward. First call: pass the full prefix (past=None). Subsequent calls:
        pass only the new token + the returned past. Returns (hidden, logits_last, past).
        hidden = concat over `hidden_fracs` layers. Set need_hidden=False for plain sampling."""
        out = self.model(input_ids=input_ids, past_key_values=past,
                         use_cache=True, output_hidden_states=need_hidden)
        h = self._pick_hidden(out.hidden_states, hidden_fracs) if need_hidden else None
        logits_last = out.logits[:, -1, :].float()
        return h, logits_last, out.past_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
        do_sample: bool = True,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample continuations. Returns the full sequences [B, T+new]."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)
