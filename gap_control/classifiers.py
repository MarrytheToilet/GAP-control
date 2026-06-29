"""Soft-attribute classifiers (Family S): bge-base-en backbone + per-dim linear head.

One small classifier per soft dimension (sentiment 3-way, emotion 4-way, style 3-way),
trained on synthetic+filtered data. The reward for a (dim, class) is the softmax
probability of that class -> a calibrated, continuous signal that supports intensity
control (handbook 5.3).

`SoftClassifierBank` loads whatever trained classifiers exist and exposes
`prob(dim, value, texts) -> [N]`, the interface attributes.Component expects.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .env import resolve_model
from .attributes import SOFT_DIMS


class SoftClassifier(nn.Module):
    """bge-base-en encoder (CLS pooling) + linear classification head."""

    def __init__(self, backbone: str, classes: List[str]):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        path = resolve_model(backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.encoder = AutoModel.from_pretrained(path)
        self.classes = list(classes)
        self.head = nn.Linear(self.encoder.config.hidden_size, len(classes))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]          # CLS pooling (bge convention)
        return self.head(cls)                       # [B, C] logits

    @torch.no_grad()
    def predict_proba(self, texts: List[str], device="cuda", batch_size=64) -> torch.Tensor:
        self.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [t if t.strip() else " " for t in texts[i:i + batch_size]]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                 truncation=True, max_length=128).to(device)
            out.append(self(enc.input_ids, enc.attention_mask).softmax(-1).cpu())
        return torch.cat(out)


class SoftClassifierBank:
    """Loads trained per-dim classifiers from `models_dir/{dim}.pt`."""

    def __init__(self, models_dir: str, backbone: str, device: str = "cuda"):
        self.device = device
        self.backbone = backbone
        self.models: Dict[str, SoftClassifier] = {}
        self.models_dir = models_dir
        for dim, classes in SOFT_DIMS.items():
            path = os.path.join(models_dir, f"{dim}.pt")
            if os.path.exists(path):
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                clf = SoftClassifier(backbone, ckpt["classes"])
                clf.load_state_dict(ckpt["state_dict"])
                clf.to(device)
                self.models[dim] = clf

    def available(self) -> List[str]:
        return list(self.models)

    def prob(self, dim: str, value: str, texts: List[str]) -> torch.Tensor:
        if dim not in self.models:
            raise KeyError(f"no trained classifier for dim {dim!r}; "
                           f"available: {self.available()}. Run scripts/train_classifier.py")
        clf = self.models[dim]
        probs = clf.predict_proba(texts, self.device)       # [N, C]
        idx = clf.classes.index(value)
        return probs[:, idx]
