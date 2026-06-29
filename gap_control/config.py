"""Configuration for GAP-Control experiments.

A single dataclass loaded from / dumped to YAML so every pipeline stage reads the same
knobs. The control target is now a *condition* (handbook 2.3): a list of attribute
components, supporting single-attribute, continuous-intensity, and multi-attribute
composition through the same field. Defaults follow handbook section 4.5.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import List

import yaml

from .env import resolve_model
from .attributes import Component, ControlCondition, NUM_SLOTS


@dataclass
class Config:
    # ---- identity ----
    task: str = "sentiment"

    # ---- control condition (list of components) ----
    # each item: {dim, value?, alpha?, length_target?, keywords?}
    control: list = field(default_factory=lambda: [
        {"dim": "sentiment", "value": "positive", "alpha": 1.0}])
    aggregate: str = "mean"            # mean | min | gmean (handbook 3.8)
    conflict_mu: float = 0.0           # R_conflict weight

    # ---- compositional control (multiple/arbitrary attributes) ----
    compositional: bool = False
    atomics: list = field(default_factory=list)   # pool of atomic attrs to cache (specs)
    max_attrs: int = 3                 # max attributes per sampled training composition
    compositions_per_prefix: int = 8   # random compositions sampled per prefix per epoch
    interaction: bool = True           # enable the interaction-correction pathway (ablatable)
    interaction_lambda: float = 0.1    # regularizer on the interaction residual
    gate_lambda: float = 0.0           # L2 penalty on the CAA gate (keeps CAA off unless it helps)
    holdout_combos: list = field(default_factory=list)  # atomic-id sets withheld from training
    adv_weight: bool = False           # weight L_adv per example by max|A| (focus on controllable states)
    use_steering: bool = False         # CAA directions enrich the control vector
    controller_type: str = "compositional"  # compositional | lmsteer | static_caa (baselines)
    linear_steer_rank: int = 0         # >0: add LM-Steer low-rank linear backbone to GAP-Control
    steer_rank: int = 8                # rank for the lmsteer baseline

    # ---- models ----
    base_model: str = "Qwen3-4B"       # resolved against GAPCTRL_MODELS_DIR if local
    backbone: str = "bge-base-en-v1.5"  # soft-classifier encoder
    classifier_dir: str = "models/classifier"
    language: str = "en"               # en | zh (affects backbone choice for classifiers)
    device: str = "cuda"
    dtype: str = "float16"

    # ---- soft-attribute reward fallback (when no trained classifier) ----
    # "classifier" (use trained bank), "llm_judge" (offline cached LM judge)
    soft_reward_backend: str = "classifier"
    reward_judge_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # ---- prefixes ----
    num_prompts: int = 20
    prefixes_per_prompt: int = 8
    prefix_min_len: int = 3
    prefix_max_len: int = 25
    prefix_gen_max_new: int = 30

    # ---- teacher advantage ----
    teacher: str = "rollout"
    topk: int = 50
    rollout_samples: int = 4
    rollout_max_new: int = 20
    rollout_temperature: float = 1.0
    rollout_batch_size: int = 256

    # ---- controller ----
    mid_layer_frac: float = 0.0        # >0: concat a mid-layer hidden (attrs more linearly decodable mid-net)
    control_dim: int = 64
    fuse_hidden: int = 512
    controller_dropout: float = 0.1

    # ---- training objective ----
    tau: float = 1.0
    value_lambda: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 20
    batch_size: int = 64
    grad_clip: float = 1.0
    val_frac: float = 0.1
    seed: int = 0

    # ---- gap projection ----
    projection: str = "l2"
    rho_min: float = 0.0
    rho_max: float = 2.0
    reward_target: float = 1.0
    kl_budget_max: float = 1.0
    # lexical term for pinpoint keyword control: a logit push on the forced keyword tokens,
    # gated by (1 - observed_satisfaction) so it is dynamic (off once the keyword appears).
    # The LM-head-tied residual handles semantics; this handles rare-token insertion.
    lexical_strength: float = 0.0
    length_eos_scale: float = 12.0     # EOS-budget bias strength for hard length control
    decode_strength: float = 1.0       # global multiplier on the residual at decode (LM-Steer-style ε knob)
    fudge_interval: int = 3            # FUDGE baseline: run the expensive candidate-scoring every k tokens (speedup)
    entropy_gate: float = 0.0          # >0: steer more where base is uncertain (gate=(1-p_max)^? ), preserves fluency on confident tokens
    gate_gamma: float = 0.0            # projection="gate": exponent on base-uncertainty u_t in the unified budget rho_t=(rho_min+rho_max*g_t)*u_t^gamma

    # ---- decoding ----
    gen_max_new: int = 40
    gen_top_p: float = 0.9
    gen_temperature: float = 0.7
    decode_topk: int = 50
    num_eval_prompts: int = 0
    samples_per_prompt: int = 1

    # ---- paths ----
    data_dir: str = "data"
    models_dir: str = "models"
    out_dir: str = "outputs"

    # ---------- derived ----------
    def resolved_base_model(self) -> str:
        return resolve_model(self.base_model)

    def condition(self) -> ControlCondition:
        comps = [Component(**c) for c in self.control]
        return ControlCondition(comps, aggregate=self.aggregate, conflict_mu=self.conflict_mu)

    def hidden_fracs(self) -> tuple:
        """Layer fractions fed to the controller (1.0 = last). Multi-layer if mid_layer_frac>0."""
        return (self.mid_layer_frac, 1.0) if self.mid_layer_frac > 0 else (1.0,)

    def soft_dims(self) -> List[str]:
        from .attributes import SOFT_DIM_SET
        return [c["dim"] for c in self.control if c["dim"] in SOFT_DIM_SET]

    @property
    def num_slots(self) -> int:
        return NUM_SLOTS

    def prompts_path(self) -> str:
        return f"{self.data_dir}/prompts/{self.task}.jsonl"

    def prefixes_path(self) -> str:
        return f"{self.data_dir}/prefixes/{self.task}.jsonl"

    def teacher_path(self) -> str:
        return f"{self.data_dir}/teacher/{self.task}/topk_advantage.pt"

    def controller_path(self) -> str:
        return f"{self.models_dir}/controller/{self.task}/checkpoint.pt"

    def steering_path(self) -> str:
        import re
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self.base_model)
        return f"{self.models_dir}/steering/{safe}.pt"

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, allow_unicode=True)
