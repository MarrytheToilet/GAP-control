# GAP-Control

**Gap-Aware Advantage Projection for Controllable Text Generation.**

> Controllable generation is amortized advantage prediction under a KL trust region.
> 可控生成 = 在 KL 信任域约束下，对控制 reward 诱导的 token-level advantage residual 进行摊销预测。

The KL-regularized optimal one-step logit intervention is the reward-induced token-level
**advantage**:

```
pi*(v | s_t, c) ∝ pi_0(v | s_t) · exp( A_c(s_t, v) / tau )
l_t* = l_t^0 + A_c(s_t, ·) / tau
```

GAP-Control trains a lightweight controller to **amortize** this advantage into a single
residual prediction, and projects the residual into a **value-gap-dependent trust region**
so it intervenes only as much as necessary. At inference there is no expert LM, no
candidate-level rollout, no prefix scorer — just one base forward + one tiny controller
forward per step.

## Controlled attributes

See **`ATTRIBUTES.md`** for the full taxonomy and rationale. Two reward classes under one
residual mechanism:

- **Soft** (classifier reward, supports continuous intensity): **sentiment** {positive,
  negative, neutral}, **emotion** {joy, anger, sadness, fear}, **style** {formal, informal,
  literary}. Built from synthetic+filtered data on a `bge-base-en-v1.5` head.
- **Hard** (exact rule verifier, training-free): **length** {short/medium/long or target N},
  **keyword** {required set}, **structure** {interrogative/exclamatory/enumeration/dialogue}.

A control condition is a list of components (`config.control`), so single-attribute,
continuous-intensity, and multi-attribute composition are all the same code path. The reward
is the mixture `R_c = Σ αᵢ Rᵢ − μ·R_conflict`.

## Layout

```
gap_control/            core library
  config.py             Config (YAML); condition-driven control; resolves local model paths
  env.py                .env loader + OpenAI-compatible client + local-model resolution
  base_lm.py            frozen base LM: hidden h_t, logits l0_t, tied W_LM, generate
  attributes.py         registry, ControlCondition, reward mixture R_c (soft+hard)
  verifiers.py          hard attributes: length / keyword / structure (exact, offline)
  classifiers.py        soft attributes: bge-base-en head + SoftClassifierBank
  synth.py              LLM-API synthetic data generation + judge filtering
  rewards.py            build_condition_reward(): wires classifiers/verifiers/judge to R_c
  controller.py         ControlEncoder (attribute-slot mixture) + GapController (r_t, V_hat)
  projection.py         center + L2 / KL gap projection (the trust region)
  teacher.py            rollout teacher: Q -> centered advantage A, value target V*
  decoding.py           GAP-Control decode + prompt-only / FUDGE-like baselines
  metrics.py            control, KL, PPL, distinct-n, bias-advantage Spearman
scripts/                pipeline CLIs (each runnable from repo root)
configs/                sentiment_mvp · multiattr_demo · *_smoke (fast/offline tests)
data/ models/ outputs/  artifacts (handbook §4.4 schema)
```

## Setup

```bash
cp .env.example .env      # then fill GAPCTRL_API_KEY / BASE_URL / MODEL for synthesis
```

Local models are read from `GAPCTRL_MODELS_DIR` (default `/home/hanyu/models`): base LM
`Qwen3-4B`, classifier backbone `bge-base-en-v1.5`. Runs are offline by default
(`HF_HUB_OFFLINE=1`).

## Baselines (handbook §3.4)

All decoding-time, base-only (no extra trained LM), modern, and fair — same base model and
sampler for every method:

| method | family | control signal |
|--------|--------|----------------|
| `prompt` | prompting | none (plain) |
| `instruct` | prompting | natural-language attribute instruction in context |
| `cfg` | contrastive decoding | classifier-free guidance: `l_uncond + γ(l_cond − l_uncond)` |
| `preadd` | expert/anti-expert (DExperts family) | contrastive prompts: `l_base + α(l_pos − l_neg)` |
| `fudge` | discriminator-guided | reward as a future predictor over top-k (inference-time) |
| `bon` | test-time sampling | best-of-N reranked by the reward |
| `gap` | **ours** | amortized advantage residual + gap projection |

`cfg`/`preadd`/`instruct` need no reward at inference; `fudge`/`bon` use the reward at
inference (the slow, inference-time-control points on the efficiency curve).

## Compositional control (arbitrary attributes, unseen combinations)

Handles any number of attributes (many soft, many hard, mixed) and generalizes to
combinations never trained on. Key fact: the advantage is **exactly linear** in the reward
mixture (`A_c = Σαᵢ Aᵢ`), so we

1. **cache per-attribute advantages from one shared rollout set** per prefix
   (`scripts/estimate_teacher_multi.py`) — every atomic's verifier/classifier scored on the
   same completions; any composition's target is then a linear combination (cheap + exact);
2. train on **randomly sampled compositions** with a controller that is **additive +
   interaction** (`CompositionalController`): a structurally-linear pathway that *provably
   composes* to unseen combos, plus a small regularized interaction residual for conflicts;
3. **hold out** combinations and measure generalization (`config.holdout_combos`).

Verified: on a held-out combo the controller composes as `cos(r_combo, mean-of-singles)=1.000`
with a 0.8% interaction term — exact additive generalization by construction.

```bash
python scripts/estimate_teacher_multi.py --config configs/compositional_demo.yaml
python scripts/train_compositional.py     --config configs/compositional_demo.yaml
python scripts/decode_gap_control.py      --config configs/compositional_demo.yaml --methods gap,prompt,cfg,preadd
```

## Multi-model synthesis (role separation)

So no model grades its own output (`.env`):
`GAPCTRL_GEN_MODELS` (rotated generators) → `GAPCTRL_FILTER_MODEL` (data QA, distinct) →
`GAPCTRL_JUDGE_MODEL` (reward/eval, distinct again). Seeds are a balanced
genre×topic×length×register grid (`seeds.py`) so each class spans many genres/registers.

## Building soft-attribute classifiers (needs the API)

```bash
python scripts/synth_data.py --dims sentiment,emotion,style --per-class 300   # generate+filter
python scripts/train_classifier.py --dim sentiment   # bge head; reports acc + ECE
python scripts/train_classifier.py --dim emotion
python scripts/train_classifier.py --dim style
```

Without trained classifiers, soft dims auto-fall back to an offline LLM judge. Hard dims need
nothing — try them offline now: `bash scripts/run_mvp.sh configs/structure_smoke.yaml`.

## Pipeline (handbook §4.1, the 7 steps)

```bash
# fast smoke test of the whole thing (a few minutes on a 3090)
bash scripts/run_mvp.sh configs/sentiment_smoke.yaml

# the real MVP run
bash scripts/run_mvp.sh configs/sentiment_mvp.yaml
```

Or stage by stage:

```bash
python scripts/build_prefixes.py            --config configs/sentiment_mvp.yaml  # 1. prefixes
python scripts/estimate_teacher_advantage.py --config configs/sentiment_mvp.yaml # 2. teacher A
python scripts/train_controller.py          --config configs/sentiment_mvp.yaml  # 3. train
python scripts/decode_gap_control.py        --config configs/sentiment_mvp.yaml --methods gap,prompt,fudge  # 4. decode
python scripts/evaluate.py                  --config configs/sentiment_mvp.yaml  # 5. main table
```

## What to look at (MVP pass criteria, handbook §4.2)

1. **Teacher has signal** — `estimate_teacher_advantage` prints `A std`; should be > 0.
2. **Controller learns advantage** — `train_controller` prints **bias-advantage Spearman**;
   target **> 0.3**. This is the core "did it amortize the advantage" check.
3. **Control + low perturbation** — `evaluate` table: GAP-Control reward/success above
   prompt-only, ideally at lower KL / similar PPL; faster than FUDGE-like (ms/token).
4. **Gap projection matters** — set `projection: none` and confirm KL/PPL get worse.

## Notes

- Reward backend falls back to an offline LLM-judge (cached `Qwen2.5-0.5B-Instruct`) if the
  HF sentiment classifier can't be downloaded.
- The controller checkpoint is tiny: it reuses the **frozen base LM head** `W_LM`, so the
  residual lives in the base model's output space (low-perturbation by construction).
- Next milestones (handbook §4.3): MVP-1 value teacher + efficiency; Full-1 continuous
  intensity (`alpha` sweep); Full-2 held-out multi-attribute composition.
```
