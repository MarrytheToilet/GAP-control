# GAP-Control Attribute Taxonomy

This is the controlled-attribute design for the paper. It is **not** a random grab-bag:
attributes are organized by their *reward mechanism*, which is itself a contribution —
GAP-Control unifies **soft** (classifier-rewarded) and **hard** (rule-verifiable) control
under one advantage-residual formulation. Every attribute contributes a term to the
reward mixture (handbook 2.3):

```
R_c(y) = Σ_i α_i · R_i(y)  −  μ · R_conflict(y),   R_i(y) ∈ [0, 1]
```

so single-attribute, continuous-intensity, and multi-attribute composition all fall out
of the same machinery.

---

## Family S — Soft semantic attributes  (reward = classifier P(attr | y) ∈ [0,1])

These need a learned reward. We **synthesize + filter** labeled data via an LLM API and
fine-tune a lightweight head on `bge-base-en-v1.5`. They support **continuous intensity**
(the reward is a calibrated probability, not a hard label).

| Dim | 中文 | Classes | Notes |
|-----|------|---------|-------|
| **sentiment** | 情感 | `positive`, `negative`, `neutral` | polarity; pos↔neg is the continuous-intensity axis (RQ4) |
| **emotion**   | 情绪 | `joy`, `anger`, `sadness`, `fear` | Ekman subset; finer affect, richer fusion than polarity |
| **style**     | 风格 | `formal`, `informal`, `literary` | formal↔informal is a second intensity axis; `literary` = vivid/figurative |

→ 3 classifiers, 10 classes total. "情感可以有几种 / 风格也可以有几种" is satisfied by
sentiment+emotion (7 affect classes) and style (3 register classes).

## Family H — Hard structural / lexical attributes  (reward = exact programmatic verifier)

**Zero training, exact reward, language-agnostic.** They work offline immediately and give
the paper a clean "hard constraint" axis that classifier-based CTG baselines handle poorly.

| Dim | 中文 | Spec (parametric) | Reward R_i |
|-----|------|-------------------|------------|
| **length**    | 长度 | bucket `short`(≤25 tok) / `medium`(26–60) / `long`(>60), or target N | 1 in-bucket, else exp decay in distance |
| **keyword**   | 关键词 | required keyword/phrase set W | fraction of W present (case/lemma-insensitive) |
| **structure** | 结构 | `interrogative` / `exclamatory` / `enumeration` / `dialogue` | 1 if pattern satisfied (regex/heuristic) else 0 |

---

## Why this set (paper rationale)

- **Mechanistic coverage.** Soft (semantic, needs a model) vs hard (structural, exact) is
  the cleanest possible axis to show the residual generalizes across reward *types*.
- **Continuous control** lives in Family S (sentiment, formality intensity) — RQ4 monotonicity.
- **Composition / conflict** is natural across families: e.g. `{positive, formal, short,
  contains "ocean"}`, or conflicting `{long, short}` / `{joy, sadness}` to stress R_conflict
  and the gap projection (RQ4 held-out + conflict combinations).
- **Baseline asymmetry.** Expert/discriminator baselines (FUDGE, DExperts) target soft
  attributes; hard constraints expose their limits, while GAP-Control treats both uniformly.

## Experiment mapping (handbook §3.2)

| Group | Uses |
|-------|------|
| A single-attribute | each soft class + each hard pattern |
| B continuous intensity | sentiment (neg↔pos), style (informal↔formal), length target |
| C multi-attribute fusion | soft×soft (sentiment×style, emotion×style), soft×hard (sentiment×length×keyword), held-out & conflict combos |
| D Chinese application | same dims via `bge-base-zh` + tokenizer-based length; structure/keyword are language-agnostic |

## Data strategy for Family S (see `scripts/synth_data.py`)

1. **Synthesize** class-conditioned text with the API (diverse prompts × domains × lengths).
2. **Filter** for label fidelity: (a) round-trip — an LLM judge must re-confirm the label;
   (b) agreement — drop examples where judge and an auxiliary signal disagree;
   (c) dedup + length/format sanity.
3. **Calibrate** for intensity: keep a graded subset (e.g. mild vs strong positive) so the
   classifier probability tracks intensity, not just the decision boundary (handbook 5.3).
4. Fine-tune `bge-base-en-v1.5` + linear head per dimension; held-out accuracy + ECE reported.

The same synthetic pipeline doubles as the source of **control prompts/prefixes** with known
target labels for teacher-advantage estimation.

---

## Constructing hard control (no classifier, no API)

Hard attributes reuse the **same rollout-teacher → advantage** path; the verifier *is* the
reward. The non-obvious construction principles (each learned from experiments):

1. **Reward must have variance** at the training prefixes. Terminal constraints
   (length/structure) need the rollout horizon long enough to actually *reach* a satisfiable
   state, and prefixes sampled near the decision point — otherwise every rollout fails and
   the advantage is identically 0.
2. **The active set must contain the constraint-relevant tokens.** The residual only acts on
   `S_t = TopK(π₀)` (`gap_control/attributes.keyword_active_ids` + the augmentation in
   `teacher.py` / `decoding.py`). A rare keyword like *ocean* is out of the top-k of a small
   model, so it is forced into `S_t` for both teacher estimation and decoding.
3. **Pinpoint lexical control needs a lexical term.** The LM-head-tied residual
   (`b = W_LMᵀ r_t`) does *semantic/distributional* shifts well but cannot spike a single
   rare token without lighting up its neighbors. So keyword control adds a small logit push
   on the forced keyword tokens (`cfg.lexical_strength`), **gated by (1 − observed
   satisfaction)** so it is dynamic. Result: keyword success 0.0 → 0.75 at KL ≈ 0.07.
4. **Length** is a budget constraint: satisfaction depends on position, so the controller
   gets the running length via the state feature (below).

## The dynamic loop (动态)

Control is **not a static bias**. Every step the controller is fed a live running-state
(`gap_control/state.py`, `STATE_DIM=4`: normalized length, length-vs-target, keyword
coverage, structure-satisfied) and the loop runs:

```
V_hat (predicted value)  →  g_t = max(0, R_target − V_hat)  →  budget ρ(g_t)  →  ||b_t||
```

For *monotone* constraints (keyword) the **observed satisfaction** clamps the gap
(`v_eff = max(V_hat, obs_sat)`), so once the keyword appears the gap is provably 0 and the
residual switches off. Verified trajectory (keyword=ocean, one sample):

```
step | ||b|| |  gap  | obs_sat |  KL
  1  | 0.08  | 1.000 |  0.000  | 1.63   strong intervention while unsatisfied
  2  | 0.10  | 0.000 |  1.000  | 0.00   keyword appears → residual off
  3+ |  ..   | 0.000 |  1.000  | 0.00   stays off
```

`scripts/analyze_trajectory.py` plots ||b_t||, g_t, V_hat, obs_sat, KL over steps — the
"证明闭环：未满足时强、满足后弱" diagnostic (handbook §5).
