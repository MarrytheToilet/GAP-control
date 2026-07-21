<div align="center">

# GAP-Control

### Composition as Supervision — Amortized Multi-Attribute Control via Shared-Rollout Advantage Distillation

<img src="assets/pr.png" alt="GAP-Control overview: train once, control any signed composition at ~1.2x base cost" width="720">

*One offline rollout pass caches every attribute's token-level advantage. Any signed mixture is then an exact linear combination **of the training targets**, distilled into a single-pass controller that decodes at ≈1.2× base cost.*

</div>

---

## TL;DR

Real controllable-generation requests are **compositions**: *"positive, formal, short, mentions ocean."* Amortized steering (LM-Steer) is cheap but **saturates** under composition; inference-time search (FUDGE, Best-of-N) is reward-grounded but **re-runs per target**.

We place the composition problem in a third location: **the supervision**. When a linear reward mixture is scored on one *shared* rollout, the base-policy token-level advantage is exactly linear in the mixture weights — so a single offline pass yields an unbiased distillation target for *every* signed composition at once. A lightweight controller distills this cache and a **value-gap gate** sizes each intervention.

The analysis part asks what the controller actually learns. The cached targets are pointwise **noise-dominated** (two independent estimates barely agree) and using the cache directly by retrieval fails, yet the distilled controller reaches the cache's information limit: **amortization acts as a denoiser, not a compromise**. This predicts — and we confirm — that control is noise-limited wherever supervision reaches: more rollouts buy more control on covered states, while the doubly novel corner (unseen composition × unseen states) stays generalization-limited.

---

## The key identity

The KL-regularized optimal one-step intervention is the reward-induced token-level **advantage**:

$$\pi^\star(v \mid s_t, c) \;\propto\; \pi_0(v \mid s_t)\,\exp\!\big(A_c(s_t,v)/\tau\big).$$

When the reward is a linear mixture $R_c = \sum_i \alpha_i R_i$ evaluated on the **same** continuations, linearity of expectation gives the exact decomposition

$$A_c \;=\; \sum_i \alpha_i A_i , \qquad V_c^\star \;=\; \sum_i \alpha_i V_i^\star ,$$

so one shared-rollout pass per prefix caches the per-attribute advantages $\{\hat A_i\}$ of **all** attributes at once, and the distillation target for **any** condition $c=\{(a_i,\alpha_i)\}$ — including signed $\alpha_i<0$ (suppression) — is a free linear combination of the cache. Exactness lives in the *training targets*; no per-request rollout, training, or search.

---

## Method: three stages

<div align="center">
<img src="assets/framework.png" width="100%" alt="GAP-Control architecture: cache once, compose in supervision, amortize online">
</div>

**1 · Cache once** *(offline)* — From each prefix, one shared rollout set ($n$ continuations per top-$K$ candidate) scores *every* atomic reward on the *same* continuations:

$$\hat A_i(s_t,v) = \hat Q_i(s_t,v) - \sum_{u\in S_t}\pi_0(u\mid s_t)\,\hat Q_i(s_t,u), \qquad \hat Q_i(s_t,v)=\tfrac1n\sum_{j} R_i(y^{(j)}).$$

**2 · Compose in supervision** *(free)* — Any signed mixture's target is $\hat A_c = \sum_i \alpha_i \hat A_i$ by the identity above. Training samples random 1–3-attribute signed compositions per prefix, so the controller sees composition *as supervision*.

**3 · Amortize online** *(one pass)* — A small controller $f_\theta(h_t, v(c))$ predicts a tied logit residual $b_t = W_{\mathrm{LM}} r_t$ and a value $\widehat V_t$ from the frozen base hidden state and the control vector $v(c)=\sum_i \alpha_i e_i$. A **value-gap gate** sizes each step:

$$\rho_t = \big(\rho_{\min} + \rho_{\max}\,g_t\big)\,u_t^{\gamma}, \qquad g_t = \max\!\big(0,\,R_{\text{target}} - \widehat V_t\big), \qquad u_t = 1 - \max_v \pi_0(v\mid s_t),$$

applied centered and set-to-norm: $b_t \leftarrow \rho_t\,\bar b_t / \lVert \bar b_t \rVert$. The controller supplies the direction, the gate the magnitude. Online cost: **one base forward + one tiny controller forward per token** (≈1.2× base). Verifier-defined attributes (length, keyword, structure) are handled by a training-free logit overlay, kept in the appendix.

<div align="center">
<img src="assets/gate.png" width="66%" alt="Value-gap gate validation">
<br><sub>The gate is adaptive, not a constant knob: the predicted value tracks final attribute relevance (left), and the intervention norm shrinks ~53% over decoding as the value gap closes (right).</sub>
</div>

---

## Results

On **base** (non-instruction-tuned) LMs, where prompting fails, GAP-Control leads decoding-time control on single *and* compositional attributes. Falcon3-3B-Base, 116 held-out prompts (zero training overlap), 95% prompt-clustered bootstrap CIs; `Judge` is the independent three-judge panel:

| Method | Rel. ↑ | Succ. ↑ | Judge ↑ | Seen pair | **Unseen pair** | **Unseen triple** | PPL ↓ | ×base |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| base | 0.29 | 0.27 | 0.54 | 0.15 | 0.11 | 0.09 | 4.46 | 1.0× |
| prompting | 0.42 | 0.40 | 0.60 | 0.35 | 0.32 | 0.24 | 6.08 | 1.0× |
| PREADD | 0.50 | 0.49 | 0.72 | 0.40 | 0.29 | 0.26 | **4.23** | 3.0× |
| LM-Steer *(tuned, rank-256)* | 0.76 | 0.77 | 0.74 | 0.50 | 0.44 | 0.38 | 7.60 | 2.0× |
| FUDGE *(tuned)* | 0.67 | 0.67 | 0.66 | 0.37 | 0.35 | 0.28 | 8.49 | 7.8× |
| Best-of-8 | 0.74 | 0.75 | 0.72 | 0.50 | 0.50 | 0.43 | 4.17 | 7.8× |
| **GAP-Control** (n=12 cache) | 0.82 | 0.84 | 0.83 | 0.60 | **0.54** | 0.48 | 6.55 | **1.2×** |
| **GAP-Control** (n=48 cache) | **0.86** | **0.88** | **0.88** | **0.70** | **0.54** | **0.50** | 6.42 | **1.2×** |

Same pattern on SmolLM2-1.7B (GAP 0.80/0.82 single, 0.39/0.40 unseen pair/triple vs LM-Steer 0.35/0.26). GAP cells are stable across three decoding seeds (std ≤ 0.013 on all compositional columns).

<div align="center">
<img src="assets/dashboard.png" width="100%" alt="Per-sample results dashboard">
<br><sub>Per-sample view (~2k generations). Left: on the <b>unseen</b> pair, GAP moves probability mass into the jointly-satisfying quadrant. Middle: full score distributions on the single-attribute task. Right: control is <b>specific</b> — targeting an attribute (row) moves that attribute most.</sub>
</div>

**What survives distillation (the mechanism).** Two independent teacher estimates agree only cos 0.30 / top-1 0.56 — the targets are noise-dominated — and kNN retrieval from the cache is *worse than no intervention*. Yet the distilled controller matches an independent teacher re-estimate: it sits at the cache's information limit. At decision level, the teacher flips the argmax on 44.7% of (state, attribute) decisions — 12% in the lowest uncertainty quartile rising to 65% in the highest — and the controller reaches 82% of the reproducible flip-capture ceiling (n=48 cache).

<div align="center">
<img src="assets/mechanism.png" width="55%" alt="Per-decision mechanism scatter">
<img src="assets/scaling.png" width="43%" alt="Rollout scaling: noise-limited vs generalization-limited">
<br><sub>Left: all 1160 held-out (state, attribute) decisions at their true (uncertainty, direction-fit) coordinates — flips concentrate at high $u_t$, where the gate intervenes. Right: growing the cache $n{=}12\to48$ buys control wherever supervision reaches; the doubly novel corner does not move.</sub>
</div>

**Frontier & cost.** At matched perplexity GAP dominates the control–fluency frontier of a swept LM-Steer; the gate (γ>0) beats every fixed-strength (gate-off) operating point. One controller also does signed suppression zero-shot ($P(\text{positive})$: 0.06 → 0.82 as $\alpha:-1\to+1$).

<div align="center">
<img src="assets/frontier_ho2.png" width="49%" alt="Control-fluency frontier, unseen pair">
<img src="assets/frontier_tri.png" width="49%" alt="Control-fluency frontier, triple">
</div>

<div align="center">
<img src="assets/results.png" width="53%" alt="Signed control sweep and control-perturbation frontier">
<img src="assets/pareto.png" width="38%" alt="Control vs latency">
<br><sub>Left: one controller spans signed control, suppression ($\alpha=-1$) to amplification ($\alpha=+1$). Right: GAP sits in the cheap-and-controllable corner — above inference-time search on control, ~7× faster.</sub>
</div>

**CompMCTG benchmark** (Zhong et al., ACL 2024; official RoBERTa evaluators, disjoint from our reward family):

| Method | Yelp Acc ↑ | Yelp joint ↑ | Fyelp Acc ↑ | Fyelp Δcomp |
|---|:--:|:--:|:--:|:--:|
| prompting | 0.53 | 0.16 | 0.51 | −0.00 |
| LM-Steer | 0.70 | 0.35 | 0.52 | −0.00 |
| **GAP-Control** | **0.77** | **0.42** | **0.60** | **−0.01** |

Near-zero degradation (Δcomp) from seen to unseen compositional splits — the composition-as-supervision property transfers.

---

## Controlled attributes

- **Soft** (classifier reward, continuous intensity): **sentiment** {positive, negative, neutral}, **emotion** {joy, anger, sadness, fear}, **style** {formal, informal, literary} — a `bge-base-en-v1.5` head trained on synthesized, judge-filtered text, then frozen.
- **Hard** (exact rule verifier, training-free overlay): **length**, **keyword**, **structure** {interrogative / exclamatory / enumeration / dialogue}.

A control condition is a list of `(attribute, weight)` components, so single-attribute, continuous-intensity, signed, and multi-attribute composition are one code path. Attribute presets for other datasets (Yelp/Fyelp/Amazon) are selected via `GAPCTRL_ATTRS`.

---

## Setup

```bash
cp .env.example .env      # fill GAPCTRL_API_KEY / BASE_URL / MODEL (synthesis & LLM-judge only)
pip install torch transformers scikit-learn pyyaml numpy
```

Local models are read from `GAPCTRL_MODELS_DIR` (base LM, `bge-base-en-v1.5` backbone); runs are offline by default (`HF_HUB_OFFLINE=1`). The API is needed **only** to synthesize classifier data and run the LLM judges — decoding and evaluation are fully local. All headline experiments fit on a single RTX 3090 (24 GB); the full n=12 cache builds in ≈23 min, n=48 in ≈92 min.

## Reproducing the paper

**0 · Frozen classifiers** (one-time, needs the API for synthesis):

```bash
python scripts/synth_data.py       --dims sentiment,emotion,style --per-class 400
python scripts/train_classifier.py --dim sentiment                # repeat per dimension
```

**1 · Main pipeline** (Falcon3-3B-Base; `configs/flc_multi.yaml` = n=12 cache, `configs/flc_multi_n48.yaml` = n=48):

```bash
python scripts/build_prefixes.py         --config configs/flc_multi_n48.yaml  # prefixes (states)
python scripts/estimate_teacher_multi.py --config configs/flc_multi_n48.yaml  # shared-rollout advantage cache
python scripts/train_compositional.py    --config configs/flc_multi_n48.yaml  # distill controller + value head
bash   scripts/run_n48_main.sh                                                # GAP cells of the main table (2 seeds)
python scripts/evaluate.py               --config configs/flc_multi_n48.yaml  # classifier metrics + PPL + CIs
```

Baselines (prompting / PREADD / FUDGE / Best-of-8 / LM-Steer with `synth_pairs.py` + `compute_steering.py`) and the SmolLM2 / Qwen2.5 suites are driven end-to-end by `scripts/run_multimodel.sh` and `scripts/run_qwen.sh`. `scripts/evaluate_std.py` adds the standardized CTG metrics (generated-text PPL, Dist-1/2/3).

**2 · Independent judges** (main-table `Judge` column): `python scripts/judge_matrix.py` — three LLM judges of distinct families rate every (model, method, setting) cell; raw ratings are persisted for threshold-free reanalysis (`scripts/judge_perattr.py` for the per-attribute view).

**3 · Mechanism & scaling analyses** (Sec. 5 of the paper):

```bash
python scripts/noise_ceiling.py        # MC noise ceiling: two independent teacher estimates
python scripts/knn_pilot.py            # retrieval pilot: direct cache use fails
python scripts/fidelity_check.py       # controller-vs-cache fidelity: atomic / seen / unseen
python scripts/mechanism_analysis.py --ckpt models/controller/flc_multi_n48/full.pt \
       --dump-json paper_mech.json     # argmax-flip capture by uncertainty quartile + ceiling
```

The n-scaling law uses the `flc_multi_p*`/`flc_multi_n48_seed*` configs (prefix count, rollout budget, seeds); the gate/frontier sweep decodes the `_gt_g*_rt*_rm*` configs against gate-off `abl_nogate*`.

**4 · CompMCTG** (clone [CG4MCTG](https://github.com/tqzhong/CG4MCTG) with its data + official evaluators into `third_party/`, gitignored):

```bash
python scripts/compmctg_prep.py  --dataset Yelp        # their data -> our classifier/prompt format
python scripts/compmctg_run.py   --dataset Yelp        # cache + controller per split, decode all combos
python scripts/compmctg_score.py --dataset Yelp --evaluator official   # their RoBERTa evaluators
```

**5 · Figures** — `paper/scripts/` regenerates every paper figure from `outputs/` decodes (shared style in `figstyle.py`): `make_dashboard.py`, `make_mechanism.py` (reads the `mechanism_analysis.py` JSON dump), `make_scaling_fig.py`, `make_frontier.py`, `make_figures.py` (control plane), `plot_figs.py` (signed + pareto, via `dump_figdata.py`), `make_grid.py` (all-pairs/triples table).

## Repository layout

```
gap_control/         core library
  attributes.py        registry + presets (GAPCTRL_ATTRS), ControlCondition, reward mixture R_c
  teacher.py           shared-rollout teacher: Q -> centered advantage A, value target V*
  controller.py        control encoder (attribute-slot mixture) + gated controller (r_t, V̂)
  projection.py        center + value-gap gate (the magnitude budget)
  decoding.py          GAP-Control decode + prompting / FUDGE / best-of-N baselines
  classifiers.py       soft attributes: bge head + classifier bank
  verifiers.py         hard attributes: length / keyword / structure (exact, offline)
  rewards.py           wires classifiers / verifiers / judge into R_c
  synth.py judge.py    LLM-API synthetic data generation + judge filtering
  base_lm.py metrics.py config.py env.py
scripts/             pipeline CLIs
  synth_data.py synth_pairs.py synth_prompts.py train_classifier.py   data + frozen classifiers
  build_prefixes.py estimate_teacher_multi.py                         prefixes -> shared-rollout cache
  compute_steering.py train_compositional.py                          controller (+ LM-Steer / CAA baselines)
  decode_gap_control.py evaluate.py evaluate_std.py score_all.py      decode + score
  judge_{matrix,perattr,eval,comp}.py                                 independent LLM-judge panel
  noise_ceiling.py knn_pilot.py fidelity_check.py mechanism_analysis.py   Sec. 5 analyses
  compmctg_{prep,run,score}.py                                        CompMCTG benchmark
  run_{n48_main,multimodel,qwen,figdata}.sh                           end-to-end drivers
configs/             experiment configs (flc_* Falcon, gem_* SmolLM2, fyelp_* CompMCTG, _* sweeps)
data/                small canonical inputs (prompts, rewards); large rollouts gitignored
paper/scripts/       figure generators (figstyle.py = shared palette)
```
