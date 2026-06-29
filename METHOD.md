# GAP-Control — method (A′: amortized compositional controllable decoding)

> **One shared-rollout advantage cache; compose any attribute combination at decoding time
> in a single amortized forward.**

A frozen base LM `π₀`. We add a lightweight controller that emits a logit residual steering
generation toward a *composition* of attributes — soft (classifier-rewarded) and hard
(verifier-rewarded), seen or unseen at training, with arbitrary (incl. negative) weights.

## Core

### ① Advantage residual (amortized, reward-grounded)
The KL-regularized optimal one-step intervention is the reward-induced token-level advantage:
`π* ∝ π₀·exp(A_c/τ)  ⇒  l* = l₀ + A_c/τ`. A controller amortizes it: `r_t = C(h_t, v(c))`,
tied through the base head `b = W_LM·r_t` (low perturbation). Trained by distilling the
teacher advantage. (Reward/RL-grounded, unlike LM-Steer's likelihood-trained linear steer.)

### ② Shared-rollout advantage cache → exact compositional decoding  ⟵ the contribution
Because the control reward is a linear mixture **on the same rollout**, the advantage is
**exactly linear** in the mixture weights:

```
R_c = Σ αᵢ Rᵢ  (same y)   ⇒   A_c = Σ αᵢ Aᵢ ,   V*_c = Σ αᵢ V*ᵢ
```

So **one rollout pass per prefix caches the per-atomic token-level advantages `{Aᵢ}` for ALL
attributes at once**, and *any* composition's teacher target is a free exact linear
combination. The controller is trained on sampled compositions and at inference composes
**any** attribute set — seen or **unseen**, soft+hard mixed, with **signed/scaled** weights —
in **one forward, zero extra rollout/training/search**. A small regularized interaction term
corrects on-policy conflicts (joy+sadness, short+detailed).

This is the wedge vs prior work (positioning table below): the cache-once-compose-any
property with reward-optimality at decoding time.

### ③ Adaptive trust region (supporting mechanism — not the headline)
A predicted value `V̂` sets a per-step KL budget `ρ(g_t)`, `g_t=max(0,R_target−V̂)`: intervene
only as much as needed; project the residual into the budget; monotone constraints clamp the
gap (auto shut-off). Note: adaptive steering strength alone is *not* claimed novel (cf.
Guiding Giants 2505.20309, Dynamic Activation Composition, CAST) — here it is a component.

## Positioning (honest)

| method | amortized 1× | reward/advantage-optimal | compose **any** combo from one cache | soft+hard |
|---|---|---|---|---|
| LM-Steer (add per-attr matrices, likelihood) | ✓ | ✗ | ✗ (train each steer separately) | ✗ |
| Twisted-SMC (per-target particles) | ✗ (N×) | ✓ | ✗ (re-run per target) | partial |
| Air-Decoding / MAGIC / CompMCTG-methods | ✗/train | ✗ | ✗ (latent/train per setup) | ✗ |
| **GAP-Control (ours)** | ✓ | ✓ | ✓ | ✓ |

LM-Steer *can* compose/scale/negate steers — but each steer is trained separately and is not
advantage-optimal. GAP's distinction is the **single shared-rollout advantage cache** from which
**any** combination (incl. unseen/signed/soft+hard) is composed exactly and amortized.

## Claims to defend empirically
1. **Compositional generalization**: small I.D.→Hold-Out joint-success gap (CompMCTG-style) vs baselines.
2. **Cache-once efficiency**: one rollout pass → all C(n,k) combinations at 1× decode cost.
3. **Coverage**: soft+hard mix, unseen combos, signed (suppression) — one controller.
4. **Control–fluency Pareto** at matched perturbation vs LM-Steer; **efficiency** vs SMC.

## Honest risk
The composition identity is mathematically simple (linearity of expectation). The contribution
rests on (i) using it for a *shared-rollout cache → cache-once-compose-any* decoding pipeline,
(ii) reward-optimality, (iii) comprehensive results. Related-work boundaries must be drawn
carefully (LM-Steer composition; SMC prompt-intersection; Air-Decoding/MAGIC multi-attribute).
