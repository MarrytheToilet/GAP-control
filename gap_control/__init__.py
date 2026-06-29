"""GAP-Control: Gap-Aware Advantage Projection for Controllable Text Generation.

Controllable generation as amortized advantage prediction under a KL trust region.

Core idea (see handbook Part 2): the KL-optimal one-step logit intervention is the
reward-induced token-level advantage,  l* = l0 + A_c(s_t, .) / tau.  We train a
lightweight controller to *amortize* this advantage into a single residual prediction,
and project the residual into a value-gap-dependent trust region so we only intervene
as much as necessary.
"""

__version__ = "0.1.0"
