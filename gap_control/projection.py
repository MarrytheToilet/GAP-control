"""Gap-aware trust-region projection (handbook 2.5).

Turns the "how well is the control objective currently satisfied" signal into a
*residual budget*. When the prefix already satisfies the target (value gap g_t small),
the budget shrinks and we barely perturb pi_0; when it does not (g_t large), we allow
stronger intervention. This is what makes the intervention *minimal but sufficient*.

All operations act on the active set S_t = TopK(pi_0) only; residual is 0 elsewhere.
Centering uses the pi_0-weighted mean so the residual matches the advantage definition
A = Q - E_{pi_0}[Q]  (softmax is shift-invariant, so this only affects the budget norm).
"""
from __future__ import annotations

import torch


def value_gap(v_hat: torch.Tensor, reward_target: float = 1.0) -> torch.Tensor:
    """g_t = max(0, R_target - V_hat).  v_hat: [B] -> g_t: [B]."""
    return (reward_target - v_hat).clamp_min(0.0)


def budget(g_t: torch.Tensor, rho_min: float, rho_max: float) -> torch.Tensor:
    """rho(g_t) = rho_min + rho_max * g_t."""
    return rho_min + rho_max * g_t


def center(b_active: torch.Tensor, p0_active: torch.Tensor) -> torch.Tensor:
    """Subtract the pi_0-weighted mean over the active set. Shapes [B, K]."""
    mean = (p0_active * b_active).sum(dim=-1, keepdim=True)
    return b_active - mean


def project_l2(b_active: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """L2-clip each row's residual to norm <= rho (handbook 2.5 version A)."""
    norm = b_active.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (rho.unsqueeze(-1) / norm).clamp(max=1.0)
    return b_active * scale


def project_kl(b_active: torch.Tensor, p0_active: torch.Tensor, kl_budget: torch.Tensor,
               iters: int = 12) -> torch.Tensor:
    """Scale residual by s in [0,1] so KL(softmax(l0+s*b) || softmax(l0)) <= budget.

    Bisection on the scalar s per row (handbook 2.5 version B). p0_active is the base
    distribution restricted+renormalized to the active set.
    """
    logp0 = p0_active.clamp_min(1e-12).log()
    lo = torch.zeros(b_active.size(0), 1, device=b_active.device)
    hi = torch.ones(b_active.size(0), 1, device=b_active.device)

    def kl_at(s):
        logits = logp0 + s * b_active
        p = logits.softmax(-1)
        return (p * (p.clamp_min(1e-12).log() - logp0)).sum(-1, keepdim=True)

    # if full residual already within budget, keep s=1
    over = kl_at(hi) > kl_budget.unsqueeze(-1)
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_big = kl_at(mid) > kl_budget.unsqueeze(-1)
        hi = torch.where(too_big, mid, hi)
        lo = torch.where(too_big, lo, mid)
    s = torch.where(over, (lo + hi) / 2, torch.ones_like(lo))
    return s * b_active


def project_to_norm(b_active: torch.Tensor, target_norm: torch.Tensor,
                    max_scale: float = 50.0) -> torch.Tensor:
    """Set each row's residual to L2 norm == target_norm (scale up OR down). The control
    DIRECTION comes from the controller; the MAGNITUDE comes from the dynamic budget."""
    norm = b_active.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (target_norm.unsqueeze(-1) / norm).clamp(max=max_scale)
    return b_active * scale


def gate_budget(g_t: torch.Tensor, u_t: torch.Tensor, rho_min: float, rho_max: float,
                gamma: float) -> torch.Tensor:
    """The unified per-token budget (eq:gate):  rho_t = (rho_min + rho_max * g_t) * u_t^gamma.
    g_t = value gap (how far from satisfying the control), u_t = base uncertainty 1-max pi0."""
    return (rho_min + rho_max * g_t) * (u_t.clamp(0.0, 1.0) ** gamma)


def gap_project(
    b_active: torch.Tensor,      # [B, K] raw residual on the active set
    p0_active: torch.Tensor,     # [B, K] base probs on active set (renormalized)
    v_hat: torch.Tensor,         # [B] predicted prefix value
    *,
    mode: str = "l2",
    rho_min: float = 0.0,
    rho_max: float = 2.0,
    reward_target: float = 1.0,
    kl_budget_max: float = 1.0,
    u_t: torch.Tensor = None,    # [B] base uncertainty 1-max pi0 (mode="gate")
    gate_gamma: float = 0.0,
) -> torch.Tensor:
    """Full pipeline: center -> compute gap budget -> project. Returns [B, K]."""
    b = center(b_active, p0_active)
    g = value_gap(v_hat, reward_target)
    if mode == "none":
        return b
    if mode == "gate":
        # ONE dynamic mechanism: magnitude = (rho_min + rho_max*g_t) * u_t^gamma, direction = b.
        # Subsumes decode_strength (=rho_max) and entropy_gate (=gamma) and restores the
        # value-gap dynamics (g_t) the L2-clip path silently dropped.
        if u_t is None:
            u_t = torch.ones_like(g)
        return project_to_norm(b, gate_budget(g, u_t, rho_min, rho_max, gate_gamma))
    if mode == "l2":
        return project_l2(b, budget(g, rho_min, rho_max))
    if mode == "kl":
        # gap scales the KL budget directly
        return project_kl(b, p0_active, budget(g, rho_min, kl_budget_max))
    raise ValueError(f"Unknown projection mode: {mode}")
