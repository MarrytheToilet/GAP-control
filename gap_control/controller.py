"""The amortized advantage-residual controller (handbook 2.4).

One lightweight module. Given the base hidden state h_t and a control vector z_c it predicts
  * r_t      -- a hidden-space residual, projected to vocab by the *tied* base LM head W_LM
                so the residual stays in the base output space (low-perturbation by design).
  * V_hat_t  -- the prefix value, used to size the trust region at decode time.

The control condition is a reward-mixture over attribute *slots* (handbook 2.3): a vector
alpha in R^{NUM_SLOTS} (see attributes.SLOTS). Because z_c is linear in this mixture,
continuous intensity (scale a slot) and multi-attribute composition (activate several slots)
need no new machinery. An optional keyword embedding conditions on a specific keyword.

The controller never stores W_LM (it's the frozen base head, passed at forward time), so
checkpoints stay tiny and the residual is guaranteed tied.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ControlEncoder(nn.Module):
    """alpha (mixture over attribute slots) [+ keyword embedding] -> z_c.

    z_c = MLP( alpha @ slot_emb  [+ kw_proj(kw_emb)] ).  Linear in the mixture.
    """

    def __init__(self, num_slots: int, control_dim: int, kw_in_dim: Optional[int] = None):
        super().__init__()
        self.slot_emb = nn.Embedding(num_slots, control_dim)
        self.kw_proj = nn.Linear(kw_in_dim, control_dim) if kw_in_dim else None
        self.mlp = nn.Sequential(
            nn.Linear(control_dim, control_dim),
            nn.GELU(),
            nn.Linear(control_dim, control_dim),
        )

    def forward(self, alpha: torch.Tensor, kw_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        mixed = alpha @ self.slot_emb.weight              # [B, control_dim]
        if self.kw_proj is not None and kw_emb is not None:
            mixed = mixed + self.kw_proj(kw_emb)
        return self.mlp(mixed)


class GapController(nn.Module):
    def __init__(self, hidden_size: int, num_slots: int,
                 control_dim: int = 64, fuse_hidden: int = 512, dropout: float = 0.1,
                 kw_in_dim: Optional[int] = None, state_dim: int = 0):
        super().__init__()
        self.encoder = ControlEncoder(num_slots, control_dim, kw_in_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size + control_dim + state_dim, fuse_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_hidden, fuse_hidden),
            nn.GELU(),
        )
        self.r_head = nn.Linear(fuse_hidden, hidden_size)
        self.v_head = nn.Linear(fuse_hidden, 1)
        nn.init.zeros_(self.r_head.weight)   # start near pi_0 (tiny residual at init)
        nn.init.zeros_(self.r_head.bias)

        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.state_dim = state_dim

    def forward(self, h_t: torch.Tensor, alpha: torch.Tensor,
                state_feat: Optional[torch.Tensor] = None,
                kw_emb: Optional[torch.Tensor] = None):
        """h_t: [B, H], alpha: [B, NUM_SLOTS], state_feat: [B, state_dim] (the live
        constraint-satisfaction signal that makes control dynamic). Returns (r_t, v_hat)."""
        z_c = self.encoder(alpha, kw_emb)
        parts = [h_t, z_c]
        if self.state_dim:
            if state_feat is None:
                state_feat = h_t.new_zeros(h_t.size(0), self.state_dim)
            parts.append(state_feat)
        feat = self.trunk(torch.cat(parts, dim=-1))
        r_t = self.r_head(feat)
        v_hat = self.v_head(feat).squeeze(-1)
        return r_t, v_hat

    @staticmethod
    def residual_from_r(r_t: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
        """b_raw = W_LM r_t over the full vocab. lm_head is the frozen base head.
        Bridges controller fp32 / base fp16 and returns fp32 for stable softmax math."""
        w_dtype = lm_head.weight.dtype
        return lm_head(r_t.to(w_dtype)).float()


class CompositionalController(nn.Module):
    """Composes an arbitrary number of attributes (any soft/hard mix) and generalizes to
    unseen combinations.

    Two pathways (the innovation):
      * additive  -- per-slot residual r_j computed independently, then summed in HIDDEN space
                     weighted by the (normalized) mixture alpha:  r_add = sum_j a_j r_j.
                     Since W_LM is linear, b_add = W_LM r_add is exactly linear in alpha ->
                     it composes to unseen combinations by construction (matches the exact
                     linear structure of the advantage, handbook 2.3).
      * interaction -- a small residual conditioned on the WHOLE mixture, regularized toward
                     zero, that corrects for attribute interactions / conflicts where pure
                     additivity fails (joy+sadness, short+detailed).

    Drop-in: forward(h, alpha, state) -> (r_t, v_hat), identical signature to GapController,
    so decoding/projection code is unchanged. residual_from_r is reused from GapController.
    """

    def __init__(self, hidden_size: int, num_slots: int, control_dim: int = 64,
                 fuse_hidden: int = 512, dropout: float = 0.1, state_dim: int = 0,
                 interaction: bool = True, caa_dirs: torch.Tensor = None,
                 linear_rank: int = 0, in_dim: int = None):
        super().__init__()
        # in_dim = dim of the hidden FED to the controller (may concat multiple layers);
        # hidden_size = dim of the residual OUTPUT (last layer, for the tied W_LM). Usually equal.
        in_dim = in_dim or hidden_size
        self.in_dim = in_dim
        self.slot_emb = nn.Embedding(num_slots, control_dim)
        # LM-Steer-style low-rank linear backbone (Han et al. 2024): a learned, input-dependent
        # per-slot steer r += sum_j a_j U_j (V_j^T h). Stronger than a fixed CAA direction;
        # our nonlinear trunk + state + gap add the adaptivity LM-Steer lacks.
        self.linear_rank = linear_rank
        if linear_rank > 0:
            self.steer_U = nn.Parameter(torch.zeros(num_slots, hidden_size, linear_rank))
            self.steer_V = nn.Parameter(torch.randn(num_slots, hidden_size, linear_rank) * 0.02)
        # CAA directions ENRICH THE CONTROL VECTOR (not injected into the output): each slot's
        # control code = learned embedding + projection of its data-derived attribute direction.
        # The controller still predicts the entire residual r_t from (h_t, control_vector);
        # b = W_LM r_t is unchanged. The control vector thus reflects attributes + intensity
        # and is freely composable (any subset of slots, any intensities via alpha).
        if caa_dirs is not None:
            self.register_buffer("caa", caa_dirs)            # [num_slots, H], unit-norm rows
            self.caa_proj = nn.Linear(hidden_size, control_dim)
        else:
            self.caa = None
        sd = state_dim
        # per-slot additive pathway (shared weights across slots)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim + control_dim + sd, fuse_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(fuse_hidden, fuse_hidden), nn.GELU())
        self.r_head = nn.Linear(fuse_hidden, hidden_size)
        self.v_head = nn.Linear(fuse_hidden, 1)
        nn.init.zeros_(self.r_head.weight); nn.init.zeros_(self.r_head.bias)

        self.interaction = interaction
        if interaction:
            self.mix_mlp = nn.Sequential(nn.Linear(control_dim, control_dim), nn.GELU(),
                                         nn.Linear(control_dim, control_dim))
            self.trunk_int = nn.Sequential(
                nn.Linear(in_dim + control_dim + sd, fuse_hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(fuse_hidden, fuse_hidden), nn.GELU())
            self.r_int = nn.Linear(fuse_hidden, hidden_size)
            self.v_int = nn.Linear(fuse_hidden, 1)
            nn.init.zeros_(self.r_int.weight); nn.init.zeros_(self.r_int.bias)
            nn.init.zeros_(self.v_int.weight); nn.init.zeros_(self.v_int.bias)

        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.state_dim = state_dim

    residual_from_r = staticmethod(GapController.residual_from_r)

    def forward(self, h_t, alpha, state_feat=None, return_aux=False):
        B = h_t.size(0)
        S = self.num_slots
        dev = h_t.device
        se = self.slot_emb(torch.arange(S, device=dev))            # [S, C]
        if self.caa is not None:                                   # CAA-enrich the control vector
            se = se + self.caa_proj(self.caa.to(se.dtype))         # [S, C]
        h_exp = h_t.unsqueeze(1).expand(B, S, -1)
        se_exp = se.unsqueeze(0).expand(B, S, -1)
        parts = [h_exp, se_exp]
        if self.state_dim:
            if state_feat is None:
                state_feat = h_t.new_zeros(B, self.state_dim)
            parts.append(state_feat.unsqueeze(1).expand(B, S, -1))
        feat = self.trunk(torch.cat(parts, dim=-1))               # [B, S, F]
        r_slots = self.r_head(feat)                               # [B, S, H]
        v_slots = self.v_head(feat).squeeze(-1)                   # [B, S]

        # L1-normalize so SIGNED coefficients work (advantage algebra: + express, - suppress,
        # fractional = intensity). For all-positive alpha this equals the old sum-normalization.
        an = alpha / alpha.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        r = torch.einsum("bs,bsh->bh", an, r_slots)              # residual predicted by controller
        v = torch.einsum("bs,bs->b", an, v_slots)
        if self.linear_rank > 0:                                 # LM-Steer low-rank linear backbone
            vh = torch.einsum("shr,bh->bsr", self.steer_V, h_t)
            rlin = torch.einsum("shr,bsr->bsh", self.steer_U, vh)
            r = r + torch.einsum("bs,bsh->bh", an, rlin)
        r_int_norm = h_t.new_zeros(B)

        if self.interaction:
            mix = self.mix_mlp(alpha @ se)                        # whole-mixture control vector (CAA-enriched)
            ip = [h_t, mix] + ([state_feat] if self.state_dim else [])
            fint = self.trunk_int(torch.cat(ip, dim=-1))
            r_i = self.r_int(fint)
            r = r + r_i
            v = v + self.v_int(fint).squeeze(-1)
            r_int_norm = r_i.norm(dim=-1)

        if return_aux:
            return r, v, r_int_norm
        return r, v


def build_controller(kind, hidden_size, num_slots, *, control_dim, fuse_hidden, dropout,
                     state_dim, caa_dirs=None, linear_rank=0, steer_rank=8, in_dim=None,
                     interaction=True):
    """Factory: GAP-Control (full) or a baseline, with one interface."""
    if kind == "lmsteer":
        return LMSteerController(hidden_size, num_slots, control_dim, fuse_hidden,
                                 dropout, state_dim, rank=steer_rank)
    if kind == "static_caa":
        return StaticCAAController(hidden_size, num_slots, control_dim, fuse_hidden,
                                   dropout, state_dim, caa_dirs=caa_dirs)
    return CompositionalController(hidden_size, num_slots, control_dim, fuse_hidden,
                                   dropout, state_dim=state_dim, caa_dirs=caa_dirs,
                                   linear_rank=linear_rank, in_dim=in_dim, interaction=interaction)


class LMSteerController(nn.Module):
    """Faithful LM-Steer baseline (Han et al. 2024): r_t = sum_j alpha_j (U_j V_j^T) h, a
    FIXED low-rank linear map of h per attribute, composed by addition, intensity = alpha.
    No control-vector conditioning beyond slot selection, no running state, no value gap.
    Same drop-in interface (h, alpha, state) -> (r, v); decode/projection unchanged."""

    def __init__(self, hidden_size, num_slots, control_dim=64, fuse_hidden=512,
                 dropout=0.1, state_dim=0, rank=8, **kw):
        super().__init__()
        self.steer_U = nn.Parameter(torch.zeros(num_slots, hidden_size, rank))
        self.steer_V = nn.Parameter(torch.randn(num_slots, hidden_size, rank) * 0.02)
        self.v_head = nn.Linear(hidden_size, 1)                  # value head for the gap (kept off in baseline configs)
        self.hidden_size, self.num_slots, self.state_dim = hidden_size, num_slots, 0

    residual_from_r = staticmethod(GapController.residual_from_r)

    def forward(self, h_t, alpha, state_feat=None, return_aux=False):
        vh = torch.einsum("shr,bh->bsr", self.steer_V, h_t)
        rlin = torch.einsum("shr,bsr->bsh", self.steer_U, vh)    # [B,S,H]
        r = torch.einsum("bs,bsh->bh", alpha, rlin)             # raw alpha = strength (epsilon)
        v = self.v_head(h_t).squeeze(-1)
        if return_aux:
            return r, v, h_t.new_zeros(h_t.size(0))
        return r, v


class StaticCAAController(nn.Module):
    """Static CAA / ActAdd baseline: r_t = scale * (alpha_hat @ d), a CONSTANT direction per
    attribute (independent of h), composed by weighted sum. The pure steering-vector method."""

    def __init__(self, hidden_size, num_slots, control_dim=64, fuse_hidden=512,
                 dropout=0.1, state_dim=0, caa_dirs=None, **kw):
        super().__init__()
        assert caa_dirs is not None, "StaticCAA needs caa_dirs"
        self.register_buffer("caa", caa_dirs)
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.v_head = nn.Linear(hidden_size, 1)
        self.hidden_size, self.num_slots, self.state_dim = hidden_size, num_slots, 0

    residual_from_r = staticmethod(GapController.residual_from_r)

    def forward(self, h_t, alpha, state_feat=None, return_aux=False):
        an = alpha / alpha.sum(-1, keepdim=True).clamp_min(1e-6)
        r = self.scale * (an @ self.caa.to(h_t.dtype))          # constant direction (no h dependence)
        r = r.expand(h_t.size(0), -1) if r.dim() == 1 else r
        v = self.v_head(h_t).squeeze(-1)
        if return_aux:
            return r, v, h_t.new_zeros(h_t.size(0))
        return r, v
