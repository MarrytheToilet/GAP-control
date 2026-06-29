"""Stage 3: train the controller (handbook 2.8 / 4.1 step 5).

Objective (clean, two terms only):
    L_adv   = KL( p* || p_theta )          advantage distillation on the active set
    L_value = (V_hat - V*)^2               value head learns the prefix value
    L       = L_adv + lambda * L_value

where, restricted to S_t = TopK(pi_0):
    p*(v)     proportional to  pi_0(v) exp(A_c(v)/tau)
    p_theta(v) proportional to pi_0(v) exp(b_theta(v)),   b_theta = (W_LM r_t)|_{S_t} (centered-free; softmax shift-invariant)

The base model is loaded only to borrow the tied LM head W_LM. We index just the active
rows of W_LM, so training never does a full-vocab matmul.

Usage:
    python scripts/train_controller.py --config configs/sentiment_mvp.yaml
"""
import _bootstrap  # noqa: F401
import argparse
import os

import torch
import torch.nn.functional as F

from gap_control.config import Config
from gap_control.base_lm import BaseLM
from gap_control.controller import GapController
from gap_control import metrics


def stack_records(records, device):
    H = records[0]["hidden"].numel()
    K = records[0]["topk_ids"].numel()
    hidden = torch.stack([r["hidden"].float() for r in records]).to(device)        # [N,H]
    topk_ids = torch.stack([r["topk_ids"] for r in records]).to(device)            # [N,K]
    base_lp = torch.stack([r["base_logprob_topk"] for r in records]).to(device)    # [N,K]
    A = torch.stack([r["A_topk"] for r in records]).to(device)                     # [N,K]
    vstar = torch.tensor([r["value_target"] for r in records], device=device)      # [N]
    alpha = torch.stack([r["alpha"].float() for r in records]).to(device)          # [N,S]
    sd = records[0].get("state_feat")
    state = (torch.stack([r["state_feat"].float() for r in records]).to(device)
             if sd is not None else None)                                          # [N,Sd]
    return hidden, topk_ids, base_lp, A, vstar, alpha, state, H, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    torch.manual_seed(cfg.seed)
    device = cfg.device

    blob = torch.load(cfg.teacher_path(), weights_only=False)
    records = blob["records"]
    print(f"[train] loaded {len(records)} teacher records")

    hidden, topk_ids, base_lp, A, vstar, alpha_all, state_all, H, K = stack_records(records, device)
    N = hidden.size(0)
    num_slots = alpha_all.size(1)
    state_dim = state_all.size(1) if state_all is not None else 0

    # borrow the frozen tied LM head W_LM (fp32), index only the active rows during training
    base = BaseLM(cfg.base_model, cfg.device, cfg.dtype)
    W = base.lm_head.weight.detach().float()                          # [V, H]
    assert W.size(1) == H, "controller/base hidden size mismatch"

    # train/val split
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(N, generator=g)
    n_val = max(1, int(N * cfg.val_frac))
    val_idx, tr_idx = perm[:n_val].to(device), perm[n_val:].to(device)

    controller = GapController(H, num_slots, cfg.control_dim, cfg.fuse_hidden,
                               cfg.controller_dropout, state_dim=state_dim).to(device)
    opt = torch.optim.AdamW(controller.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    tau = cfg.tau

    def forward_batch(idx, train=True):
        h = hidden[idx]
        a = alpha_all[idx]
        sf = state_all[idx] if state_all is not None else None
        r_t, v_hat = controller(h, a, sf)                           # [B,H], [B]
        # b_active[b,k] = r_t[b] . W[topk_ids[b,k]]
        W_sel = W[topk_ids[idx]]                                     # [B,K,H]
        b_active = (W_sel * r_t.unsqueeze(1)).sum(-1)                # [B,K]
        # restricted distributions over S_t
        logp_theta = (base_lp[idx] + b_active).log_softmax(-1)
        logp_star = (base_lp[idx] + A[idx] / tau).log_softmax(-1)
        p_star = logp_star.exp()
        l_adv = (p_star * (logp_star - logp_theta)).sum(-1).mean()
        l_val = F.mse_loss(v_hat, vstar[idx])
        return l_adv, l_val, b_active

    best_val = float("inf")
    os.makedirs(os.path.dirname(cfg.controller_path()), exist_ok=True)
    for epoch in range(cfg.epochs):
        controller.train()
        ep = tr_idx[torch.randperm(tr_idx.numel(), device=device)]
        tot = 0.0
        for s in range(0, ep.numel(), cfg.batch_size):
            idx = ep[s:s + cfg.batch_size]
            l_adv, l_val, _ = forward_batch(idx, train=True)
            loss = l_adv + cfg.value_lambda * l_val
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), cfg.grad_clip)
            opt.step()
            tot += loss.item() * idx.numel()

        # validation + key diagnostic: bias-advantage Spearman (handbook 4.2)
        controller.eval()
        with torch.no_grad():
            l_adv, l_val, b_active = forward_batch(val_idx, train=False)
            corrs = [metrics.spearman(b_active[i].cpu(), (A[val_idx][i] / tau).cpu())
                     for i in range(val_idx.numel())]
            corr = sum(corrs) / len(corrs)
        val_loss = (l_adv + cfg.value_lambda * l_val).item()
        print(f"epoch {epoch:02d} | train {tot/tr_idx.numel():.4f} | "
              f"val L_adv {l_adv:.4f} L_val {l_val:.4f} | "
              f"bias-adv Spearman {corr:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict": controller.state_dict(),
                "config": cfg.__dict__,
                "hidden_size": H, "num_slots": num_slots, "state_dim": state_dim,
                "val_spearman": corr,
            }, cfg.controller_path())

    print(f"[train] best val loss {best_val:.4f} -> {cfg.controller_path()}")
    print(f"[train] final bias-advantage Spearman {corr:.3f} "
          f"(handbook MVP target: > 0.3)")


if __name__ == "__main__":
    main()
