"""Decision-gate pilot: can the cache be composed AT INFERENCE via retrieval?

Uses the training cache (flc_multi, 240 prefixes) as a kNN database and the held-out
cache (flc_heldout, 116 unseen prefixes with their own MC ground-truth advantages) as
queries. For each query state, retrieve k nearest cached states in hidden space and
transfer their per-atomic advantages (scatter to vocab, similarity-weighted average,
gather at the query's active set). Compare to the query's own teacher estimate with the
same metrics as fidelity_check.py, so kNN / distilled-controller / noise-ceiling and the
b=0 null all sit on one scale.

Sweeps k and database size (does more cache buy fidelity? -> the 'cache size is the
knob' hypothesis behind the exact-vs-amortized spectrum).

Usage:  python scripts/knn_pilot.py
"""
import _bootstrap  # noqa: F401
import torch

DB_PATH = "data/teacher/flc_multi/topk_advantage.pt"
Q_PATH = "data/teacher/flc_heldout/topk_advantage.pt"
TAU = 0.1
VOCAB = 131072  # Falcon3 vocab (upper bound; only used as scatter size)


def load(path):
    b = torch.load(path, weights_only=False)
    recs = b["records"]
    H = torch.stack([r["hidden"].float() for r in recs])           # [N,3072]
    TK = torch.stack([r["topk_ids"] for r in recs]).long()          # [N,32]
    BLP = torch.stack([r["base_logprob_topk"].float() for r in recs])
    atoms = sorted(recs[0]["atoms"].keys())
    A = torch.stack([torch.stack([r["atoms"][a]["A"].float() for a in atoms]) for r in recs])
    return H, TK, BLP, A, atoms                                     # A: [N,m,32]


def center(vec, blp):
    """center over the active set under the (renormalized) base policy — softmax-invariant
    shift that makes cosines comparable across sources."""
    w = blp.softmax(-1)
    return vec - (w * vec).sum(-1, keepdim=True)


@torch.no_grad()
def main():
    Hd, TKd, BLPd, Ad, atoms = load(DB_PATH)
    Hq, TKq, BLPq, Aq, atoms_q = load(Q_PATH)
    assert atoms == atoms_q, "atom sets differ"
    N, m, K = Ad.shape
    Hd_n = torch.nn.functional.normalize(Hd, dim=-1)
    Hq_n = torch.nn.functional.normalize(Hq, dim=-1)
    sim = Hq_n @ Hd_n.T                                             # [Q,N]

    def evaluate(k, db_idx):
        s = sim[:, db_idx]                                          # [Q,Nd]
        topv, topi = s.topk(min(k, len(db_idx)), dim=-1)            # [Q,k]
        w = (topv / 0.05).softmax(-1)                               # similarity-weighted
        kls, coss, top1s, covs = [], [], [], []
        null_kls, null_top1s = [], []
        for q in range(Hq.shape[0]):
            # scatter neighbors' A into vocab space with weights
            num = torch.zeros(m, VOCAB); den = torch.zeros(VOCAB)
            for j, wt in zip(topi[q].tolist(), w[q].tolist()):
                d = db_idx[j]
                num[:, TKd[d]] += wt * center(Ad[d], BLPd[d].unsqueeze(0).expand(m, -1))
                den[TKd[d]] += wt
            tk = TKq[q]
            covered = den[tk] > 0
            pred = torch.where(covered.unsqueeze(0),
                               num[:, tk] / den[tk].clamp_min(1e-9), torch.zeros(m, K))
            target = Aq[q]                                          # [m,32]
            blp = BLPq[q].unsqueeze(0).expand(m, -1)
            predc, tgtc = center(pred, blp), center(target, blp)
            coss.append(torch.cosine_similarity(predc, tgtc, dim=-1))
            lp_p = (blp + pred / TAU).log_softmax(-1)
            lp_t = (blp + target / TAU).log_softmax(-1)
            kls.append((lp_t.exp() * (lp_t - lp_p)).sum(-1))
            top1s.append((lp_p.argmax(-1) == lp_t.argmax(-1)).float())
            covs.append(covered.float().mean().expand(m))
            # b=0 null (only meaningful once; same for all k/db but cheap)
            lp_0 = blp.log_softmax(-1)
            null_kls.append((lp_t.exp() * (lp_t - lp_0)).sum(-1))
            null_top1s.append((lp_0.argmax(-1) == lp_t.argmax(-1)).float())
        cat = lambda x: torch.stack(x).mean().item()
        return (cat(kls), cat(coss), cat(top1s), cat(covs), cat(null_kls), cat(null_top1s))

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    print(f"{'db_size':>7} {'k':>3} {'KL(p*|pkNN)':>12} {'cosine':>8} {'top1':>6} {'coverage':>9}")
    print("-" * 52)
    null_printed = False
    for nd in (60, 120, 240):
        db_idx = perm[:nd].sort().values
        for k in (1, 4, 8, 16):
            kl, cos, t1, cov, nkl, nt1 = evaluate(k, db_idx)
            print(f"{nd:7d} {k:3d} {kl:12.4f} {cos:8.3f} {t1:6.3f} {cov:9.3f}", flush=True)
        if not null_printed:
            print(f"{'—— b=0 null':>23}: KL(p*||π0)={nkl:.4f}  top1(π0 argmax)={nt1:.3f}")
            null_printed = True
    print("\nreference — distilled controller on the same held-out prefixes (fidelity_check):")
    print("  atomic: KL 0.4351  cosine 0.154  top1 0.548")


if __name__ == "__main__":
    main()
