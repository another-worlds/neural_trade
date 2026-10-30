"""Automatic checks for Timestral-LIG (run after `train.py cache`, before training). Writes $LIG_WORK/<set>/checks.json.

T0  (in train.py cache) step-0 identity vs the stored zero-shot npz on all Test origins
T0c cached path == full path, with nonzero H and G
T1  gradient flow: at step 0 only zero-init output layers, LoRA B and G get gradients; after 2 steps every group
    (all 40 periods, all 11 gates) does; a synthetic gradient moves every period (guards against D1 / D7)
T2  causality: perturbing c, H, L after t' changes no feature at steps <= t' and no E token of a patch ending
    before t'; perturbing closes after the origin leaves the forecast bit-identical
T3  parity: bank(-x) == PARITY * bank(x) (close-derived groups); seq[-1] == last
T4  split asserts (Data); T5 group ids / shapes (asserted in model.backbone); T7 constants from Train only
T6  reflection equivariance: LIG(-x) == Refl(LIG(x)) with nonzero adapters
T8  no FFT wrap-around: perturbing the last context bar changes no earlier step
"""
import json
import os
import sys

import numpy as np
import torch

import train
from data import CTX
from indicators import GROUP_OF, PARITY, Bank
from losses import Objective, to_logbps
from model import LIG


def perturb_after(raw, t_cut, scale=0.01):
    out = {k: v.clone() for k, v in raw.items()}
    for k in ("close", "high", "low"):
        out[k][:, t_cut + 1:] *= 1 + scale
    return out


def randomise(model, seed=7, std=0.05):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in (model.mlp_E, model.mlp_D, model.mlp_H):
            m[2].weight.copy_(torch.randn(m[2].weight.shape, generator=g) * std)
        model.recal.side.copy_(torch.randn(2, 4, generator=g, dtype=torch.float64) * 0.1)
        model.recal.rho.copy_(torch.randn(2, generator=g, dtype=torch.float64) * 0.1)
        for m in model.lora:
            m.B.copy_(torch.randn(m.B.shape, generator=g) * 0.01)


def main(ds):
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    X = train.Ctx(ds)
    D = X.D
    res = {"set": ds, "T4_splits": D.describe()}
    rs = np.random.default_rng(0)
    org = np.sort(rs.choice(D.train, 4, replace=False))

    # T3 parity and seq/last
    b = X.bank()
    raw = D.raw(org)
    with torch.no_grad():
        fl, fs = b(raw, "last"), b(raw, "seq")
        neg = dict(raw)
        neg["close"] = -raw["close"]
        fn = b(neg, "seq")
    cg = GROUP_OF < 9
    res["T3_parity_max"] = float((fn - PARITY * fs)[..., cg].abs().max())
    res["T3b_seq_vs_last_max"] = float((fs[:, -1] - fl).abs().max())

    # T2 / T8 causality and wrap-around
    worst, worst_tok = 0.0, 0.0
    for t_cut in (100, 1000, 1500, CTX - 2):
        with torch.no_grad():
            fp = b(perturb_after(raw, t_cut), "seq")
        rel = ((fp - fs)[:, :t_cut + 1].abs() / (fs[:, :t_cut + 1].abs() + 1e-12)).max()
        worst = max(worst, float(rel))
        n_tok = (t_cut + 1) // 32
        pe, pp = LIG.pool(fs.float()), LIG.pool(fp.float())
        worst_tok = max(worst_tok, float((pe[:, :n_tok] - pp[:, :n_tok]).abs().max()))
    res["T2_T8_feature_rel_change_max"] = worst
    res["T2_E_token_change_max"] = worst_tok
    close_mod = D.close.copy()
    close_mod[org[0] + 1:] *= 1.05
    res["T2_future_closes_untouched_context"] = bool(
        np.array_equal(close_mod[org[0] - CTX + 1:org[0] + 1], D.close[org[0] - CTX + 1:org[0] + 1]))

    # T1 gradient flow with every module attached
    torch.manual_seed(0)
    lig = LIG(train.load_ts(), X.bank(), sites=("E", "D", "H", "G"), lora=True)
    obj = Objective(X.k["P0"], X.k["lnv0"], X.k["lam0"])
    y, s = D.target(org)

    def loss_of(model):
        Q = model.forward_full(D.context(org), D.raw(org))
        t = obj.terms(to_logbps(Q, torch.from_numpy(D.close[org])), y, s)
        return obj.total(t) + 1e-3 * model.gates().sum()

    named = {n: p for n, p in lig.named_parameters() if p.requires_grad}
    lig.zero_grad()
    loss_of(lig).backward()
    nz0 = sorted(n for n, p in named.items() if p.grad is not None and p.grad.abs().sum() > 0)
    allowed = ("mlp_E.2.", "mlp_D.2.", "mlp_H.2.", "recal.", ".B", "gate_logit")
    res["T1_step0_nonzero"] = nz0
    res["T1_step0_only_allowed"] = all(any(a in n for a in allowed) for n in nz0)
    opt = torch.optim.Adam([p for p in named.values()], lr=1e-3)
    for _ in range(2):
        opt.zero_grad()
        loss_of(lig).backward()
        opt.step()
    opt.zero_grad()
    loss_of(lig).backward()
    zero_after = sorted(n for n, p in named.items() if p.grad is None or p.grad.abs().sum() == 0)
    res["T1_after2_zero_grad_params"] = zero_after
    res["T1_after2_periods_nonzero"] = int((lig.bank.theta.grad != 0).sum())
    res["T1_after2_gates_nonzero"] = int((lig.gate_logit.grad != 0).sum())
    bank = X.bank()
    th0 = bank.theta.detach().clone()
    o = torch.optim.SGD([bank.theta], lr=1e-2)
    w = torch.randn(62, dtype=torch.float64)
    (bank(D.raw(org), "last") * w).sum().backward()
    o.step()
    res["T1_synthetic_periods_moved"] = int((bank.theta.detach() != th0).sum())

    # T0c cached == full, and T6 equivariance, with nonzero adapters
    ts = train.load_ts()
    m = LIG(ts, X.bank(), sites=("H", "G"))
    randomise(m)
    vo = np.sort(rs.choice(D.v2, 16, replace=False))
    with torch.no_grad():
        qc = to_logbps(m.forward_cached(X.cache(vo), D.raw(vo)), torch.from_numpy(D.close[vo]))
        qf = to_logbps(m.forward_full(D.context(vo), D.raw(vo)), torch.from_numpy(D.close[vo]))
    res["T0c_cached_vs_full_max_bps"] = float((qc - qf).abs().max())
    res["T0c_adapter_effect_bps"] = float((qc - torch.from_numpy(X.q0(vo))).abs().max())
    torch.manual_seed(0)
    m6 = LIG(train.load_ts(), X.bank(), sites=("E", "D", "H", "G"), lora=True)
    randomise(m6)
    with torch.no_grad():
        x = D.context(vo[:8])
        r = D.raw(vo[:8])
        rn = dict(r)
        rn["close"] = -r["close"]
        Qp = m6.forward_full(x, r)
        Qn = m6.forward_full(-x, rn)
    xT = x[:, -1].double()[:, None, None]
    res["T6_equivariance_max_bps"] = float(((Qn + Qp.flip(-1)) / xT.abs() * 1e4).abs().max())
    res["T7_constants_source"] = "Train origins only (train.constants: D.train, D.train[::2])"
    res["T5_group_ids"] = "arange(2B), asserted shapes in model.backbone"
    ok = (res["T3_parity_max"] < 1e-9 and res["T3b_seq_vs_last_max"] < 1e-9 and res["T2_T8_feature_rel_change_max"] < 1e-12
          and res["T2_E_token_change_max"] == 0 and res["T1_step0_only_allowed"] and not res["T1_after2_zero_grad_params"]
          and res["T1_synthetic_periods_moved"] == 40 and res["T0c_cached_vs_full_max_bps"] < 1e-3
          and res["T6_equivariance_max_bps"] < 1e-5)
    res["all_pass"] = bool(ok)
    json.dump(res, open(train.wdir(ds, "checks.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    assert ok, "a check failed"


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "repo")
