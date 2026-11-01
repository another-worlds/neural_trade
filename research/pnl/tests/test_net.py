"""PnL band net unit tests (SPEC 'network' UNIT TESTS: band gradcheck, non-zero cost gradient, chain = engine F)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fast  # noqa: E402
import net as NET  # noqa: E402
import sim  # noqa: E402


def _away_from_ties(B=3, T=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    tau = (torch.rand(B, T, generator=g, dtype=torch.float64) - 0.5) * 1.6
    b = 0.05 + 0.3 * torch.rand(B, T, generator=g, dtype=torch.float64)
    p0 = (torch.rand(B, generator=g, dtype=torch.float64) - 0.5)
    return tau, b, p0


def test_band_gradcheck():
    for seed in range(5):
        tau, b, p0 = _away_from_ties(seed=seed)
        # keep every step at least 1e-3 away from a band edge so finite differences do not cross a branch
        p = torch.empty_like(tau)
        code = np.empty(tau.shape, np.int8)
        fast.band_forward(tau.numpy(), b.numpy(), p0.numpy(), p.numpy(), code)
        prev = torch.cat([p0[:, None], p[:, :-1]], 1)
        margin = torch.minimum((prev - (tau - b)).abs(), (prev - (tau + b)).abs())
        if margin.min() < 1e-3:
            continue
        tau.requires_grad_(True)
        b.requires_grad_(True)
        p0.requires_grad_(True)
        assert torch.autograd.gradcheck(NET.BandFn.apply, (tau, b, p0), eps=1e-6, atol=1e-6)


def test_band_forward_matches_definition():
    tau, b, p0 = _away_from_ties(B=2, T=200, seed=7)
    p = NET.BandFn.apply(tau, b, p0).numpy()
    for j in range(2):
        prev = float(p0[j])
        for k in range(200):
            prev = min(max(prev, float(tau[j, k] - b[j, k])), float(tau[j, k] + b[j, k]))
            prev = min(max(prev, -1.0), 1.0)
            assert abs(p[j, k] - prev) < 1e-12


def test_cost_gradient_nonzero_and_flat_policy_has_zero_pnl():
    torch.manual_seed(0)
    m = NET.PnLNet(5, "LIN")
    with torch.no_grad():
        m.head.weight[0].normal_(0, 0.5)
    X = torch.randn(4, 50, 5)
    R = torch.zeros(4, 50)
    Fd = torch.zeros(4, 50)
    s2h = torch.full((4, 50), 1e-6)
    loss, U, pabs, ql = NET.batch_loss(m, X, R, Fd, s2h, W=10, spd=20, kappa=1.0, rbar_h=0.0, s_d=0.01,
                                       band_on=False)
    m.zero_grad()
    (loss - NET.QLIKE_W * torch.tensor(ql)).backward()
    g = m.head.weight.grad[0].abs().sum()
    assert U < 0 and g > 0                      # zero returns: only cost, which the tau head can reduce
    m2 = NET.PnLNet(5, "LIN")
    with torch.no_grad():
        m2.head.weight.zero_()
    _, U2, pabs2, _ = NET.batch_loss(m2, X, torch.randn(4, 50) * 1e-3, Fd, s2h, 10, 20, 1.0, 0.0, 0.01, True)
    assert U2 == 0.0 and pabs2 == 0.0


def test_chain_utility_equals_engine_f():
    rs = np.random.default_rng(0)
    n = 500
    tau = np.tanh(rs.normal(0, 1, n))
    b = 0.02 + 0.2 * rs.random(n)
    R = rs.normal(0, 1e-3, n)
    Fd = rs.normal(0, 1e-5, n)
    day = np.repeat(np.arange(5), 100)
    U, p, Pi = NET.chain_utility(tau, b, R, Fd, day, 0.0)
    net = sim.engine_f(p, R, Fd, c_in=NET.C_IN, c_out=NET.C_OUT)
    assert np.allclose(Pi, np.bincount(day, weights=net))
    assert abs(U - (Pi.mean() - Pi.var() - Pi.mean() ** 2)) < 1e-15
