"""PnL band net (SPEC 'network' A): heads tau (target exposure), b (no-trade half-width), aux (log variance ratio);
hard band layer with an exact custom gradient; de-meaned daily mean-variance utility with exact inc/red costs.
"""
import numpy as np
import torch
from torch import nn

import fast

C_IN, C_OUT = 3e-4, 5e-4
GAMMA = 2.0
QLIKE_W = 0.05
B_INIT = float(np.log((0.05 - 0.02) / 0.98 / (1 - (0.05 - 0.02) / 0.98)))   # b = 0.05 at init


class BandFn(torch.autograd.Function):
    """p_k = clip(p_{k-1}, tau_k - b_k, tau_k + b_k), clipped to [-1, 1]; numba loop over time, batch-vectorised."""

    @staticmethod
    def forward(ctx, tau, b, p0):
        t = np.ascontiguousarray(tau.detach().numpy(), dtype=np.float64)
        bb = np.ascontiguousarray(b.detach().numpy(), dtype=np.float64)
        p0n = np.ascontiguousarray(p0.detach().numpy(), dtype=np.float64)
        p = np.empty_like(t)
        code = np.empty(t.shape, np.int8)
        fast.band_forward(t, bb, p0n, p, code)
        ctx.save_for_backward(torch.from_numpy(code))
        return torch.from_numpy(p).to(tau.dtype)

    @staticmethod
    def backward(ctx, g):
        (code,) = ctx.saved_tensors
        gn = np.ascontiguousarray(g.detach().numpy(), dtype=np.float64)
        gt, gb = np.empty_like(gn), np.empty_like(gn)
        gp0 = np.empty(gn.shape[0])
        fast.band_backward(gn, code.numpy(), gt, gb, gp0)
        return torch.from_numpy(gt).to(g.dtype), torch.from_numpy(gb).to(g.dtype), torch.from_numpy(gp0).to(g.dtype)


class PnLNet(nn.Module):
    def __init__(self, n_in, arch="LIN"):
        super().__init__()
        self.arch = arch
        if arch == "LIN":
            self.trunk, d = nn.Identity(), n_in
        else:
            self.trunk = nn.Sequential(nn.Linear(n_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.GELU())
            d = 32
        self.head = nn.Linear(d, 3)
        with torch.no_grad():
            self.head.weight.normal_(0.0, 1e-3)
            self.head.bias.copy_(torch.tensor([0.0, B_INIT, 0.0]))

    def forward(self, x):
        o = self.head(self.trunk(x))
        tau = torch.tanh(o[..., 0])
        b = 0.02 + 0.98 * torch.sigmoid(o[..., 1])
        return tau, b, o[..., 2]


def inc_red(p, p_prev):
    dx = (p - p_prev).abs()
    return (p.abs() + dx - p_prev.abs()) * 0.5, (p_prev.abs() + dx - p.abs()) * 0.5


def batch_loss(model, X, R, Fd, s2h, W, spd, kappa, rbar_h, s_d, band_on):
    """X (B, L, F); R, Fd, s2h (B, L). Loss steps are the last L - W. Returns (loss, U, mean |p|, qlike)."""
    tau, b, aux = model(X)
    B = X.shape[0]
    p = BandFn.apply(tau, b, torch.zeros(B, dtype=tau.dtype)) if band_on else tau
    p_prev = torch.cat([torch.zeros(B, 1, dtype=p.dtype), p[:, :-1]], 1)
    inc, red = inc_red(p, p_prev)
    sd = torch.sqrt(s2h)
    Rw = torch.maximum(torch.minimum(R, 6 * sd), -6 * sd)
    pnl = p * (Rw - rbar_h - Fd) - kappa * (C_IN * inc + C_OUT * red)
    pnl = pnl[:, W:]
    Pi = pnl.reshape(B, -1, spd).sum(-1)
    U = Pi.mean() - 0.5 * GAMMA * (Pi * Pi).mean()
    a = aux[:, W:]
    ql = (a + Rw[:, W:] ** 2 / (torch.exp(a) * s2h[:, W:])).mean()
    loss = -U / s_d + QLIKE_W * ql
    return loss, float(U.detach()), float(p[:, W:].abs().mean().detach()), float(ql.detach())


@torch.no_grad()
def infer(model, Xn, chunk=200_000):
    model.eval()
    tau, b, aux = [], [], []
    for i in range(0, len(Xn), chunk):
        t, bb, a = model(torch.from_numpy(np.ascontiguousarray(Xn[i:i + chunk])))
        tau.append(t.numpy())
        b.append(bb.numpy())
        aux.append(a.numpy())
    return np.concatenate(tau), np.concatenate(b), np.concatenate(aux)


def chain_utility(tau, b, R, Fd, day, rbar_h, kappa=1.0, step=0.0):
    """Hard-band chain from p0 = 0; engine-F pnl per decision; calendar-day sums; U = mean - gamma/2 mean square."""
    p = fast.band_chain(np.asarray(tau, np.float64), np.asarray(b, np.float64), 0.0, step)
    pp = np.r_[0.0, p[:-1]]
    dx = np.abs(p - pp)
    inc = (np.abs(p) + dx - np.abs(pp)) * 0.5
    red = (np.abs(pp) + dx - np.abs(p)) * 0.5
    pnl = p * (R - rbar_h - Fd) - kappa * (C_IN * inc + C_OUT * red)
    _, inv = np.unique(day, return_inverse=True)
    Pi = np.bincount(inv, weights=pnl)
    return float(Pi.mean() - 0.5 * GAMMA * (Pi * Pi).mean()), p, Pi
