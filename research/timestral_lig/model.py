"""Timestral-LIG: zero-init indicator and geometry adapters inside a frozen Timestral-28M.

Sites (every output layer starts at exactly 0, so step 0 equals zero-shot Timestral):
  E  encoder patch tokens:  x_enc += MLP_E(f_enc)                     f_enc (R, 64, 124) = per-patch last + mean
  D  forecast query:        x_dec  = forecast_query + 0.1 MLP_D(f_dec)  f_dec (R, 62) = features at the origin
  H  quantile head:         raw   += kappa * MLP_H(f_dec), before monotone_quantiles (no crossing)
  G  static recalibration:  10 parameters on the per-pass quantile fan (median shrink, width per side/step/tail)
  LoRA r4 on time-attention wQKV / wO of blocks 8-11.
Rows are [x; -x] with group_ids = arange(2B); the output is 0.5 (Q+ + flip(-Q-)), the package's symmetric average.
The mirror row's features are F * PARITY (every feature is odd or even under x -> -x).
"""
import math

import numpy as np
import torch
from scipy.stats import norm
from torch import nn

from indicators import GROUP_OF, PARITY, Bank, normalise
from timestral.layers import TwoStreamMask
from timestral.masks import (build_forecast_positions, build_forecast_time_mask, build_two_stream_variate_mask,
                             compute_enc_key_validity, group_first_observed_patch)
from timestral.model import monotone_quantiles
from timestral.scaler import denormalize_quantiles

K15 = torch.arange(1, 16, dtype=torch.float64)
LK = torch.log(K15) / math.log(15)                                   # l_k in [0, 1]
LV = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
PSI = torch.tensor(np.maximum(np.abs(norm.ppf(LV)) - 0.674, 0) / 1.652)  # 0 inside the quartiles, 1 at q0.01/q0.99


class GroupWNLinear(nn.Module):
    """Linear layer whose input columns of feature group k have a fixed Frobenius norm n0_k.

    A gate z_k that closes cannot be undone by growing the weights, so gates are identifiable."""

    def __init__(self, n_in, n_out, col_group):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        self.register_buffer("col_group", col_group.clone())
        self.register_buffer("n0", self._norms().detach())

    def _norms(self):
        sq = torch.zeros(int(self.col_group.max()) + 1).index_add_(0, self.col_group, self.lin.weight.pow(2).sum(0))
        return sq.sqrt()

    def forward(self, x):
        g = self.col_group
        w = self.lin.weight * (self.n0 / self._norms().clamp_min(1e-12))[g][None, :]
        return nn.functional.linear(x, w, self.lin.bias)


def zero_mlp(n_in, n_out, col_group, hidden=64):
    m = nn.Sequential(GroupWNLinear(n_in, hidden, col_group), nn.SiLU(), nn.Linear(hidden, n_out))
    nn.init.zeros_(m[2].weight)
    nn.init.zeros_(m[2].bias)
    return m


class LoRALinear(nn.Module):
    def __init__(self, base, r=4, alpha=8):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.A.T) @ self.B.T)


class Recal(nn.Module):
    """Static recalibration of a quantile fan u (R, 15, 21) in bps around the row's last value.

    phi_{k,s,i} = clamp(a_s + b_s l_k + (c_s + d_s l_k) psi_i, +-0.7) rescales each outward increment by
    exp(phi) (widths stay in [0.5x, 2x], quantiles stay monotone); the median shrinks by rho_k = r0 + r1 l_k.
    All parameters start at 0, so delta(u) = 0 exactly. `cond` (R, 3) adds per-row offsets to a_lo, a_hi, r0."""

    def __init__(self):
        super().__init__()
        self.side = nn.Parameter(torch.zeros(2, 4, dtype=torch.float64))   # rows: lo, hi; cols: a, b, c, d
        self.rho = nn.Parameter(torch.zeros(2, dtype=torch.float64))

    def delta(self, u, cond=None):
        lk = LK[None, :, None]
        a = self.side[:, 0].view(2, 1, 1, 1)
        if cond is not None:
            a = a + cond[:, :2].T[:, :, None, None]
        phi = []
        for s, idx in ((0, slice(0, 10)), (1, slice(11, 21))):
            b, c, d = self.side[s, 1], self.side[s, 2], self.side[s, 3]
            phi.append(torch.clamp(a[s] + b * lk + (c + d * lk) * PSI[idx][None, None, :], -0.7, 0.7))
        r0 = self.rho[0] + (cond[:, 2:3] if cond is not None else 0)
        rho = torch.clamp(r0 + self.rho[1] * LK[None, :], -0.5, 1.0)          # (R or 1, 15)
        m = u[..., 10]
        dm = (-rho * m)[..., None]
        d_hi = u[..., 11:] - u[..., 10:20]
        d_lo = u[..., 1:11] - u[..., :10]
        up = torch.cumsum(d_hi * torch.expm1(phi[1]), -1)
        down = torch.flip(torch.cumsum(torch.flip(d_lo * torch.expm1(phi[0]), [-1]), -1), [-1])
        return torch.cat([dm - down, dm, dm + up], -1)


class LIG(nn.Module):
    def __init__(self, ts_model, bank: Bank, sites=("H", "G"), lora=False, groups_off=()):
        super().__init__()
        self.ts = ts_model
        for p in self.ts.parameters():
            p.requires_grad_(False)
        self.bank = bank
        self.sites = set(sites)
        self.gate_logit = nn.Parameter(torch.full((11,), 2.0, dtype=torch.float64))
        gm = torch.ones(11, dtype=torch.float64)
        gm[list(groups_off)] = 0
        self.register_buffer("group_mask", gm)
        d = ts_model.config.d_model
        self.mlp_E = zero_mlp(124, d, torch.cat([GROUP_OF, GROUP_OF]))
        self.mlp_D = zero_mlp(62, d, GROUP_OF)
        self.mlp_H = zero_mlp(62, 15 * 21, GROUP_OF)
        kappa = torch.ones(21)
        kappa[10] = 0.03
        self.register_buffer("kappa", kappa)
        self.recal = Recal()
        self.lora = nn.ModuleList()
        if lora:
            for blk in ts_model.transformer.layers[8:12]:
                att = blk.attns[0]
                assert att.is_time
                att.wQKV, att.wO = LoRALinear(att.wQKV), LoRALinear(att.wO)
                self.lora.extend([att.wQKV, att.wO])

    # ---------------------------------------------------------------- features
    def gates(self):
        return torch.sigmoid(self.gate_logit) * self.group_mask

    def gated(self, f):
        """Normalised features (..., 62) float64 -> gated fp32, for the + rows; mirror rows use * PARITY."""
        return (f * self.gates()[GROUP_OF]).float()

    def features(self, raw, mode):
        return normalise(self.bank(raw, mode))

    @staticmethod
    def mirror(f):
        """(B, ..., 62) -> (2B, ..., 62): [f; f * PARITY]."""
        return torch.cat([f, f * PARITY.to(f.dtype)])

    @staticmethod
    def pool(fseq):
        """(R, 2048, 62) -> (R, 64, 124): per patch, the value at its last step and the patch mean."""
        r = fseq.shape[0]
        p = fseq.view(r, 64, 32, 62)
        return torch.cat([p[:, :, -1], p.mean(2)], -1)

    # ---------------------------------------------------------------- backbone pass (sites E, D, LoRA)
    def backbone(self, X, f_enc=None, f_dec=None):
        """X (R, 2048) fp32 price rows -> raw head channels (R, 15, 21) fp32, anchor loc/scale (R,), cap (R, 2)."""
        m, c = self.ts, self.ts.config
        R, T = X.shape
        ps, ll = c.patch_size, T // c.patch_size
        obs = torch.ones_like(X, dtype=torch.bool)
        gid = torch.arange(R)
        scaled, loc, scale = m.scale(X, obs)
        x_enc = m.representation(scaled, obs)
        if "E" in self.sites:
            x_enc = x_enc + self.mlp_E(f_enc)
        x_dec = m.representation.forecast_tokens(R, 1)
        if "D" in self.sites:
            x_dec = x_dec + 0.1 * self.mlp_D(f_dec)[:, None, :]
        assert x_dec.shape == (R, 1, c.d_model)
        obs_pp = obs.unflatten(-1, (ll, ps)).any(dim=-1)
        key_valid = compute_enc_key_validity(obs_pp, c.attend_missing_patches)
        time_mask = build_forecast_time_mask(ll, 1, key_valid, X.device)
        variate_mask = build_two_stream_variate_mask(gid, key_valid, 1)
        uid, inv = torch.unique(gid, return_inverse=True)
        first = group_first_observed_patch(obs_pp, inv, uid.numel())[inv]
        positions = build_forecast_positions(ll, 1, first)
        x = torch.cat([x_enc, x_dec], 1)
        mask = TwoStreamMask(mask=time_mask, enc_len=ll) if c.split_context_horizon_attn else time_mask
        h = m.transformer(x, mask, variate_mask, positions)
        raw = m.head.projection(h[:, ll:ll + 1]).reshape(R, ps, c.num_quantiles)[:, :15].float()
        a_loc, a_scale = loc[:, (ll - 1) * ps], scale[:, (ll - 1) * ps]
        cap = 1e4 * scale[:, -1]
        bounds = torch.stack([X.amin(-1) - cap, X.amax(-1) + cap], -1)
        return raw, a_loc, a_scale, bounds

    # ---------------------------------------------------------------- head tail (sites H, G) and averaging
    def head(self, raw, a_loc, a_scale, xT, f_dec=None, bounds=None):
        """Per-pass quantiles in price units (R, 15, 21) float64, after H and G (and the cap if bounds given)."""
        c = self.ts.config
        R = raw.shape[0]
        if "H" in self.sites:
            raw = raw + self.kappa * self.mlp_H(f_dec).view(R, 15, 21)
        preds = monotone_quantiles(raw, c.quantile_knots, c.quantile_width_factor)
        Q = denormalize_quantiles(preds, a_loc[:, None, None], a_scale[:, None, None], c.use_asinh).double()
        if "G" in self.sites:
            ax = xT.double().abs()[:, None, None]
            Q = Q + ax / 1e4 * self.recal.delta(1e4 * (Q - xT.double()[:, None, None]) / ax)
        if bounds is not None:
            b = bounds.double()
            Q = torch.maximum(torch.minimum(Q, b[:, 1, None, None]), b[:, 0, None, None])
        return Q

    @staticmethod
    def sym(Q):
        B = Q.shape[0] // 2
        return 0.5 * (Q[:B] + (-Q[B:]).flip(-1))

    def forward_full(self, x, raw_feats=None, placebo_perm=None, export=False):
        """x (B, 2048) fp32 closes -> symmetric-averaged price quantiles (B, 15, 21) float64."""
        B = x.shape[0]
        X = torch.cat([x, -x])
        f_enc = f_dec = None
        if self.sites & {"E", "D", "H"}:
            mode = "seq" if "E" in self.sites else "last"
            f = self.features(raw_feats, mode)
            fd = f[:, -1] if mode == "seq" else f
            if placebo_perm is not None:
                fd = fd[placebo_perm]
            f_dec = self.mirror(self.gated(fd))
            if "E" in self.sites:
                f_enc = self.pool(self.mirror(self.gated(f)))
        raw, a_loc, a_scale, bounds = self.backbone(X, f_enc, f_dec)
        Q = self.head(raw, a_loc, a_scale, X[:, -1], f_dec, bounds if export else None)
        out = self.sym(Q)
        return out.clamp_min(0) if export else out

    def forward_cached(self, cache, raw_feats=None, placebo_perm=None, export=False):
        """Same as forward_full with E = D = LoRA = 0, from cached raw head channels (tier T1)."""
        f_dec = None
        if "H" in self.sites:
            fd = self.features(raw_feats, "last")
            if placebo_perm is not None:
                fd = fd[placebo_perm]
            f_dec = self.mirror(self.gated(fd))
        Q = self.head(cache["raw"], cache["a_loc"], cache["a_scale"], cache["xT"], f_dec,
                      cache["bounds"] if export else None)
        out = self.sym(Q)
        return out.clamp_min(0) if export else out


class CFHS(nn.Module):
    """Licence-free control: FHS quantiles sigma_t sqrt(k) z_k (z from Train) with conditional recalibration.

    Works in log-bps directly; a_lo, a_hi and r0 get zero-init linear offsets from the gated features."""

    def __init__(self, bank: Bank, z):
        super().__init__()
        self.bank = bank
        self.gate_logit = nn.Parameter(torch.full((11,), 2.0, dtype=torch.float64))
        self.register_buffer("group_mask", torch.ones(11, dtype=torch.float64))
        self.register_buffer("z", torch.as_tensor(z, dtype=torch.float64))      # (15, 21)
        self.recal = Recal()
        self.cond = nn.Linear(62, 3, bias=False, dtype=torch.float64)
        nn.init.zeros_(self.cond.weight)

    gates = LIG.gates

    def forward(self, sigma, raw_feats, placebo_perm=None):
        f = normalise(self.bank(raw_feats, "last")) * self.gates()[GROUP_OF]
        q0 = sigma[:, None, None] * K15.sqrt()[None, :, None] * self.z[None]
        return q0 + self.recal.delta(q0, self.cond(f))
