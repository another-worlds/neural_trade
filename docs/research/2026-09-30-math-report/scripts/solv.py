"""Achievable BCE reduction vs ln 2 for a given AUC, under a binormal equal-variance model
(score s | y=1 ~ N(+d/2,1), s | y=0 ~ N(-d/2,1), equal priors): AUC = Phi(d/sqrt 2) and the Bayes
posterior is p(s) = sigmoid(d s), so the Bayes-optimal expected BCE is ln2 - I(Y;S) (nats)."""
import numpy as np
from scipy.stats import norm
from scipy.special import expit
from scipy.integrate import quad

for auc in (0.51, 0.52, 0.53, 0.55, 0.60):
    d = np.sqrt(2) * norm.ppf(auc)
    def bce_y1(s):
        return -np.log(expit(d * s)) * norm.pdf(s - d / 2)
    e = quad(bce_y1, -12, 12)[0]  # by symmetry the y=0 half is the same
    red = np.log(2) - e
    # Brier
    br = quad(lambda s: (1 - expit(d * s)) ** 2 * norm.pdf(s - d / 2), -12, 12)[0]
    # spread of the Bayes posterior
    sd_p = np.sqrt(quad(lambda s: (expit(d * s) - 0.5) ** 2 * 0.5 * (norm.pdf(s - d / 2) + norm.pdf(s + d / 2)), -12, 12)[0])
    # samples needed so that a 2-sigma BCE difference is detectable: per-example BCE sd ~ ? (approx: sd of log-loss ~ d-dependent)
    print(f'AUC {auc:.2f}: d={d:.4f} Bayes BCE={e:.6f}  reduction vs ln2 = {red:.2e} nats ({100*red/np.log(2):.3f}% of ln2); '
          f'Brier={br:.5f} (0.25 - {0.25-br:.2e}); sd of Bayes p = {sd_p:.4f}')

# the leader's measured validation BCE (per horizon, served epoch) vs ln 2
for h, v in (('h0', 0.6937), ('h1', 0.6932), ('h2', 0.6937)):
    print(h, 'val BCE', v, 'minus ln2 =', round(v - np.log(2), 5))

# noise level of a batch/val estimate of mean BCE near p=0.5: per-example BCE with p in 0.5+-0.02 has sd ~ |logit| ... use Bernoulli
p = 0.5 + 0.02
sd_ex = abs(np.log(p) - np.log(1 - p)) / 2
print('per-example BCE sd at p=0.52 approx', sd_ex)
