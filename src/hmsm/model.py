"""Model specification for the continuous-time hidden multi-state model (HMSM).

State space (SAP section 2)
---------------------------
    0 = community / stable      (CS)
    1 = community / unstable    (CU)
    2 = hospital  / stable      (HS)
    3 = hospital  / unstable    (HU)
    4 = death (absorbing)

Observed care setting C(t) in {community, hospital}; latent instability Z(t) in
{stable, unstable}. Admission and discharge preserve Z at the instant of
transition. Death is reachable from every living state.

Transition intensities (proposal, "Statistical methods")
    q_i,rs(t) = q0_rs * exp(kappa_k + x_i' beta_g(rs) + eta_g(rs) * b_i),
with piecewise-constant time bands k over days 0-30, 31-90, 91-365, covariate
effects shared within transition groups (admission, death) in the pilot model,
and b_i ~ N(0, sigma^2). eta is fixed to 1 for admission (scale anchor).

Encounter (observation) intensity, community states only
    lambda_i(t) = lambda0 * exp(w_i' gamma + alpha * Z_i(t) + rho * b_i).

Emissions at each encounter, indicator m = 1..M
    logit P(Y_m = 1 | Z) = a_m + d_m * Z,   d_m = exp(log_d_m) > 0
The positivity of d_m orders the latent labels (no label switching).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

STATE_NAMES = ["Community/stable", "Community/unstable",
               "Hospital/stable", "Hospital/unstable", "Death"]
TRANS = ["on_C", "rec_C", "on_H", "rec_H", "adm_S", "adm_U",
         "dis_S", "dis_U", "dth_CS", "dth_CU", "dth_HS", "dth_HU"]
BAND_EDGES = np.array([0.0, 30.0, 90.0, 365.0])  # days
HORIZON = 365.0

# plausible per-day baseline rates used as prior centres (see 02_prior_predictive)
DEFAULT_RATE_CENTRES = {
    "on_C": 1.0 / 365, "rec_C": 1.0 / 45, "on_H": 1.0 / 60, "rec_H": 1.0 / 15,
    "adm_S": 0.6 / 365, "adm_U": 4.0 / 365, "dis_S": 1.0 / 6, "dis_U": 1.0 / 10,
    "dth_CS": 0.04 / 365, "dth_CU": 0.5 / 365, "dth_HS": 0.002, "dth_HU": 0.012,
}


def band_of(t):
    """Index of the time band containing t (0, 1, 2)."""
    return np.clip(np.searchsorted(BAND_EDGES, t, side="right") - 1, 0, 2)


@dataclass
class ModelSpec:
    """Describes the parameter vector layout for a given covariate set."""
    n_x: int                       # covariates in transition / initial models
    n_w: int                       # covariates in encounter model
    n_ind: int = 3                 # diagnostic indicators
    informative: bool = True       # estimate alpha, rho (False = hidden-NI model)
    heterogeneity: bool = True     # estimate random effect
    names: list = field(default_factory=list)
    index: dict = field(default_factory=dict)

    def __post_init__(self):
        blocks = [("logq", len(TRANS)), ("kappa", 2),
                  ("beta_adm", self.n_x), ("beta_dth", self.n_x)]
        if self.heterogeneity:
            blocks += [("log_sigma", 1), ("eta_dth", 1)]
        blocks += [("log_lam0", 1), ("gamma", self.n_w)]
        if self.informative:
            blocks += [("alpha", 1)]
            if self.heterogeneity:
                blocks += [("rho", 1)]
        blocks += [("a", self.n_ind), ("log_d", self.n_ind), ("pi", 1 + self.n_x)]
        pos = 0
        for name, size in blocks:
            self.index[name] = slice(pos, pos + size)
            if size == 1:
                self.names.append(name)
            elif name == "logq":
                self.names += [f"logq[{t}]" for t in TRANS]
            else:
                self.names += [f"{name}[{j}]" for j in range(size)]
            pos += size
        self.n_par = pos

    # ------------------------------------------------------------------
    def unpack(self, theta):
        """Return a dict of named parameter blocks (with fixed values filled)."""
        th = np.asarray(theta, float)
        g = lambda k: th[self.index[k]] if k in self.index else None
        p = {
            "logq": g("logq"), "kappa": np.concatenate([[0.0], g("kappa")]),
            "beta_adm": g("beta_adm"), "beta_dth": g("beta_dth"),
            "sigma": float(np.exp(g("log_sigma")[0])) if self.heterogeneity else 0.0,
            "eta_dth": float(g("eta_dth")[0]) if self.heterogeneity else 0.0,
            "log_lam0": float(g("log_lam0")[0]), "gamma": g("gamma"),
            "alpha": float(g("alpha")[0]) if self.informative else 0.0,
            "rho": float(g("rho")[0]) if (self.informative and self.heterogeneity) else 0.0,
            "a": g("a"), "d": np.exp(g("log_d")), "pi": g("pi"),
        }
        return p

    def pack(self, p):
        th = np.zeros(self.n_par)
        th[self.index["logq"]] = p["logq"]
        th[self.index["kappa"]] = np.asarray(p["kappa"])[-2:]
        th[self.index["beta_adm"]] = p["beta_adm"]
        th[self.index["beta_dth"]] = p["beta_dth"]
        if self.heterogeneity:
            th[self.index["log_sigma"]] = np.log(max(p["sigma"], 1e-3))
            th[self.index["eta_dth"]] = p["eta_dth"]
        th[self.index["log_lam0"]] = p["log_lam0"]
        th[self.index["gamma"]] = p["gamma"]
        if self.informative:
            th[self.index["alpha"]] = p["alpha"]
            if self.heterogeneity:
                th[self.index["rho"]] = p["rho"]
        th[self.index["a"]] = p["a"]
        th[self.index["log_d"]] = np.log(p["d"])
        th[self.index["pi"]] = p["pi"]
        return th

    def restrict(self, theta_full, full_spec):
        """Map a parameter vector from a richer spec to this spec."""
        return self.pack(full_spec.unpack(theta_full))


# ----------------------------------------------------------------------
# priors (SAP section 5)
# ----------------------------------------------------------------------
@dataclass
class Prior:
    baseline_sd: float = 1.5
    rate_centres: dict = field(default_factory=lambda: dict(DEFAULT_RATE_CENTRES))
    lam0_centre: float = np.log(4.0 / 365)

    def means_sds(self, spec: ModelSpec):
        mu = np.zeros(spec.n_par)
        sd = np.ones(spec.n_par)
        mu[spec.index["logq"]] = [np.log(self.rate_centres[t]) for t in TRANS]
        sd[spec.index["logq"]] = self.baseline_sd
        mu[spec.index["log_lam0"]] = self.lam0_centre
        sd[spec.index["log_lam0"]] = self.baseline_sd
        sd[spec.index["a"]] = 1.5
        mu[spec.index["log_d"]] = np.log(1.0)  # weak regularisation of d_m
        sd[spec.index["pi"]] = 1.5
        mu[spec.index["pi"].start] = -1.0
        return mu, sd

    def logpdf(self, theta, spec: ModelSpec):
        mu, sd = self.means_sds(spec)
        lp_vec = -0.5 * ((theta - mu) / sd) ** 2
        if spec.heterogeneity:
            # half-Normal(0,1) on sigma, expressed on log-sigma with Jacobian
            j = spec.index["log_sigma"].start
            ls = theta[j]
            lp_vec[j] = -0.5 * np.exp(2 * ls) + ls
        return lp_vec.sum()

    def precision(self, theta, spec: ModelSpec):
        """Diagonal of -d2 log prior (used in the Laplace approximation)."""
        _, sd = self.means_sds(spec)
        prec = 1.0 / sd ** 2
        if spec.heterogeneity:
            j = spec.index["log_sigma"].start
            prec[j] = 2.0 * np.exp(2 * theta[j])
        return np.diag(prec)


# ----------------------------------------------------------------------
# intensities
# ----------------------------------------------------------------------
def gh_nodes(n_nodes: int):
    """Gauss-Hermite nodes/weights for E[f(b)], b ~ N(0, 1)."""
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    return x, w / w.sum()


def rates(p, x, w, b, band):
    """Transition and encounter intensities.

    Parameters
    ----------
    p : dict from ModelSpec.unpack
    x : (n, n_x) covariates ; w : (n, n_w) encounter covariates
    b : (Q,) random-effect values (already scaled by sigma)
    band : (n,) int band index
    Returns dict of arrays of shape (n, Q) keyed by TRANS + 'lam_S', 'lam_U'.
    """
    xa = x @ p["beta_adm"] if p["beta_adm"].size else np.zeros(len(x))
    xd = x @ p["beta_dth"] if p["beta_dth"].size else np.zeros(len(x))
    wg = w @ p["gamma"] if p["gamma"].size else np.zeros(len(w))
    kap = p["kappa"][band][:, None]
    out = {}
    for j, t in enumerate(TRANS):
        lp = p["logq"][j] + kap
        if t.startswith("adm"):
            lp = lp + xa[:, None] + b[None, :]
        elif t.startswith("dth"):
            lp = lp + xd[:, None] + p["eta_dth"] * b[None, :]
        else:
            lp = lp + np.zeros((1, b.size))
        out[t] = np.exp(lp)
    base = p["log_lam0"] + wg[:, None] + p["rho"] * b[None, :]
    out["lam_S"] = np.exp(base)
    out["lam_U"] = np.exp(base + p["alpha"])
    return out


def generator(r, augmented=False):
    """Full 5x5 generator (or 4-state 'ever-hospitalised' generator).

    augmented=True gives states [CS, CU, dead-before-admission, ever-admitted],
    used for the cumulative risk of any hospitalisation (SAP section 7).
    """
    n, Q = r["on_C"].shape
    if not augmented:
        G = np.zeros((n, Q, 5, 5))
        G[..., 0, 1] = r["on_C"]; G[..., 1, 0] = r["rec_C"]
        G[..., 2, 3] = r["on_H"]; G[..., 3, 2] = r["rec_H"]
        G[..., 0, 2] = r["adm_S"]; G[..., 1, 3] = r["adm_U"]
        G[..., 2, 0] = r["dis_S"]; G[..., 3, 1] = r["dis_U"]
        G[..., 0, 4] = r["dth_CS"]; G[..., 1, 4] = r["dth_CU"]
        G[..., 2, 4] = r["dth_HS"]; G[..., 3, 4] = r["dth_HU"]
    else:
        G = np.zeros((n, Q, 4, 4))
        G[..., 0, 1] = r["on_C"]; G[..., 1, 0] = r["rec_C"]
        G[..., 0, 2] = r["dth_CS"]; G[..., 1, 2] = r["dth_CU"]
        G[..., 0, 3] = r["adm_S"]; G[..., 1, 3] = r["adm_U"]
    idx = np.arange(G.shape[-1])
    G[..., idx, idx] = -G.sum(-1)
    return G


def initial_unstable_prob(p, x):
    eta = p["pi"][0] + (x @ p["pi"][1:] if x.shape[1] else 0.0)
    return 1.0 / (1.0 + np.exp(-eta))


def emission_prob(p, Y):
    """P(Y | Z) for each encounter; Y in {0,1,-1 (missing)}; returns (n, 2)."""
    out = np.ones((Y.shape[0], 2))
    for z in (0, 1):
        pr = 1.0 / (1.0 + np.exp(-(p["a"] + p["d"] * z)))
        f = np.where(Y == 1, pr, np.where(Y == 0, 1 - pr, 1.0))
        out[:, z] = f.prod(-1)
    return out
