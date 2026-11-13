"""Prior predictive check of baseline log-rate priors (SAP section 5) + state diagram.

Draws parameters from the prior and summarises implied 365-day death and
any-hospitalisation risks, which should cover clinically plausible ranges
without piling up at 0 or 1. Figure F2, table T2.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hmsm.model import ModelSpec, Prior
from hmsm.predict import standardised_estimands
from hmsm import plots

rng = np.random.default_rng(1)
spec = ModelSpec(2, 1)
X = rng.standard_normal((60, 2)); W = rng.standard_normal((60, 1))
rows = []
for sd in (0.75, 1.5, 2.5):
    mu, s = Prior(baseline_sd=sd).means_sds(spec)
    for _ in range(120):
        th = rng.normal(mu, s)
        th[spec.index["log_sigma"]] = np.log(abs(rng.standard_normal()) + 1e-3)
        e = standardised_estimands(spec.unpack(th), X, W, n_nodes=5, h=5.0)
        rows.append({"baseline_sd": sd, **e})
d = pd.DataFrame(rows)
tab = d.groupby("baseline_sd")[["death_365", "anyhosp_365", "rmdoh"]].quantile([0.05, 0.5, 0.95]).unstack().round(3)
tab.to_csv("results/tables/T2_prior_predictive.csv"); print(tab.T)
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
for sd, c in zip((0.75, 1.5, 2.5), ("#9ecae1", "#1b6ca8", "#08306b")):
    dd = d[d.baseline_sd == sd]
    ax[0].hist(dd.death_365, bins=30, range=(0, 1), alpha=0.55, color=c, label=f"SD {sd}")
    ax[1].hist(dd.anyhosp_365, bins=30, range=(0, 1), alpha=0.55, color=c)
ax[0].set_xlabel("Prior predictive 365-day death risk"); ax[1].set_xlabel("Prior predictive 365-day any-hospitalisation risk")
ax[0].legend(frameon=False, fontsize=7); fig.suptitle("Prior predictive check of baseline log-rate priors", fontweight="bold")
fig.tight_layout(); fig.savefig("results/figures/F2_prior_predictive.png", dpi=150); plt.close(fig)
plots.state_diagram("results/figures/F0_state_diagram.png")
