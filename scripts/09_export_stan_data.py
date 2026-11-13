"""Export the training cohort as Stan data (JSON) + Laplace-mode inits for R/fit_stan.R."""
import json, os, sys
import numpy as np, pandas as pd, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hmsm.model import ModelSpec, Prior, TRANS
from hmsm.simulate import StepData

proto = yaml.safe_load(open("config/protocol.yaml"))
XC, WC = proto["covariates"]["transitions"], proto["covariates"]["encounter"]
cov = pd.read_csv("data/derived/analysis_covariates.csv.gz"); ev = pd.read_csv("data/derived/events.csv.gz")
ctr = cov[cov.index_year < proto["holdout_from_year"]]
d = StepData(ctr, ev[ev.id.isin(ctr.id)], XC, WC)
mask = np.arange(d.S)[None, :] < d.nsteps[:, None]
start = np.concatenate([[1], 1 + np.cumsum(d.nsteps)[:-1]])
spec, pr = ModelSpec(4, 2), Prior()
mu, _ = pr.means_sds(spec)
data = {"N": int(d.N), "T": int(mask.sum()), "K_x": len(XC), "K_w": len(WC), "M": 3,
        "start": start.tolist(), "len": d.nsteps.tolist(), "dt": d.dt[mask].tolist(),
        "care": d.care[mask].tolist(), "band": (d.band[mask] + 1).tolist(), "etype": d.etype[mask].tolist(),
        "Y": d.Y[mask].tolist(), "X": d.X.tolist(), "W": d.W.tolist(),
        "logq_mu": mu[spec.index["logq"]].tolist(), "logq_sd": pr.baseline_sd, "lam0_mu": pr.lam0_centre}
os.makedirs("data/derived/stan", exist_ok=True)
json.dump(data, open("data/derived/stan/stan_data.json", "w"))
p = spec.unpack(np.load("results/fits/proposed.npz")["theta"])
init = {"logq": p["logq"].tolist(), "kappa_raw": p["kappa"][1:].tolist(), "beta_adm": p["beta_adm"].tolist(),
        "beta_dth": p["beta_dth"].tolist(), "sigma": p["sigma"], "eta_dth": p["eta_dth"], "log_lam0": p["log_lam0"],
        "gamma": p["gamma"].tolist(), "alpha": p["alpha"], "rho": p["rho"], "a": p["a"].tolist(),
        "log_d": np.log(p["d"]).tolist(), "pi0": float(p["pi"][0]), "pi_x": p["pi"][1:].tolist(),
        "b_raw": [0.0] * int(d.N)}
json.dump(init, open("data/derived/stan/init_laplace.json", "w"))
print("wrote data/derived/stan/{stan_data,init_laplace}.json  N =", d.N, " steps =", int(mask.sum()))
