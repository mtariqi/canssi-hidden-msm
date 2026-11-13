"""Sensitivity analyses on the training cohort (SAP sections 1, 4, 5, 9).

  quadrature : log-likelihood at the mode with 7 / 15 / 25 Gauss-Hermite nodes
  prior_sd   : baseline log-rate prior SD 0.75 and 2.5 (base 1.5)
  no_re      : no shared random effect
  reduced_em : drop indicator y3 (reduced emissions)
  same_day   : alternative same-day ordering (contacts before admission);
               requires `python scripts/05_fit_empirical.py _sens_contact_first`
Run in parts:  python scripts/07_sensitivity.py prior   |  ... structural  |  ... table
Output: T11_sensitivity.csv
"""
import json, os, sys
import numpy as np, pandas as pd, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hmsm.model import ModelSpec, Prior
from hmsm.simulate import StepData
from hmsm.fit import fit_hidden, laplace_draws
from hmsm.likelihood import loglik
from hmsm.predict import estimand_draws

part = sys.argv[1] if len(sys.argv) > 1 else "all"
proto = yaml.safe_load(open("config/protocol.yaml"))
XC, WC = proto["covariates"]["transitions"], proto["covariates"]["encounter"]
cov = pd.read_csv("data/derived/analysis_covariates.csv.gz")
ev = pd.read_csv("data/derived/events.csv.gz")
ctr = cov[cov.index_year < proto["holdout_from_year"]]
d = StepData(ctr, ev[ev.id.isin(ctr.id)], XC, WC)
base = np.load("results/fits/proposed.npz"); spec = ModelSpec(4, 2); th0 = base["theta"]
os.makedirs("results/fits/sens", exist_ok=True)
rng = np.random.default_rng(3)

def save(name, f, sp):
    np.savez(f"results/fits/sens/{name}.npz", theta=f["theta"], cov=f["cov"], converged=f["converged"],
             n_x=sp.n_x, informative=sp.informative, heterogeneity=sp.heterogeneity, n_ind=sp.n_ind)
    print(name, "converged", f["converged"], f"{f['seconds']:.0f}s", flush=True)

if part in ("prior", "all"):
    q = {n: loglik(th0, spec, d, n) for n in (7, 15, 25)}
    json.dump(q, open("results/fits/sens/quadrature.json", "w")); print("quadrature", q)
    for sd in (0.75, 2.5):
        save(f"prior_sd_{sd}", fit_hidden(d, spec, Prior(baseline_sd=sd), theta0=th0), spec)
if part in ("structural", "all"):
    sp = ModelSpec(4, 2, heterogeneity=False)
    save("no_re", fit_hidden(d, sp, theta0=sp.restrict(th0, spec)), sp)
    d2 = StepData(ctr, ev[ev.id.isin(ctr.id)], XC, WC); d2.Y[..., 2] = -1; d2._indicators()
    d2.Y = d2.Y[..., :2]; d2.n_ind = 2; d2._indicators()
    sp = ModelSpec(4, 2, n_ind=2); p0 = spec.unpack(th0); p0["a"], p0["d"] = p0["a"][:2], p0["d"][:2]
    save("reduced_emissions", fit_hidden(d2, sp, theta0=sp.pack(p0)), sp)
if part in ("table", "all"):
    std = ctr.sample(300, random_state=1); Xs, Ws = std[XC].to_numpy(), std[WC].to_numpy()
    runs = {"base": ("results/fits/proposed.npz", spec)}
    for fn in sorted(os.listdir("results/fits/sens")):
        if fn.endswith(".npz"):
            z = np.load(f"results/fits/sens/{fn}")
            runs[fn[:-4]] = (f"results/fits/sens/{fn}", ModelSpec(int(z["n_x"]), 2, n_ind=int(z["n_ind"]),
                             informative=bool(z["informative"]), heterogeneity=bool(z["heterogeneity"])))
    if os.path.exists("results/fits/proposed_sens_contact_first.npz"):
        runs["same_day_contact_first"] = ("results/fits/proposed_sens_contact_first.npz", spec)
    rows = []
    for name, (path, sp) in runs.items():
        z = np.load(path)
        dr = rng.multivariate_normal(z["theta"], z["cov"], 100)
        e = estimand_draws([sp.unpack(t) for t in dr], Xs, Ws)
        r = {"analysis": name}
        for k in ["death_30", "death_365", "anyhosp_365", "rmdoh"]:
            r[k] = f"{e[k].mean():.3f} ({np.percentile(e[k],2.5):.3f}, {np.percentile(e[k],97.5):.3f})"
        r["alpha"] = f"{z['theta'][sp.index['alpha']][0]:.2f}" if sp.informative else "-"
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv("results/tables/T11_sensitivity.csv", index=False); print(t.to_string(index=False))
