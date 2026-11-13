"""Fit the proposed model and comparators to the (synthetic) analytic cohort.

Training cohort = index years before `holdout_from_year`; the later-calendar
cohort is held out for temporal validation (script 06). Continuous predictors
are standardised with TRAINING means/SDs only (SAP section 5).
Outputs: results/fits/*.npz, tables T6-T8, figures F8-F10.
"""
import json, os, sys
import numpy as np, pandas as pd, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hmsm.model import ModelSpec, Prior, STATE_NAMES, TRANS
from hmsm.simulate import StepData
from hmsm.fit import fit_hidden, fit_naive, laplace_draws
from hmsm.predict import estimand_draws, naive_to_params, individual_risks, _propagate, standardised_estimands
from hmsm.model import rates, generator, gh_nodes, initial_unstable_prob
from hmsm import plots

proto = yaml.safe_load(open("config/protocol.yaml"))
os.makedirs("results/fits", exist_ok=True)
XC, WC = proto["covariates"]["transitions"], proto["covariates"]["encounter"]
suffix = sys.argv[1] if len(sys.argv) > 1 else ""            # e.g. _sens_contact_first
cov = pd.read_csv(f"data/derived/covariates{suffix}.csv.gz")
ev = pd.read_csv(f"data/derived/events{suffix}.csv.gz")
train = cov.index_year < proto["holdout_from_year"]
cov["comorb"] = cov.n_chronic
scal = {}
for c_new, c_raw in [("age_s", "age"), ("comorb_s", "comorb"), ("prior_util_s", "log_prior_visits")]:
    m, s = cov.loc[train, c_raw].mean(), cov.loc[train, c_raw].std()
    cov[c_new] = (cov[c_raw] - m) / s; scal[c_new] = (m, s)
json.dump(scal, open(f"results/fits/scaling{suffix}.json", "w"))
cov.to_csv(f"data/derived/analysis_covariates{suffix}.csv.gz", index=False)
ctr = cov[train]
d = StepData(ctr, ev[ev.id.isin(ctr.id)], XC, WC)
print(f"training N={d.N}  holdout N={(~train).sum()}  max steps={d.S}")

rng = np.random.default_rng(7)
fits = {}
specs = [("proposed", ModelSpec(4, 2)), ("hidden_ni", ModelSpec(4, 2, informative=False))]
for name, spec in (specs[:1] if suffix else specs):
    f = fit_hidden(d, spec)
    print(name, "converged", f["converged"], "iters", f["n_iter"], f"{f['seconds']:.0f}s")
    np.savez(f"results/fits/{name}{suffix}.npz", theta=f["theta"], cov=f["cov"],
             names=np.array(spec.names), converged=f["converged"])
    fits[name] = (f, spec)
nv = fit_naive(d, 4)
np.savez(f"results/fits/naive{suffix}.npz", theta=nv["phi"], cov=nv["cov"])
if suffix:
    sys.exit(0)

# ---------------- Table 1: baseline characteristics ----------------
def t1(g):
    return pd.Series({"N": len(g), "Age, mean (SD)": f"{g.age.mean():.1f} ({g.age.std():.1f})",
                      "Male, n (%)": f"{int(g.male.sum())} ({100*g.male.mean():.1f})",
                      "Chronic conditions, median [IQR]": f"{g.n_chronic.median():.0f} [{g.n_chronic.quantile(.25):.0f}-{g.n_chronic.quantile(.75):.0f}]",
                      "Prior-year physician visits, median [IQR]": f"{g.prior_visits.median():.0f} [{g.prior_visits.quantile(.25):.0f}-{g.prior_visits.quantile(.75):.0f}]"})
tab1 = pd.concat({"Training (index < %d)" % proto["holdout_from_year"]: t1(cov[train]),
                  "Temporal holdout": t1(cov[~train])}, axis=1)
tab1.to_csv("results/tables/T6_baseline_characteristics.csv"); print(tab1)

# ---------------- Table: parameter estimates (rate ratios) ----------------
f, spec = fits["proposed"]
dr = laplace_draws(f, 2000, rng)
lab = {"age_s": "Age (per SD)", "male": "Male sex", "comorb_s": "Comorbidity (per SD)",
       "prior_util_s": "Prior utilisation (per SD)"}
rows = []
for blk, pre in [("beta_adm", "Admission: "), ("beta_dth", "Death: ")]:
    for j, c in enumerate(XC):
        v = np.exp(dr[:, spec.index[blk].start + j]); rows.append((pre + lab[c], v))
for j, c in enumerate(WC):
    v = np.exp(dr[:, spec.index["gamma"].start + j]); rows.append(("Encounter rate: " + lab[c], v))
rows.append(("Encounter rate: unstable vs stable (exp alpha)", np.exp(dr[:, spec.index["alpha"]].ravel())))
ia, iu = spec.names.index("logq[adm_S]"), spec.names.index("logq[adm_U]")
rows.append(("Admission: unstable vs stable", np.exp(dr[:, iu] - dr[:, ia])))
idc, idu = spec.names.index("logq[dth_CS]"), spec.names.index("logq[dth_CU]")
rows.append(("Community death: unstable vs stable", np.exp(dr[:, idu] - dr[:, idc])))
fr = pd.DataFrame([{"label": l, "hr": np.median(v), "lo": np.percentile(v, 2.5), "hi": np.percentile(v, 97.5)}
                   for l, v in rows])
fr.round(3).to_csv("results/tables/T7_rate_ratios.csv", index=False)
plots.forest(fr, "results/figures/F8_rate_ratios.png", "Proposed model: rate ratios (synthetic cohort)")
oth = {"sigma": np.exp(dr[:, spec.index["log_sigma"]].ravel()), "rho": dr[:, spec.index["rho"]].ravel(),
       "eta_dth": dr[:, spec.index["eta_dth"]].ravel()}
for m in range(3):
    oth[f"P(Y{m+1}=1|stable)"] = 1 / (1 + np.exp(-dr[:, spec.index["a"].start + m]))
    oth[f"P(Y{m+1}=1|unstable)"] = 1 / (1 + np.exp(-(dr[:, spec.index["a"].start + m] + np.exp(dr[:, spec.index["log_d"].start + m]))))
pd.DataFrame([{"parameter": k, "median": np.median(v), "lo": np.percentile(v, 2.5), "hi": np.percentile(v, 97.5)}
              for k, v in oth.items()]).round(3).to_csv("results/tables/T7a_latent_parameters.csv", index=False)

# ---------------- standardised estimands with truth ----------------
std_idx = rng.choice(len(ctr), 300, replace=False)
Xs, Ws = ctr[XC].to_numpy()[std_idx], ctr[WC].to_numpy()[std_idx]
world = np.load("data/synthetic/world_params.npy", allow_pickle=True).item()
wc = pd.read_csv("data/synthetic/world_cov.csv.gz").set_index("ikn").loc[ctr.id.to_numpy()[std_idx]]
truth = standardised_estimands(world, wc[["age_s", "male", "comorb_s", "prior_util_s"]].to_numpy(),
                               wc[["prior_util_s", "comorb_s"]].to_numpy(), n_nodes=15)
est = {}
for name in ["proposed", "hidden_ni"]:
    f, sp = fits[name]
    est[name] = estimand_draws([sp.unpack(t) for t in laplace_draws(f, 200, rng)], Xs, Ws)
ndr = rng.multivariate_normal(nv["phi"], nv["cov"], 200)
est["naive"] = estimand_draws([naive_to_params(t, 4, 2) for t in ndr], Xs, Ws)
rows = []
for q in ["death_30", "death_90", "death_365", "anyhosp_30", "anyhosp_90", "anyhosp_365", "rmdoh"]:
    r = {"estimand": q, "truth (synthetic world)": round(truth[q], 3)}
    for m, e in est.items():
        r[m] = f"{np.mean(e[q]):.3f} ({np.percentile(e[q],2.5):.3f}, {np.percentile(e[q],97.5):.3f})"
    rows.append(r)
t8 = pd.DataFrame(rows); t8.to_csv("results/tables/T8_standardised_estimands.csv", index=False); print(t8.to_string(index=False))

# ---------------- state occupancy trajectories ----------------
f, sp = fits["proposed"]; p = sp.unpack(f["theta"])
xs, ws = gh_nodes(9); b = p["sigma"] * xs; n = len(Xs)
pi = initial_unstable_prob(p, Xs)
G = [generator(rates(p, Xs, Ws, b, np.full(n, k))) for k in range(3)]
init = np.zeros((n, 9, 5)); init[..., 0] = (1 - pi)[:, None]; init[..., 1] = pi[:, None]
grid, P = _propagate(G, init, 5.0)
occ = (P * ws[None, None, :, None]).sum(2).mean(1)
plots.trajectories(grid, occ, "results/figures/F9_state_occupancy.png", STATE_NAMES)

# ---------------- high-risk groups ----------------
pm = sp.unpack(f["theta"])
ir = individual_risks(pm, ctr[XC].to_numpy(), ctr[WC].to_numpy(), n_nodes=7)
g = ctr.assign(death_365=ir["death_365"], anyhosp_365=ir["anyhosp_365"], rmdoh=ir["rmdoh"])
g["age_group"] = pd.cut(g.age, [65, 75, 85, 120], right=False, labels=["65-74", "75-84", "85+"])
g["chronic_group"] = pd.cut(g.n_chronic, [2, 4, 6, 99], right=False, labels=["2-3", "4-5", "6+"])
sub = g.groupby(["age_group", "chronic_group"], observed=True)[["death_365", "anyhosp_365", "rmdoh"]].mean().round(3)
sub["n"] = g.groupby(["age_group", "chronic_group"], observed=True).size()
sub.to_csv("results/tables/T9_risk_by_subgroup.csv"); print(sub)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
for a, q, ttl in [(ax[0], "death_365", "365-day death risk"), (ax[1], "anyhosp_365", "365-day any-hospitalisation risk")]:
    pv = sub[q].unstack()
    im = a.imshow(pv.values, cmap="Blues", vmin=0); a.set_title(ttl)
    a.set_xticks(range(pv.shape[1])); a.set_xticklabels(pv.columns); a.set_xlabel("Chronic conditions")
    a.set_yticks(range(pv.shape[0])); a.set_yticklabels(pv.index); a.set_ylabel("Age group")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            a.text(j, i, f"{pv.values[i,j]:.2f}", ha="center", va="center", fontsize=8,
                   color="white" if pv.values[i, j] > pv.values.max() * 0.6 else "black")
fig.suptitle("Predicted risk by patient group (proposed model, synthetic cohort)", fontweight="bold")
fig.tight_layout(); fig.savefig("results/figures/F10_risk_by_subgroup.png", dpi=150); plt.close(fig)
