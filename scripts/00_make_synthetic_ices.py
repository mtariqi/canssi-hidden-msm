"""Create SYNTHETIC ICES-style linked tables (DAD, NACRS, OHIP, RPDB).

No real data are used or required. Output: data/synthetic/*.parquet|csv
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hmsm.model import TRANS, DEFAULT_RATE_CENTRES
from hmsm.cohort import make_synthetic_ices

SEED = 20261112
rng = np.random.default_rng(SEED)
# 'world' parameters for the synthetic population (4 transition covariates:
# age, male, comorbidity, prior utilisation; 2 encounter covariates)
p = {
    "logq": np.array([np.log(DEFAULT_RATE_CENTRES[t]) for t in TRANS]),
    "kappa": np.array([0.0, -0.25, -0.5]),
    "beta_adm": np.array([0.15, 0.10, 0.30, 0.25]),
    "beta_dth": np.array([0.50, 0.20, 0.25, 0.05]),
    "sigma": 0.6, "eta_dth": 0.7,
    "log_lam0": np.log(5.0 / 365), "gamma": np.array([0.4, 0.15]),
    "alpha": 0.9, "rho": 0.5,
    "a": np.array([-2.2, -2.0, -2.5]), "d": np.array([2.2, 1.8, 1.5]),
    "pi": np.array([-1.0, 0.3, 0.0, 0.4, 0.1]),
}
tabs = make_synthetic_ices(14000, p, rng)
os.makedirs("data/synthetic", exist_ok=True)
for k, v in tabs.items():
    v.to_csv(f"data/synthetic/{k}.csv.gz", index=False)
    print(f"{k:6s} {len(v):8d} rows")
np.save("data/synthetic/world_params.npy", p, allow_pickle=True)
