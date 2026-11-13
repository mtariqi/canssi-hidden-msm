"""Build the analytic cohort from the (synthetic) linked tables under the locked protocol.

Outputs: data/derived/{covariates,events}.csv.gz, results/tables/T1_cohort_flow.csv,
         results/figures/F1_cohort_flow.png
"""
import os, sys
import pandas as pd, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hmsm.cohort import build_cohort
from hmsm import plots

proto = yaml.safe_load(open("config/protocol.yaml"))
dates = {"dad": ["admdate", "ddate"], "nacrs": ["regdate"], "ohip": ["servdate"],
         "rpdb": ["bdate", "dthdate", "elig_start", "elig_end"]}
tabs = {k: pd.read_csv(f"data/synthetic/{k}.csv.gz", parse_dates=v, dtype={"dxcode": str, "dx10": str})
        for k, v in dates.items()}
os.makedirs("data/derived", exist_ok=True); os.makedirs("results/tables", exist_ok=True)
for label, order in [("main", proto["same_day_order"]), ("sens_contact_first", proto["same_day_order_sensitivity"])]:
    pr = dict(proto, same_day_order=order)
    cov, ev, flow, _ = build_cohort(tabs, pr)
    suffix = "" if label == "main" else f"_{label}"
    cov.to_csv(f"data/derived/covariates{suffix}.csv.gz", index=False)
    ev.to_csv(f"data/derived/events{suffix}.csv.gz", index=False)
    if label == "main":
        flow.to_csv("results/tables/T1_cohort_flow.csv", index=False)
        print(flow.to_string(index=False))
        print(ev.event.value_counts().to_string())
        plots.cohort_flow(flow, "results/figures/F1_cohort_flow.png")
