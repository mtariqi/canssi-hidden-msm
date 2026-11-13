"""Budget plan, monthly cash-flow, timeline (Gantt) and compute estimate.

Inputs : budget/budget_items.csv (from the CANSSI Ontario application),
         budget/expenditure_tracker.csv (actual invoices, filled during the award),
         results/tables/T4_sim_convergence.csv (measured pilot run times).
Outputs: T12_budget.csv, T13_monthly_cashflow.csv, T14_compute_estimate.csv,
         F12_budget_allocation.png, F13_timeline_budget.png
"""
import os, sys
import numpy as np, pandas as pd, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CAP = 5000
b = pd.read_csv("budget/budget_items.csv")
assert b.amount_cad.sum() == CAP, "budget must total the CAD 5,000 request"
assert np.allclose(b.unit_cost_cad * b.units, b.amount_cad)
t12 = b[["item", "provider", "basis", "amount_cad", "status"]].copy()
t12["share_%"] = (100 * t12.amount_cad / CAP).round(1)
t12 = pd.concat([t12, pd.DataFrame([{"item": "Total requested", "amount_cad": t12.amount_cad.sum(), "share_%": 100.0}])])
t12.to_csv("results/tables/T12_budget.csv", index=False); print(t12[["item", "amount_cad", "share_%"]].to_string(index=False))

months = np.arange(1, 13)
cf = pd.DataFrame({"month": months})
for _, r in b.iterrows():
    n = r.end_month - r.start_month + 1
    cf[r["item"]] = np.where((months >= r.start_month) & (months <= r.end_month), r.amount_cad / n, 0.0)
cf["planned_total"] = cf[b["item"]].sum(axis=1); cf["planned_cumulative"] = cf.planned_total.cumsum()
act = pd.read_csv("budget/expenditure_tracker.csv")
cf["actual_cumulative"] = np.nan
if len(act):
    act["month"] = pd.to_datetime(act.date).dt.month   # replace with award-month mapping
    cf["actual_cumulative"] = cf.month.map(act.groupby("month").amount_cad.sum()).fillna(0).cumsum()
cf.round(2).to_csv("results/tables/T13_monthly_cashflow.csv", index=False)

# ---- compute estimate from measured pilot run times ----
cfg = yaml.safe_load(open("config/simulation.yaml"))
t4 = pd.read_csv("results/tables/T4_sim_convergence.csv")
t4["n"] = t4.scenario.str.extract(r"n(\d+)_").astype(float)
per_rep = t4.groupby(["n", "scenario"]).mean_seconds.sum().groupby("n").mean()   # all 3 methods
slope, intercept = np.polyfit(per_rep.index, per_rep.values, 1)
sec = lambda n: intercept + slope * n
full = cfg["full"]; rows = []
for n in full["sizes"]:
    reps = 8 * full["replicates"]
    if n == min(full["sizes"]):
        reps += len(full["boundary_scenarios"]) * (full["boundary_replicates"] - full["replicates"])
    rows.append({"cohort_size": n, "replicates_total": reps, "seconds_per_replicate": round(sec(n), 1),
                 "cpu_hours": round(reps * sec(n) / 3600, 1),
                 "basis": "measured" if n in per_rep.index else "linear extrapolation from pilot"})
st = cfg["stress_tests"]
rows.append({"cohort_size": st["reference_n"], "replicates_total": len(st["cases"]) * st["replicates"]["full"],
             "seconds_per_replicate": round(sec(st["reference_n"]), 1),
             "cpu_hours": round(len(st["cases"]) * st["replicates"]["full"] * sec(st["reference_n"]) / 3600, 1),
             "basis": "stress tests"})
t14 = pd.DataFrame(rows)
t14.loc[len(t14)] = {"cohort_size": "TOTAL", "replicates_total": t14.replicates_total.sum(), "seconds_per_replicate": "",
                     "cpu_hours": round(t14.cpu_hours.sum(), 1), "basis": "x1.5 safety margin recommended"}
t14.to_csv("results/tables/T14_compute_estimate.csv", index=False); print(t14.to_string(index=False))

# ---- F12 allocation ----
fig, ax = plt.subplots(figsize=(6.4, 2.4))
cols = ["#1b6ca8", "#6baed6", "#c6dbef", "#fdae6b"]
left = 0
for (_, r), c in zip(b.iterrows(), cols):
    ax.barh(0, r.amount_cad, left=left, color=c, ec="white")
    ax.text(left + r.amount_cad / 2, 0, f"{r['item']}\n${r.amount_cad:,.0f}", ha="center", va="center", fontsize=7.5)
    left += r.amount_cad
ax.set_xlim(0, CAP); ax.set_yticks([]); ax.set_xlabel("CAD")
ax.set_title("Budget allocation: CAD 5,000 requested (no personnel costs)")
fig.tight_layout(); fig.savefig("results/figures/F12_budget_allocation.png", dpi=150); plt.close(fig)

# ---- F13 timeline + cumulative spend ----
phases = [("Approvals, ICES quotation, feasibility, protocol lock", 1, 2),
          ("Simulation design & algorithm prototype", 3, 4),
          ("Model evaluation & secure data preparation", 5, 7),
          ("Empirical analysis & temporal validation", 8, 10),
          ("Sensitivity analyses, manuscripts, synthetic-data package", 11, 12)]
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
for i, (lab, s, e) in enumerate(phases[::-1]):
    ax.barh(i, e - s + 1, left=s - 0.5, color="#c6dbef", ec="#1b6ca8")
    ax.text(s - 0.4, i, lab, va="center", fontsize=7.5, color="black")
ax.set_yticks([]); ax.set_title("Project timeline (months after data-access activation)")
for i, (_, r) in enumerate(b.iterrows()):
    ax2.bar(months + (i - 1.5) * 0.2, cf[r["item"]], width=0.2, color=cols[i], label=r["item"])
ax3 = ax2.twinx(); ax3.plot(months, cf.planned_cumulative, "k-o", ms=3, label="Planned cumulative")
if cf.actual_cumulative.notna().any():
    ax3.plot(months, cf.actual_cumulative, "r--", label="Actual cumulative")
ax3.set_ylim(0, CAP * 1.05); ax3.set_ylabel("Cumulative CAD"); ax2.set_ylabel("Monthly CAD")
ax2.set_xticks(months); ax2.set_xlabel("Month"); ax2.legend(fontsize=6.5, frameon=False, loc="upper left")
ax3.legend(fontsize=6.5, frameon=False, loc="center left")
fig.tight_layout(); fig.savefig("results/figures/F13_timeline_budget.png", dpi=150); plt.close(fig)
