"""Summarise simulation output: bias, RMSE, coverage, width with Monte Carlo SEs.

Outputs results/tables/T3_sim_performance.csv, T4_sim_convergence.csv,
T5_stress_tests.csv and figures F4-F7.
"""
import glob, os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hmsm.simstudy import summarise
from hmsm import plots

mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
main = pd.concat([pd.read_csv(f) for f in glob.glob(f"results/sim/{mode}_main_shard*.csv")])
summ, fails = summarise(main)
summ.round(5).to_csv("results/tables/T3_sim_performance.csv", index=False)
fails.round(2).to_csv("results/tables/T4_sim_convergence.csv", index=False)
for q, fn, ttl in [("death_365", "F4_sim_death365.png", "365-day death risk"),
                   ("anyhosp_365", "F5_sim_anyhosp365.png", "365-day risk of any hospitalisation"),
                   ("rmdoh", "F6_sim_rmdoh.png", "Restricted mean days alive outside hospital (365 d)")]:
    plots.sim_performance(summ, q, f"results/figures/{fn}", f"Simulation ({mode}): {ttl}")
plots.runtime_plot(fails, "results/figures/F7_sim_runtime.png")
# compact headline table: average over scenarios by method x estimand x observation mechanism
s = summ.copy()
s["visits"] = s.scenario.str.contains("_inf_").map({True: "informative", False: "non-informative"})
key = ["death_30", "death_365", "anyhosp_30", "anyhosp_365", "rmdoh", "alpha", "logq[adm_U]"]
head = (s[s.quantity.isin(key)].groupby(["quantity", "visits", "method"])
        [["bias", "rmse", "coverage", "width"]].mean().round(4).reset_index())
head.to_csv("results/tables/T3a_sim_headline.csv", index=False)
print(head.to_string(index=False))
print(fails.groupby("method")[["attempted", "converged", "mean_seconds"]].sum().to_string())
st = glob.glob(f"results/sim/{mode}_stress_shard*.csv")
if st:
    sd = pd.concat([pd.read_csv(f) for f in st])
    ss, sf = summarise(sd)
    ss = ss[ss.quantity.isin(["death_365", "anyhosp_365", "rmdoh", "alpha"])]
    ss.round(4).to_csv("results/tables/T5_stress_tests.csv", index=False)
    print(ss[["scenario", "method", "quantity", "R", "bias", "coverage"]].round(3).to_string(index=False))
