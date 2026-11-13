"""Run the simulation study. Resumable: completed (scenario, rep) pairs are skipped.

    python scripts/03_run_simulation.py --mode pilot
    python scripts/03_run_simulation.py --mode pilot --stress
    python scripts/03_run_simulation.py --mode full --shard 3 --n-shards 20   # cluster array job
"""
import argparse, os, sys, time
import pandas as pd, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hmsm.simstudy import scenario_grid, true_estimands, run_replicate

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
ap.add_argument("--stress", action="store_true", help="run stress tests instead of main grid")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n-shards", type=int, default=1)
ap.add_argument("--out", default="results/sim")
ap.add_argument("--budget", type=float, default=float("inf"),
                help="stop cleanly (between replicates) after this many seconds")
args = ap.parse_args()

cfg = yaml.safe_load(open("config/simulation.yaml"))
mode = cfg[args.mode]
os.makedirs(args.out, exist_ok=True)

jobs = []
if not args.stress:
    for scn in scenario_grid(mode["sizes"]):
        R = mode["replicates"]
        if args.mode == "full" and scn["scenario"] in mode.get("boundary_scenarios", []):
            R = mode["boundary_replicates"]
        jobs += [(scn, r, None) for r in range(R)]
    outfile = f"{args.out}/{args.mode}_main_shard{args.shard}.csv"
else:
    st = cfg["stress_tests"]
    base = {"n": st["reference_n"], "discrimination": "strong", "informative": True, "heterogeneity": True}
    for name, opts in st["cases"].items():
        scn = dict(base, scenario=f"stress_{name}")
        jobs += [(scn, r, opts) for r in range(st["replicates"][args.mode])]
    outfile = f"{args.out}/{args.mode}_stress_shard{args.shard}.csv"

# interleave replicates across scenarios so a partial run still covers the design
jobs.sort(key=lambda j: (j[1], j[0]["scenario"]))
jobs = [j for i, j in enumerate(jobs) if i % args.n_shards == args.shard]
done = set()
if os.path.exists(outfile):
    prev = pd.read_csv(outfile)
    done = set(zip(prev.scenario, prev.rep))
truth_cache = {}
T0, last = time.time(), 0.0
for scn, rep, stress in jobs:
    if (scn["scenario"], rep) in done:
        continue
    if time.time() - T0 + 1.3 * last > args.budget:
        print("time budget reached; rerun to resume", flush=True); break
    key = (scn["discrimination"], scn["informative"], scn["heterogeneity"])
    if key not in truth_cache:
        truth_cache[key] = true_estimands(scn)
    scn = dict(scn, truth=truth_cache[key])
    t0 = time.time()
    rows = run_replicate(scn, rep, cfg["master_seed"], cfg["n_draws"], cfg["n_std"], stress)
    pd.DataFrame(rows).to_csv(outfile, mode="a", header=not os.path.exists(outfile), index=False)
    last = time.time() - t0
    print(f"{scn['scenario']:40s} rep {rep:3d}  {last:6.1f}s", flush=True)
