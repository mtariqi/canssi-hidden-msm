# Project plan: GitHub milestones and issues

Create one GitHub **milestone** per phase and open the issues below (Projects board:
To do / In progress / Review / Done). Months count from data-access activation.

| Milestone | Months | Issues |
|---|---|---|
| M1 Approvals & protocol lock | 1-2 | Ethics submission; ICES feasibility request + itemised quotation; finalise accrual window, chronic-condition algorithms and indicator groups with clinical lead; tag `protocol-v1.0` |
| M2 Simulation & prototype | 3-4 | Pre-register `config/simulation.yaml` with pilot-count calibration; implement semi-Markov (time-since-entry) stress test; implement rare-transition scenarios; run `--mode full` on compute allocation; HMC vs Laplace agreement study on 3 replicates |
| M3 Evaluation & secure prep | 5-7 | Confirm software path inside ICES; map variable names in `sas/01_build_cohort.sas`; cohort flow with disclosure review; IPCW for loss of eligibility in the likelihood sensitivity analysis |
| M4 Empirical analysis & validation | 8-10 | HMC fit (`R/fit_stan.R`) meeting R-hat < 1.01, ESS >= 400, no divergences; temporal validation; subgroup performance where counts permit |
| M5 Sensitivity, papers, release | 11-12 | Sensitivity suite; methods manuscript; applied manuscript; trainee conference abstract; tag `v1.0.0` and archive release on Zenodo (DOI) |

Branching: `main` is protected; work on feature branches, open pull requests, CI must pass.
