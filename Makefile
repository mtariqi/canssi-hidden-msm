# End-to-end pipeline on SYNTHETIC data. `make all` reproduces every table/figure.
PY ?= python
.PHONY: all data cohort prior sim summary fit validate sensitivity budget stan test clean
all: data cohort prior sim summary fit validate sensitivity budget
data:        ; $(PY) scripts/00_make_synthetic_ices.py
cohort:      ; $(PY) scripts/01_build_cohort.py
prior:       ; $(PY) scripts/02_prior_predictive.py
sim:         ; $(PY) scripts/03_run_simulation.py --mode pilot && $(PY) scripts/03_run_simulation.py --mode pilot --stress
summary:     ; $(PY) scripts/04_summarize_simulation.py pilot
fit:         ; $(PY) scripts/05_fit_empirical.py
validate:    ; $(PY) scripts/06_validate.py
sensitivity: ; $(PY) scripts/05_fit_empirical.py _sens_contact_first && $(PY) scripts/07_sensitivity.py all
budget:      ; $(PY) scripts/08_budget_plan.py
stan:        ; $(PY) scripts/09_export_stan_data.py && Rscript R/fit_stan.R
test:        ; $(PY) -m pytest -q tests
clean:       ; rm -rf data/synthetic/* data/derived/* results/sim/* results/fits/*
