# HMC fit of the hidden multi-state model with cmdstanr (SAP section 5).
# STATUS: written for the approved secure environment; NOT executed in the
# development sandbox. Requires: cmdstanr, posterior, jsonlite, and CmdStan.
#   Rscript R/fit_stan.R            (after `python scripts/09_export_stan_data.py`)
suppressPackageStartupMessages({
  library(cmdstanr); library(posterior); library(jsonlite)
})
set.seed(20261112)
dat  <- fromJSON("data/derived/stan/stan_data.json", simplifyVector = TRUE)
init <- fromJSON("data/derived/stan/init_laplace.json", simplifyVector = TRUE)
dat$Y <- matrix(as.integer(dat$Y), ncol = dat$M)            # T x M
dat$X <- matrix(dat$X, nrow = dat$N); dat$W <- matrix(dat$W, nrow = dat$N)

mod <- cmdstan_model("stan/hidden_msm.stan", cpp_options = list(stan_threads = FALSE))
jitter_init <- function(chain_id) {                          # disperse around the Laplace mode
  lapply(init, function(v) if (is.numeric(v) && length(v) < 50) v + rnorm(length(v), 0, 0.1) else v)
}
fit <- mod$sample(data = dat, init = jitter_init, chains = 4, parallel_chains = 4,
                  iter_warmup = 1000, iter_sampling = 1000, adapt_delta = 0.9,
                  max_treedepth = 11, seed = 20261112, refresh = 100)

# ---- convergence criteria (SAP section 5): failure => reparameterise / simplify,
#      never drop chains selectively
main <- c("logq", "kappa_raw", "beta_adm", "beta_dth", "sigma", "eta_dth",
          "log_lam0", "gamma", "alpha", "rho", "a", "log_d", "pi0", "pi_x")
summ <- fit$summary(variables = main, "mean", "sd", ~quantile(.x, c(.025, .975)),
                    "rhat", "ess_bulk", "ess_tail")
diag <- fit$diagnostic_summary()
dir.create("results/hmc", showWarnings = FALSE, recursive = TRUE)
write.csv(summ, "results/hmc/hmc_summary.csv", row.names = FALSE)
ok <- all(summ$rhat < 1.01, na.rm = TRUE) && all(summ$ess_bulk >= 400, na.rm = TRUE) &&
      all(summ$ess_tail >= 400, na.rm = TRUE) && sum(diag$num_divergent) == 0
cat(sprintf("R-hat max %.3f | min bulk ESS %.0f | min tail ESS %.0f | divergences %d | PASS = %s\n",
            max(summ$rhat, na.rm = TRUE), min(summ$ess_bulk, na.rm = TRUE),
            min(summ$ess_tail, na.rm = TRUE), sum(diag$num_divergent), ok))
# posterior draws for estimands: compute with the Python predictor
#   (src/hmsm/predict.py: estimand_draws) to keep one implementation of the estimands.
draws <- as_draws_df(fit$draws(variables = main))
write.csv(draws, "results/hmc/hmc_draws.csv", row.names = FALSE)

# ---- agreement with the Laplace pilot (sanity check)
lap <- fromJSON("data/derived/stan/init_laplace.json")
cmp <- data.frame(param = c("alpha", "sigma", "rho"),
                  laplace_mode = c(lap$alpha, lap$sigma, lap$rho),
                  hmc_mean = c(mean(draws$alpha), mean(draws$sigma), mean(draws$rho)))
print(cmp)
