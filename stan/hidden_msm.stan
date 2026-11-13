// Continuous-time hidden multi-state model: HMC version (proposal: "Bayesian
// estimation will use Hamiltonian Monte Carlo where approved software permits").
// Mirrors src/hmsm/likelihood.py, except that the patient random effect is
// sampled (non-centred) instead of integrated by quadrature.
// STATUS: NOT EXECUTED in the development sandbox (no Stan available).
// Validate against the Python likelihood (R/fit_stan.R, check_against_python)
// before use. Transition order:
// 1 on_C 2 rec_C 3 on_H 4 rec_H 5 adm_S 6 adm_U 7 dis_S 8 dis_U
// 9 dth_CS 10 dth_CU 11 dth_HS 12 dth_HU
// etype: 0 band split, 1 encounter, 2 admission, 3 discharge, 4 death, 5 censor
data {
  int<lower=1> N;                       // patients
  int<lower=1> T;                       // total steps (all patients)
  int<lower=0> K_x;
  int<lower=0> K_w;
  int<lower=1> M;                       // diagnostic indicators
  array[N] int<lower=1> start;          // first step of patient i (1-based)
  array[N] int<lower=1> len;            // number of steps of patient i
  vector<lower=0>[T] dt;
  array[T] int<lower=0, upper=1> care;  // 0 community, 1 hospital
  array[T] int<lower=1, upper=3> band;
  array[T] int<lower=0, upper=5> etype;
  array[T, M] int<lower=-1, upper=1> Y; // -1 = missing / not an encounter
  matrix[N, K_x] X;
  matrix[N, K_w] W;
  vector[12] logq_mu;
  real<lower=0> logq_sd;
  real lam0_mu;
}
parameters {
  vector[12] logq;
  vector[2] kappa_raw;
  vector[K_x] beta_adm;
  vector[K_x] beta_dth;
  real<lower=0> sigma;
  real eta_dth;
  real log_lam0;
  vector[K_w] gamma;
  real alpha;
  real rho;
  vector[M] a;
  vector[M] log_d;                      // d_m = exp(log_d_m) > 0 orders the labels
  real pi0;
  vector[K_x] pi_x;
  vector[N] b_raw;
}
transformed parameters {
  vector[N] b = sigma * b_raw;
}
model {
  vector[3] kap = [0, kappa_raw[1], kappa_raw[2]]';
  vector[N] xa = X * beta_adm;
  vector[N] xd = X * beta_dth;
  vector[N] wg = W * gamma;
  vector[M] dm = exp(log_d);
  // priors (SAP section 5)
  logq ~ normal(logq_mu, logq_sd);
  kappa_raw ~ normal(0, 1);
  beta_adm ~ normal(0, 1);
  beta_dth ~ normal(0, 1);
  sigma ~ normal(0, 1);                 // half-normal via lower bound
  eta_dth ~ normal(0, 1);
  log_lam0 ~ normal(lam0_mu, logq_sd);
  gamma ~ normal(0, 1);
  alpha ~ normal(0, 1);
  rho ~ normal(0, 1);
  a ~ normal(0, 1.5);
  log_d ~ normal(0, 1);
  pi0 ~ normal(-1, 1.5);
  pi_x ~ normal(0, 1.5);
  b_raw ~ std_normal();
  // likelihood: constrained forward recursion
  for (i in 1:N) {
    real pu = inv_logit(pi0 + X[i] * pi_x);
    row_vector[2] f = [1 - pu, pu];
    real fa = exp(xa[i] + b[i]);
    real fd = exp(xd[i] + eta_dth * b[i]);
    vector[2] lam = exp(log_lam0 + wg[i] + rho * b[i] + [0, alpha]');
    for (s in start[i]:(start[i] + len[i] - 1)) {
      vector[12] q = exp(logq + kap[band[s]]);
      matrix[2, 2] A;
      row_vector[2] g;
      vector[2] v = rep_vector(1, 2);
      q[5:6] *= fa;
      q[9:12] *= fd;
      if (care[s] == 0) {
        A = [[-(q[1] + q[5] + q[9] + lam[1]), q[1]],
             [q[2], -(q[2] + q[6] + q[10] + lam[2])]];
      } else {
        A = [[-(q[3] + q[7] + q[11]), q[3]],
             [q[4], -(q[4] + q[8] + q[12])]];
      }
      g = f * matrix_exp(A * dt[s]);
      if (etype[s] == 1) {
        for (z in 0:1) {
          real lf = log(lam[z + 1]);
          for (m in 1:M) {
            if (Y[s, m] == 1) lf += log_inv_logit(a[m] + dm[m] * z);
            else if (Y[s, m] == 0) lf += log1m_inv_logit(a[m] + dm[m] * z);
          }
          v[z + 1] = exp(lf);
        }
      } else if (etype[s] == 2) {
        v = [q[5], q[6]]';
      } else if (etype[s] == 3) {
        v = [q[7], q[8]]';
      } else if (etype[s] == 4) {
        v = care[s] == 0 ? [q[9], q[10]]' : [q[11], q[12]]';
      }
      g = g .* v';
      {
        real tot = sum(g);
        target += log(tot);
        f = g / tot;
      }
    }
  }
}
