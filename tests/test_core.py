"""Core correctness tests (run: pytest -q, or python tests/test_core.py)."""
import os, sys
import numpy as np, pandas as pd
from scipy.linalg import expm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hmsm.model import ModelSpec, generator, rates
from hmsm.likelihood import expm2, loglik_patients
from hmsm.simulate import true_parameters, simulate_cohort, StepData
from hmsm.fit import fit_hidden
from hmsm.predict import standardised_estimands
from hmsm.validation import ipcw_outcome, brier

P = true_parameters({"discrimination": "strong", "informative": True, "heterogeneity": True})


def test_expm2_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(50):
        b, c = rng.exponential(0.3, 2); a = -b - rng.exponential(0.5); d = -c - rng.exponential(0.5)
        t = rng.exponential(5)
        e = np.array(expm2(*map(np.array, (a, b, c, d)), t)).reshape(2, 2)
        assert np.allclose(e, expm(np.array([[a, b], [c, d]]) * t), atol=1e-10)


def test_generator_rows_sum_to_zero():
    r = rates(P, np.zeros((3, 2)), np.zeros((3, 1)), np.array([0.0, 0.5]), np.array([0, 1, 2]))
    for aug in (False, True):
        G = generator(r, augmented=aug)
        assert np.allclose(G.sum(-1), 0)
        assert np.all(G[..., -1, :] == 0) if not aug else True


def test_pack_unpack_roundtrip():
    spec = ModelSpec(2, 1)
    th = spec.pack(P)
    assert np.allclose(spec.pack(spec.unpack(th)), th)


def test_stepdata_exposure_equals_followup():
    cov, ev, _ = simulate_cohort(200, P, np.random.default_rng(1))
    d = StepData(cov, ev, ["age_s", "comorb_s"], ["prior_util_s"])
    end = ev.groupby("id").time.max().loc[d.ids].to_numpy()
    assert np.allclose(d.dt.sum(1), end)


def test_likelihood_finite_and_quadrature_stable():
    cov, ev, _ = simulate_cohort(200, P, np.random.default_rng(2))
    d = StepData(cov, ev, ["age_s", "comorb_s"], ["prior_util_s"])
    spec = ModelSpec(2, 1); th = spec.pack(P)
    l7, l25 = loglik_patients(th, spec, d, 7).sum(), loglik_patients(th, spec, d, 25).sum()
    assert np.isfinite(l7) and abs(l7 - l25) < 0.5


def test_estimands_match_monte_carlo():
    rng = np.random.default_rng(3)
    cov, ev, _ = simulate_cohort(6000, P, rng, elig_rate=0)
    e = standardised_estimands(P, cov[["age_s", "comorb_s"]].to_numpy()[:400],
                               cov[["prior_util_s"]].to_numpy()[:400])
    mc = (ev[ev.event == "dth"].time <= 365).sum() / 6000
    assert abs(e["death_365"] - mc) < 0.02


def test_fit_recovers_alpha():
    cov, ev, _ = simulate_cohort(800, P, np.random.default_rng(4))
    d = StepData(cov, ev, ["age_s", "comorb_s"], ["prior_util_s"])
    spec = ModelSpec(2, 1)
    f = fit_hidden(d, spec)
    j = spec.index["alpha"].start
    assert f["converged"] and abs(f["theta"][j] - P["alpha"]) < 4 * np.sqrt(f["cov"][j, j])


def test_ipcw_brier_without_censoring_is_plain_brier():
    ot = pd.DataFrame({"t_adm": [np.nan] * 4, "t_dth": [10.0, np.nan, 50.0, np.nan],
                       "t_cens": [np.nan, 365.0, np.nan, 365.0]})
    ot["t_end"] = ot[["t_dth", "t_cens"]].min(axis=1)
    y, w = ipcw_outcome(ot, "death", 100)
    p = np.array([0.5, 0.1, 0.7, 0.2])
    assert np.allclose(w, 1) and np.isclose(brier(p, y, w), np.mean((y - p) ** 2))


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)
