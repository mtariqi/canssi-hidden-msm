"""Synthetic ICES-style linked tables and the protocol-locked cohort builder.

IMPORTANT: everything produced by `make_synthetic_ices` is SYNTHETIC. Table and
variable names are simplified stand-ins for DAD, NACRS, OHIP and RPDB; they do
not reproduce ICES data dictionaries. Inside the ICES environment the cohort is
built with the SAS template in sas/01_build_cohort.sas (or this module, if
Python is approved), after mapping real variable names in config/protocol.yaml.

Cohort definition (proposal 'Data and cohort'; SAP section 1)
  * index = first live discharge from an acute-care spell in the accrual window
    (inter-hospital transfers collapsed into one spell)
  * age >= 65 at index; >= 2 chronic conditions in a 2-year lookback
  * continuous OHIP eligibility through the lookback
  * follow-up to 365 days, death, loss of eligibility or end of data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate import simulate_histories

DX = {  # indicator groups (clinically distinct, non-overlapping) -> codes
    "y1": {"ohip": "428", "icd10": "I50"},   # heart-failure decompensation
    "y2": {"ohip": "491", "icd10": "J44"},   # COPD exacerbation / resp. failure
    "y3": {"ohip": "290", "icd10": "F05"},   # delirium / acute confusion
}
CHRONIC_OHIP = ["250", "401", "428", "496", "585", "290", "714", "733"]
ROUTINE_OHIP = ["V70", "917", "780"]


def make_synthetic_ices(n_people, p, rng, accrual=("2015-01-01", "2019-12-31"),
                        data_end="2020-06-30"):
    """Generate DAD / NACRS / OHIP / RPDB-like tables for a synthetic population."""
    acc0, acc1 = pd.Timestamp(accrual[0]), pd.Timestamp(accrual[1])
    dend = pd.Timestamp(data_end)
    ids = np.arange(1, n_people + 1)
    # index admission dates spread a little beyond the accrual window
    span = (acc1 - acc0).days + 240
    adm0 = acc0 - pd.Timedelta(days=120) + pd.to_timedelta(rng.integers(0, span, n_people), "D")
    los0 = rng.integers(2, 15, n_people)
    dis0 = adm0 + pd.to_timedelta(los0, "D")
    age_at = rng.normal(77, 8, n_people).clip(50, 100)
    birth = dis0 - pd.to_timedelta((age_at * 365.25).astype(int), "D")
    sex = rng.choice(["F", "M"], n_people)
    n_chronic = rng.poisson(2.6, n_people)
    prior_visits = rng.negative_binomial(3, 0.25, n_people)
    elig_start = dis0 - pd.to_timedelta(rng.choice([4000, 4000, 4000, 500], n_people), "D")
    inst_type = rng.choice(["acute", "acute", "acute", "acute", "acute", "rehab"], n_people)
    died_in_index = rng.random(n_people) < 0.04

    # standardised covariates for the generating model (population scale)
    age_s = (age_at - 77) / 8
    com_s = (n_chronic - 2.6) / 1.6
    pu_s = (np.log1p(prior_visits) - 2.1) / 0.6
    X = np.column_stack([age_s, (sex == "M").astype(float), com_s, pu_s])
    W = np.column_stack([pu_s, com_s])
    elig_loss = rng.exponential(365 / 0.03, n_people)
    to_end = (dend - dis0).days.to_numpy().astype(float)
    cens = np.minimum.reduce([np.full(n_people, 365.0), elig_loss, np.maximum(to_end, 0.0)])
    ev, truth = simulate_histories(X, W, p, rng, cens, dep_sd=0.3, miss_y=0.05)

    # ---------------- DAD (acute spells, including index) ----------------
    dad = [pd.DataFrame({"ikn": ids, "admdate": adm0, "ddate": dis0, "inst_type": inst_type,
                         "dx10main": rng.choice(["I50", "J44", "N39", "S72", "I63"], n_people),
                         "discharge_disp": np.where(died_in_index, "died", "home")})]
    # a subset of index stays involve an inter-hospital transfer (two records)
    tr_mask = rng.random(n_people) < 0.08
    if tr_mask.any():
        first_end = adm0[tr_mask] + pd.to_timedelta(np.maximum(los0[tr_mask] // 2, 1), "D")
        dad[0].loc[tr_mask, "ddate"] = first_end
        dad[0].loc[tr_mask, "discharge_disp"] = "transfer_acute"
        dad.append(pd.DataFrame({"ikn": ids[tr_mask], "admdate": first_end, "ddate": dis0[tr_mask],
                                 "inst_type": "acute", "dx10main": "I50",
                                 "discharge_disp": np.where(died_in_index[tr_mask], "died", "home")}))
    # post-index events -> calendar dates (date-only records)
    ev = ev.copy()
    ev["ikn"] = ids[ev["id"].to_numpy()]
    ev["date"] = dis0[ev["id"].to_numpy()] + pd.to_timedelta(np.floor(ev["time"]).astype(int), "D")
    adm = ev[ev.event == "adm"].copy(); dis = ev[ev.event == "dis"].copy()
    adm["k"] = adm.groupby("ikn").cumcount(); dis["k"] = dis.groupby("ikn").cumcount()
    sp = adm.merge(dis[["ikn", "k", "date"]], on=["ikn", "k"], how="left", suffixes=("", "_d"))
    dth = ev[ev.event == "dth"].set_index("ikn")["date"]
    # spells without discharge end in death or are open at censoring
    open_end = sp["date_d"].isna()
    sp.loc[open_end, "date_d"] = sp.loc[open_end, "ikn"].map(dth)
    disp = np.where(sp["date_d"].isna(), "still_in", np.where(open_end, "died", "home"))
    sp["date_d"] = sp["date_d"].fillna(dend)
    dad.append(pd.DataFrame({"ikn": sp.ikn, "admdate": sp.date, "ddate": sp.date_d,
                             "inst_type": "acute", "dx10main": "I50", "discharge_disp": disp}))
    dad = pd.concat(dad, ignore_index=True)

    # ---------------- NACRS / OHIP encounters ----------------
    enc = ev[ev.event == "enc"].copy()
    is_ed = rng.random(len(enc)) < 0.2
    rows_n, rows_o = [], []
    for (ikn, date, y1, y2, y3, ed) in zip(enc.ikn, enc.date, enc.y1, enc.y2, enc.y3, is_ed):
        ys = {"y1": y1, "y2": y2, "y3": y3}
        if min(ys.values()) < 0:                            # diagnosis field missing
            codes = [None]
        else:
            codes = [DX[k]["ohip" if not ed else "icd10"] for k, v in ys.items() if v == 1]
            if not codes:
                codes = [rng.choice(ROUTINE_OHIP) if not ed else "R69"]
        for c in codes:
            (rows_n if ed else rows_o).append((ikn, date, c))
    nacrs = pd.DataFrame(rows_n, columns=["ikn", "regdate", "dx10"])
    ohip_fu = pd.DataFrame(rows_o, columns=["ikn", "servdate", "dxcode"])
    # lookback claims carrying chronic-condition codes and prior utilisation
    lb = []
    for i in range(n_people):
        k = min(n_chronic[i], len(CHRONIC_OHIP))
        for code in rng.choice(CHRONIC_OHIP, k, replace=False):
            for _ in range(2):
                lb.append((ids[i], adm0[i] - pd.Timedelta(days=int(rng.integers(30, 700))), code))
        for _ in range(prior_visits[i]):
            lb.append((ids[i], adm0[i] - pd.Timedelta(days=int(rng.integers(1, 365))), "V70"))
    ohip = pd.concat([ohip_fu, pd.DataFrame(lb, columns=["ikn", "servdate", "dxcode"])],
                     ignore_index=True)

    # ---------------- RPDB ----------------
    death = pd.Series(pd.NaT, index=ids)
    death.loc[dth.index] = dth.values
    death.loc[ids[died_in_index]] = dis0[died_in_index]
    elig_end = dis0 + pd.to_timedelta(np.where(elig_loss < 365, elig_loss, 9999).astype(int), "D")
    rpdb = pd.DataFrame({"ikn": ids, "bdate": birth, "sex": sex, "dthdate": death.values,
                         "elig_start": elig_start, "elig_end": elig_end})
    truth = truth.assign(ikn=ids[truth["id"].to_numpy()])
    world_cov = pd.DataFrame({"ikn": ids, "age_s": age_s, "male": X[:, 1], "comorb_s": com_s,
                              "prior_util_s": pu_s})   # generating-scale covariates (truth only)
    return {"dad": dad, "nacrs": nacrs, "ohip": ohip, "rpdb": rpdb, "truth": truth,
            "world_cov": world_cov}


# ======================================================================
# cohort builder
# ======================================================================
def collapse_transfers(dad, gap_days=1):
    """Collapse inter-hospital transfers into admission spells."""
    d = dad[dad.inst_type == "acute"].sort_values(["ikn", "admdate"]).copy()
    prev_end = d.groupby("ikn")["ddate"].shift()
    prev_disp = d.groupby("ikn")["discharge_disp"].shift()
    new_spell = ~((prev_disp == "transfer_acute") & ((d.admdate - prev_end).dt.days <= gap_days))
    d["spell"] = new_spell.groupby(d.ikn).cumsum()
    sp = d.groupby(["ikn", "spell"]).agg(admdate=("admdate", "min"), ddate=("ddate", "max"),
                                         discharge_disp=("discharge_disp", "last")).reset_index()
    return sp


def build_cohort(tables, protocol):
    """Apply the locked protocol. Returns (covariates, events, flow table, spells)."""
    dad, nacrs, ohip, rpdb = (tables[k] for k in ["dad", "nacrs", "ohip", "rpdb"])
    acc0, acc1 = pd.Timestamp(protocol["accrual_start"]), pd.Timestamp(protocol["accrual_end"])
    dend = pd.Timestamp(protocol["data_end"])
    flow = []
    flow.append(("Persons with >=1 hospital record (all institution types)", dad.ikn.nunique()))
    spells = collapse_transfers(dad, protocol["transfer_gap_days"])
    flow.append(("  with an acute-care spell (non-acute excluded; transfers collapsed)", spells.ikn.nunique()))
    live = spells[(spells.discharge_disp != "died") & (spells.discharge_disp != "still_in")]
    live = live[(live.ddate >= acc0) & (live.ddate <= acc1)]
    idx = live.sort_values("ddate").groupby("ikn").head(1).rename(
        columns={"admdate": "index_adm", "ddate": "index_date"})
    flow.append(("  with a live acute discharge in the accrual window (first = index)", len(idx)))
    c = idx.merge(rpdb, on="ikn")
    c["age"] = (c.index_date - c.bdate).dt.days / 365.25
    c = c[c.age >= protocol["min_age"]]
    flow.append((f"  aged >= {protocol['min_age']} at index", len(c)))
    lbd = protocol["lookback_days"]
    c = c[c.elig_start <= c.index_adm - pd.Timedelta(days=lbd)]
    flow.append((f"  continuously eligible for {lbd}-day lookback", len(c)))
    o = ohip.merge(c[["ikn", "index_adm"]], on="ikn")
    lb = o[(o.servdate < o.index_adm) & (o.servdate >= o.index_adm - pd.Timedelta(days=lbd))]
    chronic = lb[lb.dxcode.isin(protocol["chronic_codes"])].groupby("ikn").dxcode.nunique()
    c["n_chronic"] = c.ikn.map(chronic).fillna(0).astype(int)
    c = c[c.n_chronic >= protocol["min_chronic"]]
    flow.append((f"  with >= {protocol['min_chronic']} chronic conditions in lookback", len(c)))
    yr = lb[lb.servdate >= lb.index_adm - pd.Timedelta(days=365)]
    c["prior_visits"] = c.ikn.map(yr.groupby("ikn").size()).fillna(0)
    c = c[c.dthdate.isna() | (c.dthdate > c.index_date)]
    flow.append(("Final analytic cohort", len(c)))

    # ---------------- follow-up events ----------------
    c = c.reset_index(drop=True)
    c["fu_end"] = pd.concat([c.index_date + pd.Timedelta(days=365), c.elig_end,
                             pd.Series(dend, index=c.index)], axis=1).min(axis=1)
    sp = spells.merge(c[["ikn", "index_date", "fu_end"]], on="ikn")
    sp = sp[(sp.admdate >= sp.index_date) & (sp.admdate <= sp.fu_end)]  # index spell ends on index_date
    ev = [pd.DataFrame({"ikn": sp.ikn, "date": sp.admdate, "event": "adm"}),
          pd.DataFrame({"ikn": sp.ikn, "date": sp.ddate, "event": "dis"})[
              (sp.discharge_disp == "home").to_numpy() & (sp.ddate <= sp.fu_end).to_numpy()]]
    # encounters: union of indicator codes per person-day
    code2ind = {v["ohip"]: k for k, v in DX.items()} | {v["icd10"]: k for k, v in DX.items()}
    e1 = ohip.rename(columns={"servdate": "date", "dxcode": "code"})
    e2 = nacrs.rename(columns={"regdate": "date", "dx10": "code"})
    enc = pd.concat([e1, e2]).merge(c[["ikn", "index_date", "fu_end"]], on="ikn")
    enc = enc[(enc.date >= enc.index_date) & (enc.date <= enc.fu_end)]
    for k in DX:
        enc[k] = (enc.code.map(code2ind) == k).astype(int)
    enc["missing_dx"] = enc.code.isna()
    enc = enc.groupby(["ikn", "date"]).agg(y1=("y1", "max"), y2=("y2", "max"), y3=("y3", "max"),
                                          miss=("missing_dx", "all")).reset_index()
    for k in DX:
        enc.loc[enc.miss, k] = -1
    enc["event"] = "enc"
    ev.append(enc[["ikn", "date", "event", "y1", "y2", "y3"]])
    d = c[c.dthdate.notna() & (c.dthdate <= c.fu_end)]
    ev.append(pd.DataFrame({"ikn": d.ikn, "date": d.dthdate, "event": "dth"}))
    ev = pd.concat(ev, ignore_index=True)
    ev = ev.merge(c[["ikn", "index_date", "fu_end"]], on="ikn")
    ev["time"] = (ev.date - ev.index_date).dt.days.astype(float)
    order = {e: i for i, e in enumerate(protocol["same_day_order"])}
    ev["ord"] = ev.event.map(order)
    ev = ev.sort_values(["ikn", "time", "ord"])
    # drop contacts while in hospital (attributed to the spell) and after death
    ev = _drop_inpatient_contacts(ev)
    # censoring record
    has_dth = set(ev.loc[ev.event == "dth", "ikn"])
    cz = c[~c.ikn.isin(has_dth)]
    cens = pd.DataFrame({"ikn": cz.ikn, "event": "cens",
                         "time": (cz.fu_end - cz.index_date).dt.days.astype(float)})
    ev = pd.concat([ev, cens], ignore_index=True)
    for k in DX:
        ev[k] = ev[k].fillna(-1).astype(int)
    ev = ev.sort_values(["ikn", "time"], kind="stable")
    ev["id"] = ev["ikn"]
    cov = c.assign(id=c.ikn, male=(c.sex == "M").astype(float),
                   log_prior_visits=np.log1p(c.prior_visits),
                   index_year=c.index_date.dt.year)
    return cov, ev[["id", "time", "event", "y1", "y2", "y3"]], pd.DataFrame(flow, columns=["step", "n"]), spells


def _drop_inpatient_contacts(ev):
    keep, prev_ikn, inhosp, dead = [], None, False, False
    for ikn, e in zip(ev.ikn.to_numpy(), ev.event.to_numpy()):
        if ikn != prev_ikn:
            inhosp, dead, prev_ikn = False, False, ikn
        if dead:
            keep.append(False); continue
        if e == "adm":
            keep.append(not inhosp); inhosp = True
        elif e == "dis":
            keep.append(inhosp); inhosp = False
        elif e == "enc":
            keep.append(not inhosp)
        else:
            keep.append(True); dead = e == "dth"
    return ev[np.array(keep)]
