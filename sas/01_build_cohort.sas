/*=====================================================================
  01_build_cohort.sas  --  cohort construction inside the ICES environment
  STATUS: TEMPLATE. Not executed (no SAS or ICES access during development).
  Dataset and variable names below are PLACEHOLDERS: map them to the ICES
  data dictionary supplied with the data-access agreement before running.
  Mirrors src/hmsm/cohort.py and config/protocol.yaml (SAP section 1).
=====================================================================*/
%let accrual_start = '01JAN2016'd;   /* finalise after feasibility */
%let accrual_end   = '30JUN2019'd;
%let data_end      = '30JUN2020'd;
%let lookback      = 730;
%let min_age       = 65;
%let min_chronic   = 2;
%let transfer_gap  = 1;

/* libname placeholders -- set by the ICES analyst environment */
/* libname dad   "<ICES path>"; libname nacrs "<ICES path>";        */
/* libname ohip  "<ICES path>"; libname rpdb  "<ICES path>";        */

/* 1. acute DAD records, sorted */
proc sql;
  create table acute as
  select ikn, admdate, ddate, discharge_disp
  from dad.dad_records                         /* PLACEHOLDER */
  where inst_type = 'acute'
  order by ikn, admdate;
quit;

/* 2. collapse inter-hospital transfers into spells */
data spells_long;
  set acute;
  by ikn;
  retain spell prev_end prev_disp;
  if first.ikn then do; spell = 0; prev_end = .; prev_disp = ''; end;
  if not (prev_disp = 'transfer_acute' and admdate - prev_end <= &transfer_gap) then spell + 1;
  prev_end = ddate; prev_disp = discharge_disp;
run;
proc sql;
  create table spells as
  select ikn, spell, min(admdate) as admdate format=date9., max(ddate) as ddate format=date9.
       , max(case when discharge_disp = 'died' then 1 else 0 end) as died_in_spell
  from spells_long group by ikn, spell;
quit;

/* 3. index = first live discharge in accrual window */
proc sort data=spells(where=(died_in_spell = 0 and &accrual_start <= ddate <= &accrual_end)) out=live;
  by ikn ddate;
run;
data idx; set live; by ikn; if first.ikn;
  rename admdate = index_adm ddate = index_date; run;

/* 4. age, eligibility, chronic conditions, prior utilisation */
proc sql;
  create table c1 as
  select i.*, r.bdate, r.sex, r.dthdate, r.elig_start, r.elig_end,
         (i.index_date - r.bdate) / 365.25 as age
  from idx i inner join rpdb.persons r on i.ikn = r.ikn      /* PLACEHOLDER */
  where calculated age >= &min_age
    and r.elig_start <= i.index_adm - &lookback;
  create table chronic as
  select c.ikn, count(distinct o.dxcode) as n_chronic
  from c1 c inner join ohip.claims o on c.ikn = o.ikn       /* PLACEHOLDER */
  where o.servdate < c.index_adm and o.servdate >= c.index_adm - &lookback
    and o.dxcode in ('250','401','428','496','585','290','714','733')  /* PLACEHOLDER algorithm */
  group by c.ikn;
  create table cohort as
  select c.*, coalesce(h.n_chronic, 0) as n_chronic
  from c1 c left join chronic h on c.ikn = h.ikn
  where calculated n_chronic >= &min_chronic
    and (c.dthdate = . or c.dthdate > c.index_date);
quit;

/* 5. cohort flow counts for disclosure-reviewed output (cells < 6 suppressed per ICES rules) */
proc sql;
  select count(*) as n_final from cohort;
quit;

/* 6. follow-up events (adm/dis/enc/dth/cens) with same-day order
      adm < dis < enc < dth, as in config/protocol.yaml; export ONLY inside the
      secure environment to the Python/R analysis area. No record-level export. */
