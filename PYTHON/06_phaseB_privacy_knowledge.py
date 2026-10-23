"""
Phase B: Step 11 (H5, interview privacy) and Step 12 (knowledge items: attempt vs correct).
Outputs in results/06_*.csv and results/06_summary.json
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from stats_utils import make_X, fit, table, ame_binary, smd, PROVINCES

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
XLSX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/project/WVS_Pak_Data.xlsx")
wide = pd.read_csv(ROOT / "data" / "analysis_wide.csv")
z = lambda s: (s - s.mean()) / s.std()
d = pd.DataFrame({
    "id": wide.id, "psu": wide.psu, "female": wide.female, "y": wide.dk_count_domestic_political,
    "age_z": z(wide.age), "educ_z": z(wide.education_isced), "literate": wide.literate,
    "income_z": z(wide.income_scale), "employed": wide.employed, "urban": wide.urban, "province": wide.province,
    "attention_z": z(wide.attention_index), "digital_z": z(wide.news_digital), "tv_z": z(wide.news_tv),
    "others_present": wide.others_present, "resp_interest_low": wide.resp_interest_low,
    "duration_z": z(wide.duration_min)})
d = d[d.notna().all(axis=1)].reset_index(drop=True)
RES = ["age_z", "educ_z", "literate", "income_z", "employed", "urban"]
AWR = ["attention_z", "digital_z", "tv_z"]
PF = {"province": PROVINCES}
out = {}

# ---------------------------------------------------------------- Step 11a: female x others_present in NB model
desc = d.groupby(["female", "others_present"]).y.agg(["size", "mean"]).round(2)
desc.to_csv(R / "06_privacy_descriptives.csv"); print(desc)
psu_mix = d.groupby("psu").others_present.agg(["min", "max"])
out["psu_with_both_privacy_conditions_%"] = float(((psu_mix["min"] == 0) & (psu_mix["max"] == 1)).mean() * 100)

bp = lambda df: make_X(df, ["female", "others_present"] + RES + AWR + ["resp_interest_low", "duration_z"], PF,
                       [("female", "others_present")])
rp = fit("negbin", d.y, bp(d), cluster=d.psu); tp = table(rp); tp.to_csv(R / "06_negbin_female_x_privacy.csv")
print(tp.loc[["female", "others_present", "female:others_present"]])
eff = []
for f_ in (0, 1):
    sub = d[d.female == f_].copy()
    a = ame_binary(rp, sub, bp, var="others_present")
    eff.append({"group": "women" if f_ else "men", "pred_private": round(a["pred0"], 2),
                "pred_others_present": round(a["pred1"], 2), "difference": round(a["ame"], 2),
                "ci_low": round(a["ci_low"], 2), "ci_high": round(a["ci_high"], 2)})
eff = pd.DataFrame(eff); eff.to_csv(R / "06_privacy_effect_by_sex.csv", index=False); print(eff)

# ---------------------------------------------------------------- Step 11b: propensity-score matching within sex
COV = RES + AWR
def psm(df, caliper_sd=0.2):
    Xps = make_X(df, COV, PF)
    r = fit("logit", df.others_present, Xps)
    lp = np.asarray(Xps, float) @ r["coef"]
    t_idx = np.where(df.others_present == 1)[0]; c_idx = np.where(df.others_present == 0)[0]
    cal = caliper_sd * lp.std()
    dist = np.abs(lp[t_idx][:, None] - lp[c_idx][None, :])
    j = dist.argmin(1); okm = dist[np.arange(len(t_idx)), j] <= cal
    t_m, c_m = t_idx[okm], c_idx[j[okm]]
    att = df.y.values[t_m].mean() - df.y.values[c_m].mean()
    return att, t_m, c_m, Xps
def cluster_boot(df, fn, reps=500, seed=11):
    rng = np.random.default_rng(seed); groups = df.psu.unique(); idx = df.groupby("psu").indices; res = []
    for _ in range(reps):
        bd = pd.concat([df.iloc[idx[g]] for g in rng.choice(groups, len(groups))], ignore_index=True)
        try: res.append(fn(bd))
        except Exception: pass
    return np.array(res)

rows, bal_rows = [], []
for f_, lab in [(1, "women"), (0, "men")]:
    sd_ = d[d.female == f_].reset_index(drop=True)
    att, t_m, c_m, Xps = psm(sd_)
    bs = cluster_boot(sd_, lambda bd: psm(bd)[0])
    rows.append({"group": lab, "treated_others_present": int(sd_.others_present.sum()), "matched_treated": len(t_m),
                 "ATT_extra_dk_answers": round(att, 3), "ci_low": round(np.quantile(bs, .025), 3),
                 "ci_high": round(np.quantile(bs, .975), 3), "boot_reps": len(bs)})
    Xa = Xps.drop(columns="(Intercept)")
    for c in Xa.columns:
        tb = Xa[c].values[sd_.others_present == 1]; cb = Xa[c].values[sd_.others_present == 0]
        bal_rows.append({"group": lab, "covariate": c, "smd_before": round(smd(tb, cb), 3),
                         "smd_after": round(smd(Xa[c].values[t_m], Xa[c].values[c_m]), 3)})
psm_df = pd.DataFrame(rows); bal = pd.DataFrame(bal_rows)
psm_df.to_csv(R / "06_psm_privacy.csv", index=False); bal.to_csv(R / "06_psm_balance.csv", index=False)
print(psm_df.to_string(index=False))
out["max_abs_smd_after_matching"] = float(bal.smd_after.abs().max())
out["max_abs_smd_before_matching"] = float(bal.smd_before.abs().max())

# ---------------------------------------------------------------- Step 12: knowledge items
raw = pd.read_excel(XLSX, sheet_name="original", usecols=lambda c: c.split(":")[0] in ("D_INTERVIEW", "Q91", "Q92", "Q93"))
raw.columns = [c.split(":")[0] for c in raw.columns]
# Correct answers: Q91 asks which country is NOT a permanent member of the UN Security Council
# (WVS-7 questionnaire) -> India (code 3); Q92 IMF headquarters -> Washington DC (1);
# Q93 Amnesty International -> human rights (2). Verify Q91 wording against the WVS-7 master questionnaire.
KEY = {"Q91": 3, "Q92": 1, "Q93": 2}
kn = raw.melt(id_vars="D_INTERVIEW", var_name="item", value_name="code").rename(columns={"D_INTERVIEW": "id"})
kn["attempt"] = (kn.code > 0).astype(int)
kn["correct"] = (kn.code == kn.item.map(KEY)).astype(int)
kn = kn.merge(d, on="id")
rates = kn.groupby(["item", "female"]).agg(attempt_pct=("attempt", "mean"),
        correct_pct_all=("correct", "mean")).mul(100).round(1)
rates["correct_pct_of_attempts"] = (kn[kn.attempt == 1].groupby(["item", "female"]).correct.mean() * 100).round(1)
rates.to_csv(R / "06_knowledge_rates.csv"); print(rates)
IF = {"province": PROVINCES, "item": ["Q91", "Q92", "Q93"]}
bk = lambda df: make_X(df, ["female"] + RES + AWR, IF)
kr = []
for stage, data in [("attempt (all respondents)", kn), ("correct (among attempts)", kn[kn.attempt == 1])]:
    data = data.reset_index(drop=True); yv = data.attempt if stage.startswith("attempt") else data.correct
    r = fit("logit", yv, bk(data), cluster=data.psu); t = table(r)
    a = ame_binary(r, data, bk)
    kr.append({"stage": stage, "n_rows": len(data), "OR_female": t.loc["female", "exp_b"],
               "OR_ci": f"{t.loc['female','exp_ci_low']:.2f}-{t.loc['female','exp_ci_high']:.2f}", "p": t.loc["female", "p"],
               "pred_men_%": round(100 * a["pred0"], 1), "pred_women_%": round(100 * a["pred1"], 1),
               "AME_pp": round(100 * a["ame"], 1), "AME_ci_pp": f"{100*a['ci_low']:.1f} to {100*a['ci_high']:.1f}"})
    t.to_csv(R / f"06_logit_knowledge_{stage.split()[0]}.csv")
kr = pd.DataFrame(kr); kr.to_csv(R / "06_knowledge_models.csv", index=False); print(kr.to_string(index=False))
(R / "06_summary.json").write_text(json.dumps(out, indent=1)); print(out)
