"""
Phase B: Step 6 (diagnostics), Step 7 (H1), Step 9 (H3: silence vs ignorance).
Outcome: number of "don't know" answers on the 59 domestic political items.
Sample: complete cases on all model variables (same sample for every nested model).
Outputs in results/05_*.csv, results/05_*.png, results/05_summary.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats, special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stats_utils import make_X, fit, table, ame_binary, PROVINCES

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
wide = pd.read_csv(ROOT / "data" / "analysis_wide.csv")
z = lambda s: (s - s.mean()) / s.std()
d = pd.DataFrame({
    "id": wide.id, "psu": wide.psu, "female": wide.female, "y": wide.dk_count_domestic_political,
    "share": wide.dk_share_domestic_political, "age_z": z(wide.age), "educ_z": z(wide.education_isced),
    "literate": wide.literate, "income_z": z(wide.income_scale), "employed": wide.employed, "urban": wide.urban,
    "province": wide.province, "attention_z": z(wide.attention_index), "digital_z": z(wide.news_digital),
    "tv_z": z(wide.news_tv), "others_present": wide.others_present, "resp_interest_low": wide.resp_interest_low,
    "duration_z": z(wide.duration_min), "high_awareness": wide.high_awareness})
d = d[d.drop(columns="high_awareness").notna().all(axis=1)].reset_index(drop=True)
S = {"n": len(d), "n_women": int(d.female.sum()), "n_men": int((1 - d.female).sum())}
print("Analysis sample:", S)

RES = ["age_z", "educ_z", "literate", "income_z", "employed", "urban"]
AWR = ["attention_z", "digital_z", "tv_z"]
INT = ["others_present", "resp_interest_low", "duration_z"]
PF = {"province": PROVINCES}
spec = {"M0": [], "M1": RES, "M2": RES + AWR, "M3": RES + AWR + INT}
build = {m: (lambda df, v=v, m=m: make_X(df, ["female"] + v, PF if m != "M0" else None)) for m, v in spec.items()}

# ---------------------------------------------------------------- Step 6: diagnostics
diag = {}
Xf = build["M3"](d)
rp = fit("poisson", d.y, Xf, cluster=d.psu)
rn = fit("negbin", d.y, Xf, cluster=d.psu)
lr = 2 * (rn["ll"] - rp["ll"])
diag["alpha_negbin"] = float(np.exp(rn["coef"][-1]))
diag["LR_poisson_vs_negbin"] = float(lr); diag["LR_p_boundary"] = float(stats.chi2.sf(lr, 1) / 2)
mu = np.exp(np.asarray(Xf) @ rp["coef"])
aux = ((d.y - mu) ** 2 - d.y) / mu                      # Cameron-Trivedi auxiliary regression
b = (aux * mu).sum() / (mu ** 2).sum(); res = aux - b * mu
se = np.sqrt((res ** 2).sum() / (len(d) - 1) / (mu ** 2).sum())
diag["cameron_trivedi_t"] = float(b / se); diag["cameron_trivedi_p"] = float(2 * stats.t.sf(abs(b / se), len(d) - 1))
mun = np.exp(np.asarray(Xf) @ rn["coef"][:-1]); r_ = 1 / diag["alpha_negbin"]
diag["zeros_observed_%"] = float((d.y == 0).mean() * 100)
diag["zeros_poisson_%"] = float(np.exp(-mu).mean() * 100)
diag["zeros_negbin_%"] = float(((r_ / (r_ + mun)) ** r_).mean() * 100)
Xv = Xf.drop(columns="(Intercept)")
vif = {}
for c in Xv.columns:
    o = Xv.drop(columns=c); A = np.c_[np.ones(len(o)), o.values]
    coef, *_ = np.linalg.lstsq(A, Xv[c].values, rcond=None)
    r2 = 1 - ((Xv[c] - A @ coef) ** 2).sum() / ((Xv[c] - Xv[c].mean()) ** 2).sum()
    vif[c] = round(1 / (1 - r2), 2)
diag["VIF_max"] = max(vif.values()); diag["VIF"] = vif
g = wide.groupby("psu").dk_share_domestic_political; k_ = g.size().mean()
msb = (g.size() * (g.mean() - wide.dk_share_domestic_political.mean()) ** 2).sum() / (g.ngroups - 1)
msw = g.var(ddof=1).mul(g.size() - 1).sum() / (len(wide) - g.ngroups)
diag["ICC_psu_domestic_share"] = float((msb - msw) / (msb + (k_ - 1) * msw))
prov = (wide.assign(sex=wide.female.map({0: "men", 1: "women"}))
        .pivot_table(index="province", columns="sex", values="dk_share_domestic_political", aggfunc="mean") * 100).round(2)
prov["n"] = wide.province.value_counts(); prov["gap_pp"] = prov.women - prov.men
prov.to_csv(R / "05_dk_by_province_sex.csv")
print("\nDIAGNOSTICS", json.dumps({k: v for k, v in diag.items() if k != "VIF"}, indent=1))
print(prov)

# ---------------------------------------------------------------- Steps 7 and 9a: nested NB models M0-M3
fits = {}; rows = []
for m in spec:
    fits[m] = fit("negbin", d.y, build[m](d), cluster=d.psu)
    t = table(fits[m]); t.to_csv(R / f"05_negbin_{m}.csv")
    a = ame_binary(fits[m], d, build[m])
    rows.append({"model": m, "female_b": t.loc["female", "estimate"], "female_se": t.loc["female", "se"],
                 "IRR": t.loc["female", "exp_b"], "IRR_ci": f"{t.loc['female','exp_ci_low']:.2f}-{t.loc['female','exp_ci_high']:.2f}",
                 "p": t.loc["female", "p"], "AME_count": round(a["ame"], 3),
                 "AME_ci": f"{a['ci_low']:.2f}-{a['ci_high']:.2f}", "pred_men": round(a["pred0"], 2),
                 "pred_women": round(a["pred1"], 2), "alpha": round(float(np.exp(fits[m]["coef"][-1])), 3),
                 "loglik": round(fits[m]["ll"], 1), "converged": fits[m]["converged"]})
stepwise = pd.DataFrame(rows); stepwise.to_csv(R / "05_stepwise_female.csv", index=False)
print("\nSTEPWISE (female effect)\n", stepwise.to_string(index=False))
full = pd.concat({m: table(fits[m])[["exp_b", "p"]] for m in spec}, axis=1)
full.to_csv(R / "05_negbin_M0_M3_IRR.csv"); print(full.round(3).to_string())

# ---------------------------------------------------------------- Step 9b: KHB decomposition
def ols_resid(yv, Xdf):
    A = np.asarray(Xdf, float); c, *_ = np.linalg.lstsq(A, np.asarray(yv, float), rcond=None)
    return np.asarray(yv, float) - A @ c

def khb(df, base_vars, mediators):
    """Female coefficient in full model vs reduced model with mediators replaced by their
    residuals from (female + base controls). Difference = part of the gap running through mediators."""
    Xb_ = make_X(df, ["female"] + base_vars, PF)
    dd = df.copy()
    for m_ in mediators: dd[m_ + "_res"] = ols_resid(df[m_], Xb_)
    full_ = fit("negbin", dd.y, make_X(dd, ["female"] + base_vars + mediators, PF))
    red_ = fit("negbin", dd.y, make_X(dd, ["female"] + base_vars + [m_ + "_res" for m_ in mediators], PF))
    bf, br = full_["coef"][1], red_["coef"][1]
    return br, bf, br - bf

def cluster_boot(df, fn, reps=300, seed=7):
    rng = np.random.default_rng(seed); groups = df.psu.unique(); idx = df.groupby("psu").indices; out = []
    for _ in range(reps):
        pick = rng.choice(groups, len(groups), replace=True)
        bd = pd.concat([df.iloc[idx[g_]] for g_ in pick], ignore_index=True)
        try: out.append(fn(bd))
        except Exception: pass
    return np.array(out)

khb_rows = []
for label, base, med in [("awareness, net of resources", RES, AWR),
                         ("resources + awareness", [], RES + AWR)]:
    br, bf, ind = khb(d, base, med)
    bs = cluster_boot(d, lambda bd: khb(bd, base, med)[2], reps=300)
    khb_rows.append({"mediators": label, "female_b_reduced": round(br, 4), "female_b_full": round(bf, 4),
                     "indirect_b": round(ind, 4), "indirect_ci_low": round(np.quantile(bs, .025), 4),
                     "indirect_ci_high": round(np.quantile(bs, .975), 4), "pct_of_total_explained": round(100 * ind / br, 1),
                     "boot_reps": len(bs)})
khb_df = pd.DataFrame(khb_rows); khb_df.to_csv(R / "05_khb.csv", index=False)
print("\nKHB\n", khb_df.to_string(index=False))

# ---------------------------------------------------------------- Step 9c: Oaxaca-Blinder (pooled, twofold) on % don't know
def oaxaca(df):
    yv = df.share * 100
    Xo = make_X(df, RES + AWR, PF); Xp = Xo.assign(female=df.female)   # pooled model incl. group dummy (Jann 2008)
    c, *_ = np.linalg.lstsq(Xp.values, yv.values, rcond=None); beta = pd.Series(c[:-1], index=Xo.columns)
    dx = Xo[df.female == 1].mean() - Xo[df.female == 0].mean()
    contrib = dx * beta; gap = yv[df.female == 1].mean() - yv[df.female == 0].mean()
    res_cols = [c_ for c_ in Xo.columns if c_ in RES or c_.startswith("province")]
    return {"gap": gap, "explained_total": contrib.sum(), "explained_resources": contrib[res_cols].sum(),
            "explained_awareness": contrib[AWR].sum(), "unexplained": gap - contrib.sum()}
oa = oaxaca(d)
ob = pd.DataFrame(list(cluster_boot(d, oaxaca, reps=1000)))
oax = pd.DataFrame({"estimate_pp": pd.Series(oa), "ci_low": ob.quantile(.025), "ci_high": ob.quantile(.975)}).round(3)
oax["pct_of_gap"] = (100 * oax.estimate_pp / oa["gap"]).round(1)
oax.to_csv(R / "05_oaxaca.csv"); print("\nOAXACA (percentage points of domestic DK share)\n", oax)

# ---------------------------------------------------------------- Step 9d: high-awareness subgroup
sub_rows = []
for lab, mask in [("high awareness", d.high_awareness == 1), ("not high awareness", d.high_awareness == 0)]:
    sd_ = d[mask].reset_index(drop=True)
    rr = fit("negbin", sd_.y, build["M1"](sd_), cluster=sd_.psu); t = table(rr)
    a = ame_binary(rr, sd_, build["M1"])
    sub_rows.append({"group": lab, "n": len(sd_), "women": int(sd_.female.sum()), "IRR_female": t.loc["female", "exp_b"],
                     "IRR_ci": f"{t.loc['female','exp_ci_low']:.2f}-{t.loc['female','exp_ci_high']:.2f}",
                     "p": t.loc["female", "p"], "pred_men": round(a["pred0"], 2), "pred_women": round(a["pred1"], 2)})
sub = pd.DataFrame(sub_rows); sub.to_csv(R / "05_high_awareness_subgroup.csv", index=False)
print("\nSUBGROUP\n", sub.to_string(index=False))

# ---------------------------------------------------------------- Step 9e: female x attention interaction
bi = lambda df: make_X(df, ["female"] + RES + AWR, PF, [("female", "attention_z")])
ri = fit("negbin", d.y, bi(d), cluster=d.psu); ti = table(ri); ti.to_csv(R / "05_negbin_female_x_attention.csv")
print("\nINTERACTION\n", ti.loc[["female", "attention_z", "female:attention_z"]])
grid = np.linspace(-1.5, 1.5, 13); rng = np.random.default_rng(3)
draws = rng.multivariate_normal(ri["coef"], ri["vcov"], 2000)[:, :-1]
pp = []
for f_ in (0, 1):
    for gv in grid:
        dd = d.assign(female=f_, attention_z=gv); Xg = np.asarray(bi(dd), float)
        sims = np.exp(Xg @ draws.T).mean(0)
        pp.append({"female": f_, "attention_z": gv, "pred": np.exp(Xg @ ri["coef"][:-1]).mean(),
                   "lo": np.quantile(sims, .025), "hi": np.quantile(sims, .975)})
pp = pd.DataFrame(pp); pp.to_csv(R / "05_pred_by_attention.csv", index=False)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for f_, col, lab in [(0, "#1f6f8b", "Men"), (1, "#d1495b", "Women")]:
    s_ = pp[pp.female == f_]
    ax.plot(s_.attention_z, s_.pred, color=col, label=lab, lw=2)
    ax.fill_between(s_.attention_z, s_.lo, s_.hi, color=col, alpha=.15)
ax.set_xlabel("Political attention index (SD units)"); ax.set_ylabel("Predicted \"don't know\" answers (of 59)")
ax.set_title("Figure 2. Predicted \"don't know\" answers by political attention and sex", fontsize=10)
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(R / "05_fig2_pred_by_attention.png", dpi=200)

summary = {"sample": S, "diagnostics": {k: v for k, v in diag.items() if k != "VIF"},
           "interaction_female_x_attention": ti.loc["female:attention_z"].to_dict()}
(R / "05_summary.json").write_text(json.dumps(summary, indent=1, default=float))
print("\nSaved Phase B Step 6/7/9 outputs.")
