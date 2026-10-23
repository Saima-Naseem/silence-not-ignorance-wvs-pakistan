"""
Phase B: Step 13 (robustness checks) and Step 14 (machine-learning check).

Each robustness variant re-estimates
  H1: negative binomial, domestic "don't know" count, female + resources (M1)   -> IRR for female
  H2: fractional logit, two rows per respondent (domestic vs non-political share),
      female x political + resources, cluster-robust SE                          -> OR for female x political
ML: histogram gradient boosting (Poisson loss), grouped 5-fold CV by sampling unit,
    permutation importance on held-out folds, partial dependence (SHAP not installed here).
Outputs in results/07_*.csv, results/07_*.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import mean_poisson_deviance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stats_utils import make_X, fit, table, holm, PROVINCES

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
XLSX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/project/WVS_Pak_Data.xlsx")
wide = pd.read_csv(ROOT / "data" / "analysis_wide.csv")
long = pd.read_csv(ROOT / "data" / "analysis_long.csv")
raw = pd.read_excel(XLSX, sheet_name="original", usecols=lambda c: c.split(":")[0] in ("D_INTERVIEW", "E_RESPINT", "N_TOWN"))
raw.columns = [c.split(":")[0] for c in raw.columns]
raw = raw.rename(columns={"D_INTERVIEW": "id", "N_TOWN": "town"})
wide = wide.merge(raw, on="id")
z = lambda s: (s - s.mean()) / s.std()
RES = ["age_z", "educ_z", "literate", "income_z", "employed", "urban"]
PF = {"province": PROVINCES}

def base_frame(w):
    return pd.DataFrame({"id": w.id.values, "psu": w.psu.values, "female": w.female.values,
        "age_z": z(w.age).values, "educ_z": z(w.education_isced).values, "literate": w.literate.values,
        "income_z": z(w.income_scale).values, "employed": w.employed.values, "urban": w.urban.values,
        "province": w.province.values, "town": w.town.values})

def within(d, y, fe):
    """Keep units of `fe` that contain both sexes and at least one positive outcome."""
    y = pd.Series(np.asarray(y, float))
    keep = (d.groupby(fe).female.transform(lambda s: s.nunique() > 1).values
            & y.groupby(d[fe].values).transform("sum").gt(0).values)
    return d[keep].reset_index(drop=True), y[keep].reset_index(drop=True), keep

def run_h1(d, y, weights=None, fe=None):
    if fe:
        d, y, _ = within(d, y, fe)
        levels = sorted(d[fe].unique()); X = make_X(d, ["female"] + RES, {fe: levels})
    else:
        X = make_X(d, ["female"] + RES, PF)
    r = fit("negbin", y, X, w=weights, cluster=d.psu); t = table(r)
    return t.loc["female", "exp_b"], t.loc["female", "exp_ci_low"], t.loc["female", "exp_ci_high"], t.loc["female", "p"], len(d)

def run_h2(d, pol_share, non_share, n_pol, n_non, weights=None):
    h = pd.concat([d.assign(political=1, share=np.asarray(pol_share), n=np.asarray(n_pol, float)),
                   d.assign(political=0, share=np.asarray(non_share), n=np.asarray(n_non, float))], ignore_index=True)
    wv = h.n.values * (np.r_[weights, weights] if weights is not None else 1)
    r = fit("logit", h.share, make_X(h, ["female", "political"] + RES, PF, [("female", "political")]), w=wv, cluster=h.psu)
    t = table(r).loc["female:political"]
    return t.exp_b, t.exp_ci_low, t.exp_ci_high, t.p

rows = []
def record(name, h1, h2):
    rows.append({"variant": name, "n": h1[4], "H1_IRR_female": round(h1[0], 3), "H1_CI": f"{h1[1]:.2f}-{h1[2]:.2f}",
                 "H1_p": h1[3], "H2_OR_female_x_political": None if h2 is None else round(h2[0], 3),
                 "H2_CI": None if h2 is None else f"{h2[1]:.2f}-{h2[2]:.2f}", "H2_p": None if h2 is None else h2[3]})
    print(rows[-1])

cc = base_frame(wide).notna().all(axis=1).values
W = wide[cc].reset_index(drop=True); d = base_frame(W)
nd, nn = W.n_items_domestic_political, W.n_items_non_political

# 0 main specification
record("main specification", run_h1(d, W.dk_count_domestic_political),
       run_h2(d, W.dk_share_domestic_political, W.dk_share_non_political, nd, nn))
# 1 binary outcome: any "don't know" on domestic items (logit), H1 only
rb = fit("logit", (W.dk_count_domestic_political > 0).astype(int), make_X(d, ["female"] + RES, PF), cluster=d.psu)
tb = table(rb).loc["female"]
rows.append({"variant": "binary outcome: any DK (logit; value is an odds ratio)", "n": len(d), "H1_IRR_female": round(tb.exp_b, 3),
             "H1_CI": f"{tb.exp_ci_low:.2f}-{tb.exp_ci_high:.2f}", "H1_p": tb.p, "H2_OR_female_x_political": None, "H2_CI": None, "H2_p": None})
# 2 "no answer" counted as "don't know"
y2 = W.dk_count_domestic_political + W.noanswer_count_domestic_political
record("no answer counted as DK", run_h1(d, y2.rename("y2")),
       run_h2(d, y2 / nd, (W.dk_count_non_political + W.noanswer_count_non_political) / nn, nd, nn))
# 3 exclude low-quality interviews (respondent not interested, or duration in bottom 5%)
lowq = (W.E_RESPINT == 3) | (W.duration_min <= W.duration_min.quantile(.05))
Wq, dq = W[~lowq].reset_index(drop=True), d[~lowq.values].reset_index(drop=True)
record(f"exclude low-quality interviews (n dropped = {int(lowq.sum())})", run_h1(dq, Wq.dk_count_domestic_political),
       run_h2(dq, Wq.dk_share_domestic_political, Wq.dk_share_non_political, Wq.n_items_domestic_political, Wq.n_items_non_political))
# 4 core political institutions only (drop non-state confidence items) and 5 drop regime-preference items
items = pd.read_csv(ROOT / "data" / "item_sets.csv")
non_state = ["Q66", "Q67", "Q68", "Q75", "Q77", "Q78", "Q79", "Q80", "Q81"]
for name, drop in [("core political items only (9 non-state confidence items dropped)", non_state),
                   ("regime-preference items dropped", items.item[items.item_block == "regime_preferences"].tolist())]:
    keep_items = items.item[(items.item_set == "domestic_political") & ~items.item.isin(drop)]
    sub = long[long.item.isin(keep_items)].groupby("id").dk.agg(["sum", "count"])
    s = W[["id"]].merge(sub, left_on="id", right_index=True, how="left")
    record(name + f" [{len(keep_items)} items]", run_h1(d, s["sum"].rename("y")),
           run_h2(d, s["sum"] / s["count"], W.dk_share_non_political, s["count"], nn))
# 6 sampling-unit fixed effects (compare women and men within the same sampling unit), H1 only
record("sampling-unit fixed effects (only the 7 mixed-sex PSUs contribute)", run_h1(d, W.dk_count_domestic_political, fe="psu"), None)
# 6b town fixed effects: compare women and men living in the same town (25 towns contain both sexes)
dt, _, kt = within(d, W.dk_count_domestic_political, "town"); Wt = W[kt].reset_index(drop=True)
ht = pd.concat([dt.assign(political=1, share=Wt.dk_share_domestic_political.values, n=Wt.n_items_domestic_political.values.astype(float)),
                dt.assign(political=0, share=Wt.dk_share_non_political.values, n=Wt.n_items_non_political.values.astype(float))], ignore_index=True)
tl = sorted(dt.town.unique())
rt = fit("logit", ht.share, make_X(ht, ["female", "political"] + RES, {"town": tl}, [("female", "political")]), w=ht.n, cluster=ht.psu)
tt_ = table(rt).loc["female:political"]
record(f"town fixed effects (within-town comparison, {len(tl)} towns)", run_h1(d, W.dk_count_domestic_political, fe="town"),
       (tt_.exp_b, tt_.exp_ci_low, tt_.exp_ci_high, tt_.p))
# 7 age post-stratification weights (census: 18-29 42%, 30-49 37%, 50+ 21%; cleaning_log step 20)
ag = pd.cut(W.age, [17, 29, 49, 200], labels=["18-29", "30-49", "50+"])
census = {"18-29": .42, "30-49": .37, "50+": .21}; samp = ag.value_counts(normalize=True)
wts = ag.map(lambda a: census[a] / samp[a]).astype(float).values
record("age post-stratification weights", run_h1(d, W.dk_count_domestic_political, weights=wts),
       run_h2(d, W.dk_share_domestic_political, W.dk_share_non_political, nd, nn, weights=wts))
# 8 multiple imputation of missing covariates (20 imputations, Rubin's rules), H1 only
Wall = wide.reset_index(drop=True); M = 20; bs, vs = [], []
imp_cols = ["age", "education_isced", "literate", "income_scale", "employed", "urban", "female"]
for m in range(M):
    imp = IterativeImputer(sample_posterior=True, random_state=m, max_iter=15)
    A = pd.DataFrame(imp.fit_transform(Wall[imp_cols]), columns=imp_cols)
    A["employed"] = (A.employed > .5).astype(float)
    dm = base_frame(Wall.assign(**{c: A[c] for c in imp_cols if c != "female"}))
    r = fit("negbin", Wall.dk_count_domestic_political, make_X(dm, ["female"] + RES, PF), cluster=dm.psu)
    bs.append(r["coef"][1]); vs.append(r["vcov"][1, 1])
qbar, ubar, bvar = np.mean(bs), np.mean(vs), np.var(bs, ddof=1); T = ubar + (1 + 1 / M) * bvar
se = np.sqrt(T); p = 2 * stats.norm.sf(abs(qbar / se))
rows.append({"variant": f"multiple imputation (M = {M}, full N)", "n": len(Wall), "H1_IRR_female": round(np.exp(qbar), 3),
             "H1_CI": f"{np.exp(qbar-1.96*se):.2f}-{np.exp(qbar+1.96*se):.2f}", "H1_p": p,
             "H2_OR_female_x_political": None, "H2_CI": None, "H2_p": None})
rob = pd.DataFrame(rows); rob.to_csv(R / "07_robustness.csv", index=False)
print("\nROBUSTNESS\n", rob.to_string(index=False))

# ---------------------------------------------------------------- Holm correction across the main hypothesis tests
p05 = pd.read_csv(R / "05_stepwise_female.csv").set_index("model")
p06 = pd.read_csv(R / "06_negbin_female_x_privacy.csv", index_col=0)
main = pd.DataFrame({"hypothesis": ["H1 female effect (NB, M1)", "H2 female x political (main spec, fractional logit)",
                                    "H3 female effect net of awareness (NB, M2)", "H4 female x block (Wald, R checkpoint 1)",
                                    "H5 female x others present (NB)"],
                     "p": [p05.loc["M1", "p"], rows[0]["H2_p"], p05.loc["M2", "p"], 6.031e-10, p06.loc["female:others_present", "p"]]})
main["p_holm"] = holm(main.p); main.to_csv(R / "07_holm_main_tests.csv", index=False)
print("\nHOLM\n", main.to_string(index=False))

# ---------------------------------------------------------------- Step 14: machine-learning check
feat = pd.DataFrame({"female": wide.female, "age": wide.age, "education": wide.education_isced, "literate": wide.literate,
    "income": wide.income_scale, "employed": wide.employed, "urban": wide.urban,
    "Sindh": (wide.province == "Sindh").astype(int), "KP": (wide.province == "KP").astype(int),
    "Balochistan": (wide.province == "Balochistan").astype(int), "attention": wide.attention_index,
    "digital_news": wide.news_digital, "tv_news": wide.news_tv, "others_present": wide.others_present,
    "resp_interest_low": wide.resp_interest_low, "duration": wide.duration_min})
yml = wide.dk_count_domestic_political.values; groups = wide.psu.values
gkf = GroupKFold(n_splits=5); dev_m, dev_b, imps = [], [], []
for k, (tr, te) in enumerate(gkf.split(feat, yml, groups)):
    mdl = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                        min_samples_leaf=30, l2_regularization=1.0, random_state=k)
    mdl.fit(feat.iloc[tr], yml[tr])
    dev_m.append(mean_poisson_deviance(yml[te], mdl.predict(feat.iloc[te])))
    dev_b.append(mean_poisson_deviance(yml[te], np.full(len(te), yml[tr].mean())))
    pi = permutation_importance(mdl, feat.iloc[te], yml[te], n_repeats=20, random_state=k,
                                scoring=lambda est, X_, y_: -mean_poisson_deviance(y_, est.predict(X_)))
    imps.append(pi.importances_mean)
imp = pd.DataFrame({"feature": feat.columns, "importance_mean": np.mean(imps, 0), "importance_sd_across_folds": np.std(imps, 0)})
imp["rank"] = imp.importance_mean.rank(ascending=False).astype(int)
imp = imp.sort_values("rank"); imp.to_csv(R / "07_ml_permutation_importance.csv", index=False)
cv = pd.DataFrame({"fold": range(1, 6), "deviance_model": dev_m, "deviance_baseline": dev_b})
cv["deviance_explained_%"] = 100 * (1 - cv.deviance_model / cv.deviance_baseline)
cv.to_csv(R / "07_ml_cv.csv", index=False)
print("\nML CV\n", cv.round(3).to_string(index=False)); print(imp.round(4).to_string(index=False))

full_m = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=30, l2_regularization=1.0, random_state=0).fit(feat, yml)
grid = np.quantile(feat.attention.dropna(), np.linspace(.05, .95, 10))
pdp = []
for f_ in (0, 1):
    for g_ in grid:
        pdp.append({"female": f_, "attention": g_, "pred": full_m.predict(feat.assign(female=f_, attention=g_)).mean()})
pdp = pd.DataFrame(pdp); pdp.to_csv(R / "07_ml_partial_dependence.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
top = imp.head(10)[::-1]
ax[0].barh(top.feature, top.importance_mean, xerr=top.importance_sd_across_folds, color="#7a6c9e")
ax[0].set_title("Permutation importance (held-out folds)", fontsize=10); ax[0].set_xlabel("Increase in Poisson deviance")
for f_, col, lab in [(0, "#1f6f8b", "Men"), (1, "#d1495b", "Women")]:
    s_ = pdp[pdp.female == f_]; ax[1].plot(s_.attention, s_.pred, color=col, label=lab, lw=2)
ax[1].set_title("Partial dependence: attention x sex", fontsize=10); ax[1].set_xlabel("Political attention index")
ax[1].set_ylabel("Predicted \"don't know\" answers"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(R / "07_fig_ml.png", dpi=200)
print("\nSaved robustness and ML outputs.")
