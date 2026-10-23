"""
Silence, not ignorance? Gender, "don't know" responses and political awareness in Pakistan
Phase A, Steps 1-4: data audit, recoding, item sets, awareness measures, missingness.

Input : WVS_Pak_Data.xlsx, sheet 'original' (WVS Wave 7, Pakistan 2018, N = 1,995)
Output: data/analysis_wide.csv, data/analysis_long.csv, data/item_sets.csv, data/town.csv,
        results/01_audit_log.txt, results/01_reliability_factor.txt,
        results/01_missingness_by_sex.csv

Run from the project root:  python python/01_prepare_data.py  [path/to/WVS_Pak_Data.xlsx]
Requires only pandas, numpy, scikit-learn, openpyxl.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

ROOT = Path(__file__).resolve().parents[1]
XLSX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/project/WVS_Pak_Data.xlsx")
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)
log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); log_lines.append(s)

# ---------------------------------------------------------------- Step 1: import and audit
raw = pd.read_excel(XLSX, sheet_name="original")
labels = {c.split(":")[0]: c.split(":", 1)[1].strip() if ":" in c else c for c in raw.columns}
df = raw.copy()
df.columns = [c.split(":")[0] for c in raw.columns]

log("== STEP 1: AUDIT ==")
log("Rows, columns:", df.shape)
log("Duplicate interview IDs:", int(df["D_INTERVIEW"].duplicated().sum()))
log("Sex (Q260) counts:", df["Q260"].value_counts().sort_index().to_dict(), "(1 = male, 2 = female)")
const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
log("Constant columns (dropped from analysis):", const)
log("Note: W_WEIGHT, S018 and PWGHT are constant -> no design weights; analysis is unweighted.")
log("Note: S_INTLANGUAGE is Urdu for all respondents -> interview language cannot be tested.")

# ---------------------------------------------------------------- Step 2: item sets and recoding
def q(a, b): return [f"Q{i}" for i in range(a, b + 1)]
blocks = {
    "confidence_institutions": q(65, 81),
    "corruption": q(112, 120),
    "surveillance": q(196, 198),
    "electoral_integrity": q(224, 233) + ["Q234", "Q234A"],
    "regime_preferences": q(235, 239),
    "democracy": q(241, 253),
}
item_sets = {
    "domestic_political": sum(blocks.values(), []),
    # Q4 (importance of politics) is an awareness item, so it is excluded from the placebo set
    "non_political": [c for c in q(1, 27) + q(46, 63) + q(158, 175) if c != "Q4"],
    # Q82 = combined regional organisation; Q82_SAARC duplicates it and is not used
    "international": q(82, 89),
    "knowledge": q(91, 93),
}
for k, v in item_sets.items():
    missing_cols = [c for c in v if c not in df.columns]
    assert not missing_cols, f"{k}: columns not found {missing_cols}"
    log(f"Item set {k}: {len(v)} items")

rows = []
for s, items in item_sets.items():
    for it in items:
        blk = next((b for b, v in blocks.items() if it in v), s)
        rows.append({"item": it, "item_set": s, "item_block": blk, "label": labels[it]})
item_table = pd.DataFrame(rows)
item_table.to_csv(ROOT / "data" / "item_sets.csv", index=False)

# Coding: -1 = don't know -> dk = 1; -2 = no answer (kept as separate flag);
# -3, -4, -5 = not asked / not applicable -> missing; valid answers -> dk = 0.
all_items = item_table["item"].tolist()
codes = df[all_items]
dk = (codes == -1).astype(float)
na_flag = (codes == -2).astype(float)
eligible = codes.isin([-1, -2]) | (codes >= 0)          # asked of the respondent
dk = dk.where(eligible)
na_flag = na_flag.where(eligible)
out_of_scope = int((~eligible).sum().sum())
log(f"Cells coded -3/-4/-5 (set to missing): {out_of_scope}")

wide = pd.DataFrame({"id": df["D_INTERVIEW"], "psu": df["I_PSU"]})
for s, items in item_sets.items():
    wide[f"dk_count_{s}"] = dk[items].sum(axis=1, min_count=1)
    wide[f"n_items_{s}"] = dk[items].notna().sum(axis=1)
    wide[f"dk_share_{s}"] = wide[f"dk_count_{s}"] / wide[f"n_items_{s}"]
    wide[f"noanswer_count_{s}"] = na_flag[items].sum(axis=1, min_count=1)
for b, items in blocks.items():
    wide[f"dk_share_block_{b}"] = dk[items].mean(axis=1)

# ---------------------------------------------------------------- Step 3: predictors and awareness measures
def clean(s): return s.where(s >= 0)
wide["female"] = (df["Q260"] == 2).astype(int)
wide["age"] = clean(df["Q262"])
wide["education_isced"] = clean(df["Q275"])
wide["literate"] = (df["E1_LITERACY"] == 1).astype(int)
wide["income_scale"] = clean(df["Q288"])
emp = clean(df["Q279"])
wide["employed"] = emp.isin([1, 2, 3]).astype(float).where(emp.notna())
wide["homemaker"] = (emp == 5).astype(float).where(emp.notna())
wide["chief_wage_earner"] = (clean(df["Q285"]) == 1).astype(float).where(clean(df["Q285"]).notna())
wide["urban"] = (df["H_URBRURAL"] == 1).astype(int)
wide["province"] = df["N_REGION_WVS"].map({586010: "Punjab", 586011: "Sindh",
                                           586012: "KP", 586013: "Balochistan"})
assert wide["province"].notna().all(), "unexpected province code"
wide["others_present"] = (df["F_INTPRIVACY"] == 2).astype(int)
wide["resp_interest_low"] = (df["E_RESPINT"] >= 2).astype(int)   # 2 somewhat, 3 not interested
wide["duration_min"] = df["K_DURATION"]

# Awareness items reversed so that higher = more engaged
wide["politics_important"] = 5 - clean(df["Q4"])     # 1 very ... 4 not at all  -> 4 ... 1
wide["political_interest"] = 5 - clean(df["Q199"])   # 1 very ... 4 not at all  -> 4 ... 1
wide["political_discussion"] = 4 - clean(df["Q200"]) # 1 frequently ... 3 never -> 3 ... 1
news = {"Q201": "news_newspaper", "Q202": "news_tv", "Q203": "news_radio", "Q204": "news_mobile",
        "Q205": "news_email", "Q206": "news_internet", "Q207": "news_social", "Q208": "news_talk"}
for k, v in news.items():
    wide[v] = 6 - clean(df[k])                       # 1 daily ... 5 never -> 5 ... 1

def cronbach_alpha(frame):
    f = frame.dropna(); k = f.shape[1]
    return k / (k - 1) * (1 - f.var(ddof=1).sum() / f.sum(axis=1).var(ddof=1)), len(f)

rel = []
att_items = ["politics_important", "political_interest", "political_discussion"]
a, n = cronbach_alpha(wide[att_items])
rel.append(f"Political attention index (Q4, Q199, Q200): Cronbach alpha = {a:.3f} (n = {n})")
rel.append("Inter-item correlations:\n" + wide[att_items].corr().round(3).to_string())
z = (wide[att_items] - wide[att_items].mean()) / wide[att_items].std()
wide["attention_index"] = z.mean(axis=1).where(z.notna().sum(axis=1) >= 2)

# Exploratory factor analysis of news sources (2 factors, varimax), standardized complete cases
nv = list(news.values())
nf = wide[nv].dropna()
fa = FactorAnalysis(n_components=2, rotation="varimax", random_state=1).fit((nf - nf.mean()) / nf.std())
load = pd.DataFrame(fa.components_.T, index=nv, columns=["factor1", "factor2"]).round(3)
rel.append(f"\nNews sources EFA (2 factors, varimax, n = {len(nf)}):\n" + load.to_string())
f1 = load["factor1"].abs() >= load["factor2"].abs()
groups = {"factor1": [i for i in nv if f1[i]], "factor2": [i for i in nv if not f1[i]]}
rel.append(f"Items assigned by highest loading: {groups}")
traditional = ["news_newspaper", "news_tv", "news_radio"]
digital = ["news_mobile", "news_email", "news_internet", "news_social"]
for name, items in [("traditional media", traditional), ("digital media", digital)]:
    a, n = cronbach_alpha(wide[items])
    rel.append(f"{name} index {items}: Cronbach alpha = {a:.3f} (n = {n})")
rel.append("Decision: the traditional-media index is unreliable (alpha < 0.5), so it is not used. "
           "News exposure is measured by the digital-media index plus TV news as a single item.")
wide["news_digital"] = wide[digital].mean(axis=1)
wide["news_tv_weekly"] = (wide["news_tv"] >= 4).astype(float).where(wide["news_tv"].notna())

# Awareness tiers: tertiles of the attention index; plus the pre-specified "high awareness" group
wide["attention_tertile"] = pd.qcut(wide["attention_index"], 3, labels=["low", "medium", "high"])
wide["high_awareness"] = (
    (wide["political_interest"] >= 3) & (wide["political_discussion"] >= 2) & (wide["news_tv"] >= 4)
).astype(float).where(wide[["political_interest", "political_discussion", "news_tv"]].notna().all(axis=1))
rel.append("\nHigh-awareness group (interested, discusses at least occasionally, TV news at least weekly):")
rel.append(pd.crosstab(wide["high_awareness"], wide["female"], margins=True).to_string())
(ROOT / "results" / "01_reliability_factor.txt").write_text("\n".join(rel))
print("\n".join(rel))

# ---------------------------------------------------------------- Step 4: missingness of predictors by sex
preds = ["age", "education_isced", "literate", "income_scale", "employed", "chief_wage_earner",
         "urban", "others_present", "politics_important", "political_interest",
         "political_discussion", "attention_index", "news_digital", "news_tv"]
miss = pd.DataFrame({
    "missing_men_%": wide.loc[wide.female == 0, preds].isna().mean() * 100,
    "missing_women_%": wide.loc[wide.female == 1, preds].isna().mean() * 100,
}).round(2)
miss.to_csv(ROOT / "results" / "01_missingness_by_sex.csv")
cc_vars = ["age", "education_isced", "literate", "income_scale", "employed", "urban",
           "attention_index", "news_digital", "news_tv"]
log("\n== STEP 4: MISSINGNESS ==\n" + miss.to_string())
log("Complete cases on core model variables:", int(wide[cc_vars].notna().all(axis=1).sum()))

# ---------------------------------------------------------------- Save files
wide.to_csv(ROOT / "data" / "analysis_wide.csv", index=False)
long = dk.copy()
long.insert(0, "id", df["D_INTERVIEW"])
long = long.melt(id_vars="id", var_name="item", value_name="dk").dropna(subset=["dk"])
nal = na_flag.copy(); nal.insert(0, "id", df["D_INTERVIEW"])
nal = nal.melt(id_vars="id", var_name="item", value_name="no_answer")
long = long.merge(nal, on=["id", "item"], how="left").merge(item_table[["item", "item_set", "item_block"]], on="item")
long = long.merge(wide[["id", "psu", "female"]], on="id")
long["political_item"] = (long["item_set"] == "domestic_political").astype(int)
long.to_csv(ROOT / "data" / "analysis_long.csv", index=False)
# Town identifiers, used for the town fixed-effects robustness check (07 and R checkpoint 2)
pd.DataFrame({"id": df["D_INTERVIEW"], "town": df["N_TOWN"]}).to_csv(ROOT / "data" / "town.csv", index=False)
log(f"\nSaved analysis_wide.csv ({wide.shape}) and analysis_long.csv ({long.shape})")
(ROOT / "results" / "01_audit_log.txt").write_text("\n".join(log_lines))
