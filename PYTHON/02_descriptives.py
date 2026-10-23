"""
Phase A, Step 5: descriptive analysis and bivariate tests.

Input : data/analysis_wide.csv, data/analysis_long.csv, data/item_sets.csv (from 01_prepare_data.py)
Output: results/02_table1_dk_by_set_and_sex.csv
        results/02_dk_by_block_and_sex.csv
        results/02_item_level_tests.csv
        results/02_respondent_level_tests.csv
        results/02_fig1_item_gaps_domestic.png
        results/02_fig_sets_by_sex.png

Tests: chi-square (2x2) per item with Holm correction and Cohen's h;
       Welch t-test and Mann-Whitney U on respondent-level don't-know counts,
       with Cohen's d and rank-biserial correlation.
Requires pandas, numpy, scipy, matplotlib.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
wide = pd.read_csv(ROOT / "data" / "analysis_wide.csv")
long = pd.read_csv(ROOT / "data" / "analysis_long.csv")
items = pd.read_csv(ROOT / "data" / "item_sets.csv")
SETS = ["domestic_political", "non_political", "international", "knowledge"]
men, women = wide.female == 0, wide.female == 1

def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1, running)
    return adj

def cohens_h(p1, p2): return 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))

# ---------------------------------------------------------------- Table 1: by item set and sex
rows = []
for s in SETS:
    sh, ct = wide[f"dk_share_{s}"], wide[f"dk_count_{s}"]
    m, w = sh[men].mean() * 100, sh[women].mean() * 100
    rows.append({"item_set": s, "n_items": int(wide[f"n_items_{s}"].max()),
                 "men_dk_%": round(m, 2), "women_dk_%": round(w, 2),
                 "gap_pp": round(w - m, 2), "ratio_w_to_m": round(w / m, 2) if m > 0 else np.nan,
                 "men_mean_count": round(ct[men].mean(), 2), "women_mean_count": round(ct[women].mean(), 2),
                 "men_noanswer_mean": round(wide.loc[men, f"noanswer_count_{s}"].mean(), 3),
                 "women_noanswer_mean": round(wide.loc[women, f"noanswer_count_{s}"].mean(), 3)})
t1 = pd.DataFrame(rows)
t1.to_csv(R / "02_table1_dk_by_set_and_sex.csv", index=False)
print("TABLE 1: Don't-know rates by item set and sex\n", t1.to_string(index=False))

# ---------------------------------------------------------------- By domestic block (preview of H4)
brow = []
for b in items.loc[items.item_set == "domestic_political", "item_block"].unique():
    col = f"dk_share_block_{b}"
    m, w = wide.loc[men, col].mean() * 100, wide.loc[women, col].mean() * 100
    brow.append({"block": b, "n_items": int((items.item_block == b).sum()),
                 "men_dk_%": round(m, 2), "women_dk_%": round(w, 2),
                 "gap_pp": round(w - m, 2), "ratio": round(w / m, 2)})
tb = pd.DataFrame(brow).sort_values("gap_pp", ascending=False)
tb.to_csv(R / "02_dk_by_block_and_sex.csv", index=False)
print("\nDon't-know by domestic block\n", tb.to_string(index=False))

# ---------------------------------------------------------------- Item-level chi-square tests
irows = []
for it, g in long.groupby("item"):
    tab = pd.crosstab(g.female, g.dk).reindex(index=[0, 1], columns=[0.0, 1.0], fill_value=0)
    pm, pw = tab.loc[0, 1.0] / tab.loc[0].sum(), tab.loc[1, 1.0] / tab.loc[1].sum()
    chi2, p, _, _ = stats.chi2_contingency(tab, correction=False) if tab.values.min() >= 0 and tab[1.0].sum() > 0 else (np.nan, 1.0, 0, 0)
    irows.append({"item": it, "men_dk_%": pm * 100, "women_dk_%": pw * 100,
                  "gap_pp": (pw - pm) * 100, "cohens_h": cohens_h(pm, pw), "chi2": chi2, "p": p})
it_df = pd.DataFrame(irows).merge(items, on="item")
it_df["p_holm_within_set"] = it_df.groupby("item_set")["p"].transform(holm)
it_df = it_df.round({"men_dk_%": 2, "women_dk_%": 2, "gap_pp": 2, "cohens_h": 3, "chi2": 2})
it_df.sort_values(["item_set", "gap_pp"], ascending=[True, False]).to_csv(R / "02_item_level_tests.csv", index=False)
summ = it_df.groupby("item_set").apply(lambda d: pd.Series({
    "items": len(d), "women_higher": int((d.gap_pp > 0).sum()),
    "significant_holm_05": int((d.p_holm_within_set < .05).sum()),
    "median_gap_pp": round(d.gap_pp.median(), 2), "median_cohens_h": round(d.cohens_h.median(), 3)}), include_groups=False)
print("\nItem-level tests (Holm-corrected within each set)\n", summ.to_string())

# ---------------------------------------------------------------- Respondent-level tests
trows = []
for s in SETS:
    x, y = wide.loc[women, f"dk_count_{s}"], wide.loc[men, f"dk_count_{s}"]
    t, pt = stats.ttest_ind(x, y, equal_var=False)
    U, pu = stats.mannwhitneyu(x, y, alternative="two-sided")
    d = (x.mean() - y.mean()) / np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    rb = 2 * U / (len(x) * len(y)) - 1          # positive = women higher
    trows.append({"item_set": s, "welch_t": round(t, 2), "p_welch": pt, "cohens_d": round(d, 3),
                  "mann_whitney_U": U, "p_mw": pu, "rank_biserial": round(rb, 3)})
# Within-person placebo contrast: domestic share minus non-political share
diff = wide["dk_share_domestic_political"] - wide["dk_share_non_political"]
t, pt = stats.ttest_ind(diff[women], diff[men], equal_var=False)
trows.append({"item_set": "difference: domestic share - non-political share", "welch_t": round(t, 2), "p_welch": pt,
              "cohens_d": round((diff[women].mean() - diff[men].mean()) / np.sqrt((diff[women].var() + diff[men].var()) / 2), 3)})
tt = pd.DataFrame(trows)
tt["p_holm"] = holm(tt["p_welch"])
tt.to_csv(R / "02_respondent_level_tests.csv", index=False)
print("\nRespondent-level tests\n", tt.to_string(index=False))
print(f"\nPlacebo difference score (pp): men {diff[men].mean()*100:.2f}, women {diff[women].mean()*100:.2f}")

# ---------------------------------------------------------------- Figures
dom = it_df[it_df.item_set == "domestic_political"].sort_values("gap_pp")
fig, ax = plt.subplots(figsize=(8, 13))
ylab = [f"{r.item}  {r.label[:48]}" for r in dom.itertuples()]
yy = np.arange(len(dom))
ax.hlines(yy, dom["men_dk_%"], dom["women_dk_%"], color="#bbbbbb", lw=1.5)
ax.scatter(dom["men_dk_%"], yy, color="#1f6f8b", label="Men", zorder=3, s=22)
ax.scatter(dom["women_dk_%"], yy, color="#d1495b", label="Women", zorder=3, s=22)
ax.set_yticks(yy); ax.set_yticklabels(ylab, fontsize=7)
ax.set_xlabel("Respondents answering \"don't know\" (%)")
ax.set_title("Figure 1. \"Don't know\" by sex, 59 domestic political items\nWVS-7 Pakistan 2018 (N = 1,995)", fontsize=10)
ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(R / "02_fig1_item_gaps_domestic.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
xx = np.arange(len(t1)); w = .38
ax.bar(xx - w/2, t1["men_dk_%"], w, color="#1f6f8b", label="Men")
ax.bar(xx + w/2, t1["women_dk_%"], w, color="#d1495b", label="Women")
for i, r in t1.iterrows():
    ax.text(i - w/2, r["men_dk_%"] + .8, f'{r["men_dk_%"]:.1f}', ha="center", fontsize=8)
    ax.text(i + w/2, r["women_dk_%"] + .8, f'{r["women_dk_%"]:.1f}', ha="center", fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels(["Domestic\npolitical (59)", "Non-political\n(62)", "International\n(8)", "Knowledge\n(3)"])
ax.set_ylabel("Mean \"don't know\" share (%)"); ax.legend(); ax.grid(axis="y", alpha=.3)
ax.set_title("\"Don't know\" share by item set and sex", fontsize=10)
fig.tight_layout(); fig.savefig(R / "02_fig_sets_by_sex.png", dpi=200); plt.close(fig)
print("\nFigures saved.")
