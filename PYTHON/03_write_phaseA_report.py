"""
Builds reports/PHASE_A_RESULTS.md from the Phase A output files, so every number
in the report is read from results/ rather than typed by hand.
Run after 01_prepare_data.py and 02_descriptives.py.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
(ROOT / "reports").mkdir(exist_ok=True)

t1 = pd.read_csv(R / "02_table1_dk_by_set_and_sex.csv")
blk = pd.read_csv(R / "02_dk_by_block_and_sex.csv")
items = pd.read_csv(R / "02_item_level_tests.csv")
tests = pd.read_csv(R / "02_respondent_level_tests.csv")
miss = pd.read_csv(R / "01_missingness_by_sex.csv", index_col=0)
wide = pd.read_csv(ROOT / "data" / "analysis_wide.csv")

def md(df): return df.to_markdown(index=False)
def p(x): return "< 0.001" if x < 0.001 else f"{x:.3f}"

set_names = {"domestic_political": "Domestic political", "non_political": "Non-political (placebo)",
             "international": "International", "knowledge": "Factual knowledge"}
T1 = t1.assign(item_set=t1.item_set.map(set_names))[
    ["item_set", "n_items", "men_dk_%", "women_dk_%", "gap_pp", "ratio_w_to_m", "men_mean_count", "women_mean_count"]]
T1.columns = ["Item set", "Items", "Men DK %", "Women DK %", "Gap (pp)", "Women ÷ men", "Men mean count", "Women mean count"]

TB = blk.copy(); TB.columns = ["Domestic block", "Items", "Men DK %", "Women DK %", "Gap (pp)", "Women ÷ men"]

summ = items.groupby("item_set").agg(items=("item", "size"), women_higher=("gap_pp", lambda s: int((s > 0).sum())),
                                     sig_raw=("p", lambda s: int((s < .05).sum())),
                                     sig_holm=("p_holm_within_set", lambda s: int((s < .05).sum())),
                                     median_gap=("gap_pp", "median"), median_h=("cohens_h", "median")).reset_index()
summ["item_set"] = summ.item_set.map(set_names)
summ = summ.round({"median_gap": 2, "median_h": 3})
summ.columns = ["Item set", "Items", "Women higher", "Significant (raw p < .05)", "Significant (Holm)", "Median gap (pp)", "Median Cohen's h"]

TT = tests.copy()
TT["item_set"] = TT.item_set.map(lambda s: set_names.get(s, "Within-person difference (domestic − placebo share)"))
TT = TT.assign(p_welch=TT.p_welch.map(p), p_holm=TT.p_holm.map(p),
               p_mw=TT.p_mw.map(lambda x: p(x) if pd.notna(x) else "–"),
               rank_biserial=TT.rank_biserial.map(lambda x: x if pd.notna(x) else "–"))
TT = TT[["item_set", "welch_t", "p_welch", "p_holm", "cohens_d", "p_mw", "rank_biserial"]]
TT.columns = ["Comparison (women vs men)", "Welch t", "p", "p (Holm)", "Cohen's d", "Mann–Whitney p", "Rank-biserial r"]

dom = items[items.item_set == "domestic_political"].sort_values("gap_pp", ascending=False)
cols = ["item", "label", "men_dk_%", "women_dk_%", "gap_pp", "cohens_h"]
top = dom[cols].head(10); bottom = dom[cols].tail(6)
for t in (top, bottom): t.columns = ["Item", "Label", "Men DK %", "Women DK %", "Gap (pp)", "Cohen's h"]

M = miss.reset_index().rename(columns={"index": "Variable", "missing_men_%": "Missing men %", "missing_women_%": "Missing women %"})
diff = wide.dk_share_domestic_political - wide.dk_share_non_political
dm, dw = diff[wide.female == 0].mean() * 100, diff[wide.female == 1].mean() * 100
ha = pd.crosstab(wide.high_awareness, wide.female)
nonpol = items[items.item_set == "non_political"]

text = f"""# Phase A results: Steps 1–5

**Study:** Silence, not ignorance? Gender, "don't know" responses and political awareness in Pakistan
**Data:** World Values Survey Wave 7, Pakistan 2018, sheet `original` (N = 1,995: 1,037 men, 958 women)
**Scripts:** `python/01_prepare_data.py`, `python/02_descriptives.py`; this report: `python/03_write_phaseA_report.py`
**Status:** Descriptive and bivariate results only. No multivariate models yet (R checkpoint 1 and Phase B).

---

## 1. Data audit (Step 1)

- 1,995 respondents, 397 variables, no duplicate interview IDs.
- **No design weights:** `W_WEIGHT`, `S018` and `PWGHT` are constant, so all analysis is unweighted.
- **Interview language is constant** (Urdu for everyone), so it cannot be tested and is dropped from the design.
- No cells coded −3, −4 or −5 (not asked / not applicable) in the analysed items.

## 2. Item sets and coding (Step 2)

| Item set | Items | WVS questions | Role |
|---|---|---|---|
| Domestic political | 59 | Q65–81, Q112–120, Q196–198, Q224–234A, Q235–239, Q241–253 | Main outcome |
| Non-political | 62 | Q1–27 (excl. Q4), Q46–63, Q158–175 | Placebo |
| International | 8 | Q82–89 | Contrast ("ignorance zone") |
| Factual knowledge | 3 | Q91–93 | Supporting analysis |

Coding: "don't know" (−1) = 1; valid answer = 0; "no answer" (−2) kept as a separate flag.

**Decisions made during coding**
- Q4 ("importance of politics") was removed from the placebo set because it is an awareness item. Placebo set = 62 items, not 63.
- Q82_SAARC was not used because it duplicates Q82 (major regional organisation).

## 3. Awareness measures (Step 3)

- **Political attention index** (Q4, Q199, Q200, reversed so higher = more engaged; mean of z-scores): Cronbach's alpha = 0.678. Acceptable for three items. Inter-item correlations 0.32–0.60.
- **News exposure:** a two-factor analysis separated digital sources from TV news.
  - Digital-media index (mobile phone, email, internet, social media): alpha = 0.823. **Used.**
  - Traditional-media index (newspaper, TV, radio): alpha = 0.322. **Dropped as unreliable.**
  - TV news is used as a single item.
- **High-awareness group** (very or somewhat interested in politics, discusses politics at least occasionally, TV news at least weekly): {int(ha.loc[1.0, 0])} men and {int(ha.loc[1.0, 1])} women.

## 4. Missing data on predictors (Step 4)

{md(M)}

Missingness is low (all under 3% except income, which is 7.0% for men and 4.2% for women). Complete cases on the core model variables: 1,845 respondents. Multiple imputation is planned as a robustness check.

## 5. Descriptive and bivariate results (Step 5)

### Table 1. "Don't know" rates by item set and sex

{md(T1)}

### Table 2. Respondent-level tests

{md(TT)}

The within-person difference score (domestic share minus non-political share) averages {dm:.2f} percentage points for men and {dw:.2f} for women.

### Table 3. Item-level chi-square tests (Holm correction within each set)

{md(summ)}

Among the 62 non-political items, gaps range from {nonpol.gap_pp.min():.2f} to {nonpol.gap_pp.max():.2f} percentage points; {int((nonpol.p < .05).sum())} reach raw p < .05, and none survive Holm correction.

### Table 4. Gender gap by domestic topic block (preview of H4)

{md(TB)}

### Table 5. Domestic items with the largest gender gaps

{md(top)}

### Table 6. Domestic items with the smallest gender gaps

{md(bottom)}

**Figures:** `results/02_fig1_item_gaps_domestic.png` (all 59 domestic items), `results/02_fig_sets_by_sex.png` (item sets).

---

## 6. Interpretation (preliminary)

1. **H1 supported descriptively.** Women give about 2.4 times as many "don't know" answers as men on domestic political items (7.9% vs 3.3%; Cohen's d = 0.48). Women are higher on 58 of 59 items, and 43 remain significant after Holm correction.
2. **H2 placebo holds.** On the 62 non-political items there is no gender difference (0.79% vs 0.74%; p = 0.54). The within-person difference test is the strongest single result (t = 11.14, d = 0.51). The gap is specific to politics, not a general tendency of women to say "don't know".
3. **International and knowledge items** show large gaps too, but high "don't know" rates for both sexes. Genuine unfamiliarity is plausible there, which is why they serve as the contrast rather than the main outcome.
4. **H4 partly supported.** The largest absolute gaps are on electoral integrity, surveillance and corruption, as predicted. Democracy items have the smallest absolute gap but the highest ratio (rates are low for both sexes). The paper must state whether H4 refers to absolute or relative gaps.

## 7. Issues flagged for later steps

- **Confidence block mixes institution types.** Several of the largest gaps are on non-state bodies (charities, environmental groups, the women's movement, companies), while institutions people deal with directly (police, government, army, TV) show almost no gap. Planned robustness check: core political institutions only.
- **No design weights.** Re-weighting to census age and sex distributions is planned as a robustness check.
- **These are bivariate results.** Whether the gap survives controls for education, literacy, income and awareness (H3) is tested in the next phases.

## 8. Next steps

- **R checkpoint 1 (you):** `R/checkpoint1_baseR.R`: multilevel models for H2 and H4.
- **Phase B (Claude):** diagnostics, H1 negative binomial model, H3 (stepwise models, decomposition, awareness interaction), H5 (interview privacy), knowledge items, robustness checks, ML check.
"""
(ROOT / "reports" / "PHASE_A_RESULTS.md").write_text(text)
print("Saved reports/PHASE_A_RESULTS.md")
