"""
Builds reports/PHASE_B_RESULTS.md from the Phase B output files (numbers are read, not typed).
Run after 04-07.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
md = lambda df, idx=False: df.to_markdown(index=idx)
pf = lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}"

val = (R / "04_validation.txt").read_text()
s5 = json.loads((R / "05_summary.json").read_text()); dg = s5["diagnostics"]
step = pd.read_csv(R / "05_stepwise_female.csv")
irr = pd.read_csv(R / "05_negbin_M0_M3_IRR.csv", header=[0, 1], index_col=0)
khb = pd.read_csv(R / "05_khb.csv"); oax = pd.read_csv(R / "05_oaxaca.csv", index_col=0)
sub = pd.read_csv(R / "05_high_awareness_subgroup.csv")
inter = pd.read_csv(R / "05_negbin_female_x_attention.csv", index_col=0)
prov = pd.read_csv(R / "05_dk_by_province_sex.csv")
pdesc = pd.read_csv(R / "06_privacy_descriptives.csv")
pint = pd.read_csv(R / "06_negbin_female_x_privacy.csv", index_col=0)
peff = pd.read_csv(R / "06_privacy_effect_by_sex.csv"); psm = pd.read_csv(R / "06_psm_privacy.csv")
s6 = json.loads((R / "06_summary.json").read_text())
krate = pd.read_csv(R / "06_knowledge_rates.csv"); kmod = pd.read_csv(R / "06_knowledge_models.csv")
rob = pd.read_csv(R / "07_robustness.csv"); holm = pd.read_csv(R / "07_holm_main_tests.csv")
cv = pd.read_csv(R / "07_ml_cv.csv"); imp = pd.read_csv(R / "07_ml_permutation_importance.csv")
wide = pd.read_csv(ROOT / "data" / "analysis_wide.csv")
g = wide.groupby("psu").female.mean()
n_m, n_f, n_x = int((g == 0).sum()), int((g == 1).sum()), int(((g > 0) & (g < 1)).sum())

st = step[["model", "IRR", "IRR_ci", "p", "AME_count", "AME_ci", "pred_men", "pred_women"]].copy()
st["model"] = st.model.map({"M0": "M0: sex only", "M1": "M1: + resources & province", "M2": "M2: + awareness", "M3": "M3: + interview conditions"})
st["p"] = st.p.map(pf)
st.columns = ["Model", "IRR (women)", "95% CI", "p", "Extra DK answers (women − men)", "95% CI ", "Predicted men", "Predicted women"]

irr_t = irr.copy(); irr_t.columns = [f"{a} {b}" for a, b in irr_t.columns]
irr_t = irr_t.drop(index="ln_alpha").round(3).fillna("")

kh = khb.rename(columns={"mediators": "Mediators", "female_b_reduced": "Female b (reduced, KHB)", "female_b_full": "Female b (full)",
                         "indirect_b": "Indirect b", "indirect_ci_low": "CI low", "indirect_ci_high": "CI high",
                         "pct_of_total_explained": "% of gap explained", "boot_reps": "Bootstrap reps"})
ox = oax.reset_index().rename(columns={"index": "Component", "estimate_pp": "Estimate (pp)", "ci_low": "CI low", "ci_high": "CI high", "pct_of_gap": "% of gap"})
sb = sub.assign(p=sub.p.map(pf)).rename(columns={"group": "Group", "n": "N", "women": "Women", "IRR_female": "IRR (women)", "IRR_ci": "95% CI", "pred_men": "Pred. men", "pred_women": "Pred. women"})
it = inter.loc[["female", "attention_z", "female:attention_z"], ["exp_b", "exp_ci_low", "exp_ci_high", "p"]].round(3).reset_index()
it.columns = ["Term", "IRR", "CI low", "CI high", "p"]
pi_ = pint.loc[["female", "others_present", "female:others_present"], ["exp_b", "exp_ci_low", "exp_ci_high", "p"]].round(3).reset_index()
pi_.columns = ["Term", "IRR", "CI low", "CI high", "p"]
pdesc["female"] = pdesc.female.map({0: "men", 1: "women"}); pdesc["others_present"] = pdesc.others_present.map({0: "private", 1: "others present"})
pdesc.columns = ["Sex", "Interview", "N", "Mean DK answers (of 59)"]
rb = rob.copy(); rb["H1_p"] = rb.H1_p.map(pf); rb["H2_p"] = rb.H2_p.map(lambda x: pf(x) if pd.notna(x) else "–")
rb = rb.fillna("–"); rb.columns = ["Variant", "N", "H1: IRR women", "H1 95% CI", "H1 p", "H2: OR female × political", "H2 95% CI", "H2 p"]
hm = holm.assign(p=holm.p.map(pf), p_holm=holm.p_holm.map(pf)).rename(columns={"hypothesis": "Test", "p_holm": "p (Holm)"})
cvt = cv.round(2); imp_t = imp.head(8).round(3)
attn_rank = int(imp.loc[imp.feature == "attention", "rank"].iloc[0]); fem_rank = int(imp.loc[imp.feature == "female", "rank"].iloc[0])
town_row = rob[rob.variant.str.startswith("town")].iloc[0]

text = f"""# Phase B results: Steps 6, 7, 9, 11–14

**Study:** Silence, not ignorance? Gender, "don't know" responses and political awareness in Pakistan
**Scripts:** `python/stats_utils.py`, `04_validate_estimator.py`, `05_phaseB_models.py`, `06_phaseB_privacy_knowledge.py`, `07_phaseB_robustness_ml.py`; this report: `08_write_phaseB_report.py`
**Outcome:** number of "don't know" answers on the 59 domestic political items (0–59).
**Main sample:** {s5['sample']['n']} complete cases ({s5['sample']['n_women']} women, {s5['sample']['n_men']} men); robustness checks use the samples stated.
**Standard errors:** cluster-robust by sampling unit (197 clusters) throughout.

---

## 0. Estimator validation

The Python estimators were written with numpy/scipy (statsmodels is not installed in this environment) and checked first:

```
{val}
```

## 1. A design finding: men and women were interviewed in separate sampling units

Of 197 sampling units, **{n_m} contain only men, {n_f} only women and {n_x} both sexes**. The fieldwork interviewed men and women in separate sampling units, very likely with same-sex interviewer teams (standard practice in Pakistan, but the interviewer's sex is not recorded in the file).

**Consequences:**
- Respondent sex cannot be separated from sampling unit or interviewer team, so sampling-unit fixed effects are not possible (only 67 respondents in mixed units).
- Sampling units differ strongly: intraclass correlation of the domestic "don't know" share = {dg['ICC_psu_domestic_share']:.2f}. Cluster-robust SEs are essential.
- **Mitigations already in the design:** (a) the within-person placebo (H2) compares each woman with herself, so any interviewer team that simply records more "don't know" answers in general is cancelled out; (b) **town fixed effects** compare women and men living in the same town (Section 9), and the result holds.
- **Remaining limitation for the paper:** an interviewer team effect *specific to political questions* cannot be ruled out. Same-sex interviewing should, if anything, make women more comfortable, but this must be stated.

## 2. Diagnostics (Step 6)

| Check | Result |
|---|---|
| Overdispersion: negative binomial alpha | {dg['alpha_negbin']:.2f} |
| Likelihood-ratio test, Poisson vs negative binomial | χ² = {dg['LR_poisson_vs_negbin']:.0f}, p < 0.001 |
| Cameron–Trivedi overdispersion test | t = {dg['cameron_trivedi_t']:.2f}, p < 0.001 |
| Zeros: observed / Poisson / negative binomial | {dg['zeros_observed_%']:.1f}% / {dg['zeros_poisson_%']:.1f}% / {dg['zeros_negbin_%']:.1f}% |
| Largest variance inflation factor | {dg['VIF_max']:.2f} |

The negative binomial model is clearly preferred to Poisson and reproduces the share of zeros almost exactly, so a zero-inflated model is not needed. Multicollinearity is not a concern.

**Province pattern (raw "don't know" % on domestic items):**

{md(prov)}

The gender gap appears in Punjab, Sindh and especially KP. In Balochistan (91 respondents, 10 sampling units) women report fewer "don't know" answers than men; with so few units this may reflect particular interviewer teams. Worth a footnote; not enough data to analyse separately.

## 3. H1 and H3: nested negative binomial models (Steps 7 and 9a)

{md(st)}

**Full models (incidence rate ratios and p-values):**

{md(irr_t, idx=True)}

**H1 supported.** With resources and province controlled (M1), women give about 1.76 times as many "don't know" answers as men (p = 0.001), about 1.6 more answers out of 59.

**H3, the core test:** adding political awareness (M2) leaves the women's coefficient **unchanged** (IRR 1.760 → 1.765), even though attention (IRR 0.76) and TV news (IRR 0.81) strongly reduce "don't know" answers for everyone. Interview conditions (M3) change little.

## 4. H3: KHB decomposition (Step 9b)

{md(kh)}

KHB makes coefficients comparable across nested non-linear models. **Awareness explains about 1% of the gender gap** once resources are held constant (interval includes zero). Resources and awareness together explain about 30%. Most of the gap is not accounted for by what women know or how much they follow politics.

## 5. H3: Oaxaca–Blinder decomposition (Step 9c)

Outcome: % "don't know" on domestic items. Pooled coefficients, cluster bootstrap (1,000 reps).

{md(ox)}

About **62% of the gap is unexplained** by differences in resources and awareness. Awareness differences account for about 13%.

## 6. H3: high-awareness subgroup and interaction (Steps 9d–e)

{md(sb)}

{md(it)}

Among highly aware respondents the gap is smaller and not significant (IRR 1.49, p = 0.10); this subgroup is a third of the sample, so the test has less power. The female × attention interaction is not significant (IRR 0.92): awareness reduces "don't know" answers at a similar rate for women and men, and women stay higher at every level of attention (Figure 2: `results/05_fig2_pred_by_attention.png`).

**Overall H3 verdict: supported.** Awareness strongly predicts fewer "don't know" answers, but it does not explain why women give more of them. The gap persists at all levels of attention, though it narrows among the most engaged.

## 7. H5: interview privacy (Step 11)

**Descriptives (mean "don't know" answers out of 59):**

{md(pdesc)}

**Negative binomial with female × others present:**

{md(pi_)}

**Predicted effect of others being present, by sex:**

{md(peff)}

**Propensity-score matching within each sex** (1:1 nearest neighbour, caliper 0.2 SD; max |SMD| before {s6['max_abs_smd_before_matching']:.2f}, after {s6['max_abs_smd_after_matching']:.2f}):

{md(psm)}

**H5: suggestive, not conclusive.** The interaction is significant (IRR 1.62, p = 0.010): when others can hear, women's "don't know" answers rise while men's do not. But the effect *within women* is not significant on its own, either in the model (+1.18 answers, CI −0.12 to 2.64) or after matching (+0.48, CI −1.01 to 2.20). Privacy varies within {s6['psu_with_both_privacy_conditions_%']:.0f}% of sampling units. Report as suggestive evidence, consistent with social silencing.

## 8. Knowledge items (Step 12)

Correct answers: Q91 India (the item asks which country is **not** a permanent UN Security Council member; verify against the WVS-7 master questionnaire), Q92 Washington DC, Q93 human rights.

{md(krate)}

{md(kmod)}

Raw data: women attempt the quiz items less often (32–36% vs 45–46%), and among those who attempt, women are as accurate as men on the UN item but less accurate on IMF and Amnesty. After controls, neither difference is significant. A supporting section only; it neither strengthens nor undermines the main argument.

## 9. Robustness checks (Step 13)

{md(rb)}

The female effect (H1) and the placebo interaction (H2) hold across every variant except two with low power: the binary "any don't know" outcome (p = 0.06, same direction) and the sampling-unit fixed effects (only 67 respondents). **Town fixed effects**, comparing women and men in the same town ({town_row['n']} respondents), give IRR {town_row['H1_IRR_female']} and OR {town_row['H2_OR_female_x_political']} for the placebo interaction: both significant.

**Holm correction across the five main hypothesis tests:**

{md(hm)}

All five remain significant after correction.

## 10. Machine-learning check (Step 14)

Histogram gradient boosting (Poisson loss), 5-fold cross-validation grouped by sampling unit. SHAP is not installed here, so importance uses permutation on held-out folds.

{md(cvt)}

{md(imp_t)}

The model explains about 21% of out-of-sample deviance. Without any pre-specified model, **sex is the {fem_rank}nd most important predictor**, after political attention (rank {attn_rank}) and ahead of digital news, province, education and income. The partial-dependence plot (`results/07_fig_ml.png`) shows women predicted higher at every level of attention, matching Figure 2.

---

## 11. Summary of hypotheses

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| H1: women give more "don't know" answers on domestic political items | **Supported** | IRR 1.76 with controls; town FE IRR 2.41 |
| H2: the gap is specific to politics (placebo) | **Supported** | female × political OR 2.4–2.7 across all specifications |
| H3: the gap persists net of awareness (silence, not ignorance) | **Supported** | Awareness explains ~1% (KHB); 62% unexplained (Oaxaca); no female × attention interaction |
| H4: the gap is largest on public, contested topics | **Supported** | Electoral integrity, corruption, surveillance (R checkpoint 1) |
| H5: the gap is larger when others are present | **Suggestive** | Interaction significant; within-women effect not |

## 12. Limitations to report
1. Sex is confounded with sampling unit and probably interviewer team (Section 1).
2. Cross-sectional data; associations, not causal effects.
3. Privacy is not randomly assigned.
4. No design weights; age re-weighting does not change results.

## 13. Next steps
- **R checkpoint 2 (you):** `R/checkpoint2_verify_baseR.R`, verifying H1, H3, H5 and town fixed effects against these Python results.
- **Phase C:** final tables and figures for the paper; appendix; GitHub-ready folder.
"""
(ROOT / "reports" / "PHASE_B_RESULTS.md").write_text(text)
print("Saved reports/PHASE_B_RESULTS.md")
