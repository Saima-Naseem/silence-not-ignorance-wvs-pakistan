"""
Phase C: assembles the content of every paper table (main text and online appendix)
from the results files and writes paper/tables_spec.json, which R-free Node script
paper/build_docx.js turns into Word documents. No number is typed by hand.
"""
from pathlib import Path
import json, re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"; P = ROOT / "paper"; P.mkdir(exist_ok=True)
stars = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
pf = lambda p: "< .001" if p < .001 else f"{p:.3f}".replace("0.", ".")
f2 = lambda x: f"{x:.2f}"
LAB = {"female": "Woman", "political": "Political item", "female:political": "Woman × political item",
       "age_z": "Age (z)", "educ_z": "Education (z)", "literate": "Literate", "income_z": "Income (z)",
       "employed": "Employed", "urban": "Urban", "province[Sindh]": "Sindh", "province[KP]": "Khyber Pakhtunkhwa",
       "province[Balochistan]": "Balochistan", "provinceSindh": "Sindh", "provinceKP": "Khyber Pakhtunkhwa",
       "provinceBalochistan": "Balochistan", "attention_z": "Political attention (z)", "digital_z": "Digital news use (z)",
       "tv_z": "TV news use (z)", "others_present": "Others could follow interview", "resp_interest_low": "Respondent not very interested in interview",
       "duration_z": "Interview duration (z)", "political:educ_z": "Political item × education", "political:literate": "Political item × literate",
       "political:attention_z": "Political item × attention", "political:digital_z": "Political item × digital news",
       "political:tv_z": "Political item × TV news"}
T = {"main": [], "appendix": []}
def add(where, num, title, columns, rows, note, widths=None, landscape=False):
    T[where].append({"number": num, "title": title, "columns": columns, "rows": rows, "note": note,
                     "widths": widths, "landscape": landscape})

# ================================================================ MAIN TABLE 1
t1 = pd.read_csv(R / "02_table1_dk_by_set_and_sex.csv"); tt = pd.read_csv(R / "02_respondent_level_tests.csv")
wide = pd.read_csv(ROOT / "data" / "analysis_wide.csv")
names = {"domestic_political": "Domestic political", "non_political": "Non-political (placebo)",
         "international": "International organisations", "knowledge": "Factual knowledge"}
rows = []
for _, r in t1.iterrows():
    x = tt[tt.item_set == r.item_set].iloc[0]
    rows.append([names[r.item_set], str(r.n_items), f2(r["men_dk_%"]), f2(r["women_dk_%"]), f2(r.gap_pp),
                 f2(r.ratio_w_to_m), f"{x.cohens_d:.2f}", pf(x.p_holm)])
diff = wide.dk_share_domestic_political - wide.dk_share_non_political
dm, dw = diff[wide.female == 0].mean() * 100, diff[wide.female == 1].mean() * 100
x = tt[tt.item_set.str.startswith("difference")].iloc[0]
rows.append(["Within-person difference (domestic − placebo)", "–", f2(dm), f2(dw), f2(dw - dm), "–", f"{x.cohens_d:.2f}", pf(x.p_holm)])
add("main", "1", "\"Don't know\" responses by item set and sex, WVS-7 Pakistan 2018",
    ["Item set", "Items", "Men (%)", "Women (%)", "Gap (pp)", "Women ÷ men", "Cohen's d", "p"], rows,
    "Mean percentage of items answered \"don't know\" per respondent. N = 1,995 (1,037 men, 958 women). p from Welch t-tests on "
    "respondent-level counts, Holm-adjusted across the five comparisons; Mann–Whitney tests give the same conclusions (Appendix Table A4). "
    "Within-person difference = each respondent's domestic share minus their non-political share.",
    widths=[2900, 700, 900, 1000, 900, 1000, 900, 800])

# ================================================================ MAIN TABLE 2
order = ["female", "age_z", "educ_z", "literate", "income_z", "employed", "urban", "province[Sindh]", "province[KP]",
         "province[Balochistan]", "attention_z", "digital_z", "tv_z", "others_present", "resp_interest_low", "duration_z"]
mods = {m: pd.read_csv(R / f"05_negbin_{m}.csv", index_col=0) for m in ["M0", "M1", "M2", "M3"]}
step = pd.read_csv(R / "05_stepwise_female.csv").set_index("model")
rows = []
for term in order:
    row = [LAB[term]]
    for m, t in mods.items():
        if term in t.index:
            r = t.loc[term]; row.append(f"{r.exp_b:.2f}{stars(r.p)}\n({r.exp_ci_low:.2f}–{r.exp_ci_high:.2f})")
        else: row.append("")
    rows.append(row)
rows.append(["Dispersion (alpha)"] + [f"{step.loc[m,'alpha']:.2f}" for m in mods])
rows.append(["Log-likelihood"] + [f"{step.loc[m,'loglik']:.1f}" for m in mods])
rows.append(["Women − men, predicted \"don't know\" answers"] + [f"{step.loc[m,'AME_count']:.2f}\n({step.loc[m,'AME_ci'].replace('-', '–')})" for m in mods])
rows.append(["Respondents / sampling units"] + ["1,845 / 197"] * 4)
add("main", "2", "Negative binomial models of \"don't know\" answers on 59 domestic political items",
    ["", "M0: Sex only", "M1: + Resources", "M2: + Awareness", "M3: + Interview"], rows,
    "Incidence rate ratios with 95% confidence intervals from cluster-robust standard errors (197 sampling units). Reference province: Punjab. "
    "Same complete-case sample in all models. Predicted difference = average marginal effect of being a woman on the number of \"don't know\" "
    "answers (out of 59), 95% simulation interval. Key models were re-estimated in R (MASS::glm.nb) with identical coefficients. "
    "* p < .05, ** p < .01, *** p < .001.", widths=[3000, 1520, 1520, 1520, 1520])

# ================================================================ MAIN TABLE 3 (R checkpoint 1)
txt = (R / "R_checkpoint1_output.txt").read_text()
def r_coef_block(header):
    block = txt.split(header, 1)[1].split("\n\n")[0].strip().splitlines()[1:]
    out = {}
    for ln in block:
        parts = ln.split()
        if len(parts) == 5: out[parts[0]] = dict(est=float(parts[1]), se=float(parts[2]), OR=float(np.exp(float(parts[1]))), p=float(parts[4]))  # OR from the estimate, not R's 3-decimal print (avoids double rounding)
    return out
h2a = r_coef_block("---- H2-M1: female x political item, controls as main effects ----")
h2b = r_coef_block("---- H2-M2: + political x (education, literacy, attention, digital news, TV news) ----")
gl = r_coef_block("---- Cross-check: quasi-binomial GLM, cluster-robust SE by psu (H2-M1) ----") if False else None
order3 = ["female", "political", "female:political", "educ_z", "literate", "income_z", "age_z", "employed", "urban",
          "provinceSindh", "provinceKP", "provinceBalochistan", "attention_z", "digital_z", "tv_z", "political:educ_z",
          "political:literate", "political:attention_z", "political:digital_z", "political:tv_z"]
fmt = lambda d: f"{d['OR']:.2f}{stars(d['p'])}\n({np.exp(d['est']-1.96*d['se']):.2f}–{np.exp(d['est']+1.96*d['se']):.2f})"
rows = [[LAB[k], fmt(h2a[k]) if k in h2a else "", fmt(h2b[k]) if k in h2b else ""] for k in order3]
rows.append(["Respondents / rows / sampling units", "1,845 / 3,690 / 197", "1,845 / 3,690 / 197"])
add("main", "3", "Multilevel placebo models: \"don't know\" on political versus non-political items",
    ["", "Model 1", "Model 2: + awareness and education × item type"], rows,
    "Odds ratios with 95% confidence intervals. Each respondent contributes two observations: the share of \"don't know\" answers on 59 "
    "domestic political items and on 62 non-political items (weighted by number of items). Multilevel logistic models (penalized "
    "quasi-likelihood, R MASS::glmmPQL) with random intercepts for sampling unit and respondent. The key term is Woman × political item. "
    "A quasi-binomial GLM with cluster-robust standard errors gives OR = 2.48 (p < .001) for this term. * p < .05, ** p < .01, *** p < .001.",
    widths=[3600, 2700, 2700])

# ================================================================ APPENDIX
items = pd.read_csv(ROOT / "data" / "item_sets.csv")
add("appendix", "A1", "Item sets and coding",
    ["Item set", "Items", "WVS questions", "Role in analysis"],
    [["Domestic political", "59", "Q65–81, Q112–120, Q196–198, Q224–234A, Q235–239, Q241–253", "Main outcome"],
     ["Non-political", "62", "Q1–27 (excluding Q4), Q46–63, Q158–175", "Placebo comparison"],
     ["International organisations", "8", "Q82–89", "Contrast"],
     ["Factual knowledge", "3", "Q91–93", "Supporting analysis"],
     ["Political awareness", "11", "Q4, Q199, Q200, Q201–208", "Predictor / benchmark"]],
    "\"Don't know\" (−1) coded 1, valid answers 0; \"no answer\" (−2) kept as a separate flag. Q4 (importance of politics) is an awareness item and "
    "was removed from the placebo set. Q82_SAARC duplicates Q82 and was not used. No items were coded −3 to −5.", widths=[2200, 700, 4300, 1800])

rel = (R / "01_reliability_factor.txt").read_text()
load = re.search(r"News sources EFA.*?\n(.*?)\nItems assigned", rel, re.S).group(1).strip().splitlines()[1:]
alpha = lambda name: re.search(name + r".*?alpha = ([0-9.]+)", rel).group(1)
lab_news = {"news_newspaper": "Daily newspaper", "news_tv": "TV news", "news_radio": "Radio news", "news_mobile": "Mobile phone",
            "news_email": "Email", "news_internet": "Internet", "news_social": "Social media", "news_talk": "Talk with friends"}
rows = [[lab_news[l.split()[0]], l.split()[1], l.split()[2]] for l in load]
rows += [["Cronbach's alpha: political attention index (Q4, Q199, Q200)", alpha("Political attention index"), ""],
         ["Cronbach's alpha: digital media index (mobile, email, internet, social media)", alpha("digital media index"), ""],
         ["Cronbach's alpha: traditional media index (newspaper, TV, radio): not used", alpha("traditional media index"), ""]]
add("appendix", "A2", "Awareness measures: factor loadings and reliability",
    ["News source / measure", "Factor 1", "Factor 2"], rows,
    "Exploratory factor analysis of news sources, two factors, varimax rotation, standardized complete cases (n = 1,764). "
    "News exposure is measured by the digital-media index and TV news as a single item.", widths=[5600, 1700, 1700])

miss = pd.read_csv(R / "01_missingness_by_sex.csv")
add("appendix", "A3", "Missing data on predictors by sex (%)", ["Variable", "Men", "Women"],
    [[r.iloc[0], f2(r.iloc[1]), f2(r.iloc[2])] for _, r in miss.iterrows()],
    "Complete cases on core model variables: 1,845. Multiple imputation results in Table A11.", widths=[5000, 2000, 2000])

tt2 = tt.copy()
add("appendix", "A4", "Respondent-level tests by item set",
    ["Comparison", "Welch t", "p", "p (Holm)", "Cohen's d", "Mann–Whitney p", "Rank-biserial r"],
    [[names.get(r.item_set, "Within-person difference"), f2(r.welch_t), pf(r.p_welch), pf(r.p_holm), f"{r.cohens_d:.2f}",
      pf(r.p_mw) if pd.notna(r.p_mw) else "–", f"{r.rank_biserial:.2f}" if pd.notna(r.rank_biserial) else "–"] for _, r in tt2.iterrows()],
    "Positive values: women higher.", widths=[2800, 900, 900, 900, 900, 1300, 1300])

it = pd.read_csv(R / "02_item_level_tests.csv")
dom = it[it.item_set == "domestic_political"].sort_values("gap_pp", ascending=False)
add("appendix", "A5", "\"Don't know\" by sex for each domestic political item",
    ["Item", "Wording (abbreviated)", "Men (%)", "Women (%)", "Gap (pp)", "Cohen's h", "p (Holm)"],
    [[r["item"], re.sub(r"\s+", " ", r["label"])[:70], f2(r["men_dk_%"]), f2(r["women_dk_%"]), f2(r["gap_pp"]), f"{r['cohens_h']:.2f}", pf(r["p_holm_within_set"])]
     for _, r in dom.iterrows()],
    "Chi-square tests, Holm correction within the 59 items. Women higher on 58 of 59 items; 43 significant after correction. "
    "On the 62 non-political items, none is significant after correction.", widths=[650, 4150, 800, 900, 800, 850, 850])

khb = pd.read_csv(R / "05_khb.csv"); oax = pd.read_csv(R / "05_oaxaca.csv", index_col=0)
add("appendix", "A6", "Decomposing the gender gap: KHB and Oaxaca–Blinder",
    ["Decomposition", "Estimate", "95% CI", "% of gap"],
    [[f"KHB: indirect effect via {r.mediators}", f"{r.indirect_b:.3f}", f"{r.indirect_ci_low:.3f} to {r.indirect_ci_high:.3f}", f"{r.pct_of_total_explained:.1f}"] for _, r in khb.iterrows()] +
    [[f"Oaxaca–Blinder: {k.replace('_', ' ')}", f"{r.estimate_pp:.2f} pp", f"{r.ci_low:.2f} to {r.ci_high:.2f}", f"{r.pct_of_gap:.1f}"] for k, r in oax.iterrows()],
    "KHB: log-count scale, negative binomial models, cluster bootstrap (300 reps). Oaxaca–Blinder: pooled twofold decomposition of the "
    "percentage of domestic items answered \"don't know\", cluster bootstrap (1,000 reps).", widths=[4600, 1300, 2000, 1100])

sub = pd.read_csv(R / "05_high_awareness_subgroup.csv"); inter = pd.read_csv(R / "05_negbin_female_x_attention.csv", index_col=0)
rows = [[f"Woman, {r.group} (n = {r.n})", f"{r.IRR_female:.2f}", r.IRR_ci.replace("-", "–"), pf(r.p)] for _, r in sub.iterrows()]
rows += [[LAB.get(k, "Woman × political attention"), f"{inter.loc[k,'exp_b']:.2f}", f"{inter.loc[k,'exp_ci_low']:.2f}–{inter.loc[k,'exp_ci_high']:.2f}", pf(inter.loc[k, "p"])]
         for k in ["female", "attention_z", "female:attention_z"]]
add("appendix", "A7", "High-awareness subgroup and woman × attention interaction",
    ["Term", "IRR", "95% CI", "p"], rows,
    "Negative binomial models with resource and province controls (and awareness in the interaction model); cluster-robust SEs. "
    "High awareness: very or somewhat interested in politics, discusses politics at least occasionally, watches TV news at least weekly.",
    widths=[5200, 1100, 1600, 1100])

blk = re.search(r"Predicted 'don't know' % and gender gap by block:\n(.*?)\n\n", txt, re.S).group(1).splitlines()[1:]
wald = re.search(r"Wald_chi2\s+df\s+p\s*\n\s*([0-9.e+-]+)\s+([0-9.e+-]+)\s+([0-9.e+-]+)", txt)
bn = {"democracy": "Meaning of democracy", "regime_preferences": "Regime preferences", "confidence_institutions": "Confidence in institutions",
      "corruption": "Corruption", "surveillance": "Government surveillance", "electoral_integrity": "Electoral integrity"}
add("appendix", "A8", "Gender gap by topic block (multilevel model)",
    ["Block", "Men (%)", "Women (%)", "Gap (pp)", "95% interval"],
    [[bn[l.split()[1]], l.split()[2], l.split()[3], l.split()[4], f"{l.split()[5]} to {l.split()[6]}"] for l in blk],
    f"Predicted probabilities for a typical respondent (random effects at zero), with all controls including awareness. "
    f"Joint Wald test of woman × block: χ² = {float(wald.group(1)):.1f}, df = {int(float(wald.group(2)))}, p < .001. R MASS::glmmPQL.",
    widths=[3400, 1200, 1300, 1200, 1900])

pdesc = pd.read_csv(R / "06_privacy_descriptives.csv"); pint = pd.read_csv(R / "06_negbin_female_x_privacy.csv", index_col=0)
peff = pd.read_csv(R / "06_privacy_effect_by_sex.csv"); psm = pd.read_csv(R / "06_psm_privacy.csv")
bal = pd.read_csv(R / "06_psm_balance.csv")
rows = [[f"{'Women' if r.female else 'Men'}, {'others could follow' if r.others_present else 'private'}: mean \"don't know\" (n = {r['size']})", f"{r['mean']:.2f}", "", ""] for _, r in pdesc.iterrows()]
rows += [[f"IRR: {LAB.get(k, 'Woman × others could follow')}", f"{pint.loc[k,'exp_b']:.2f}", f"{pint.loc[k,'exp_ci_low']:.2f}–{pint.loc[k,'exp_ci_high']:.2f}", pf(pint.loc[k, 'p'])]
         for k in ["female", "others_present", "female:others_present"]]
rows += [[f"Predicted effect of others present, {r.group}", f"{r.difference:+.2f}", f"{r.ci_low:.2f} to {r.ci_high:.2f}", ""] for _, r in peff.iterrows()]
rows += [[f"Matched estimate (ATT), {r.group} ({r.matched_treated} matched)", f"{r.ATT_extra_dk_answers:+.2f}", f"{r.ci_low:.2f} to {r.ci_high:.2f}", ""] for _, r in psm.iterrows()]
add("appendix", "A9", "Interview privacy (H5)", ["Quantity", "Estimate", "95% CI", "p"], rows,
    f"Negative binomial model with all controls; cluster-robust SEs. The R re-estimate gives p = .018 for the interaction. Matching: 1:1 nearest "
    f"neighbour on the propensity score within each sex, caliper 0.2 SD; largest absolute standardized difference {bal.smd_before.abs().max():.2f} "
    f"before and {bal.smd_after.abs().max():.2f} after matching; cluster bootstrap CIs (500 reps).", widths=[5200, 1100, 1600, 1100])

kr = pd.read_csv(R / "06_knowledge_rates.csv"); km = pd.read_csv(R / "06_knowledge_models.csv")
qn = {"Q91": "UN Security Council", "Q92": "IMF headquarters", "Q93": "Amnesty International"}
rows = [[f"{qn[r['item']]}, {'women' if r['female'] else 'men'}", f"{r.attempt_pct:.1f}", f"{r.correct_pct_all:.1f}", f"{r.correct_pct_of_attempts:.1f}"] for _, r in kr.iterrows()]
rows += [[f"Logit, {r.stage}: OR for women {r.OR_female:.2f} ({r.OR_ci.replace('-', '–')}), p = {pf(r.p)}", "", "", ""] for _, r in km.iterrows()]
add("appendix", "A10", "Knowledge items: attempting an answer and answering correctly",
    ["Item / model", "Attempted (%)", "Correct, all (%)", "Correct, of attempts (%)"], rows,
    "Correct answers: Q91 India (the item asks which country is not a permanent member), Q92 Washington DC, Q93 human rights. "
    "Logit models stack the three items with item fixed effects, resource, awareness and province controls; cluster-robust SEs.",
    widths=[5000, 1300, 1300, 1400])

rob = pd.read_csv(R / "07_robustness.csv")
add("appendix", "A11", "Robustness checks",
    ["Variant", "N", "H1: IRR woman", "95% CI", "p", "H2: OR woman × political", "95% CI", "p"],
    [[r.variant[0].upper() + r.variant[1:], str(r.n), f"{r.H1_IRR_female:.2f}", str(r.H1_CI).replace("-", "–"), pf(r.H1_p),
      f"{r.H2_OR_female_x_political:.2f}" if pd.notna(r.H2_OR_female_x_political) else "–",
      str(r.H2_CI).replace("-", "–") if pd.notna(r.H2_CI) else "–", pf(r.H2_p) if pd.notna(r.H2_p) else "–"] for _, r in rob.iterrows()],
    "H1: negative binomial, resource and province controls. H2: fractional logit, two rows per respondent, cluster-robust SEs. "
    "The binary-outcome row reports an odds ratio. Sampling-unit fixed effects use only the seven mixed-sex units. Town fixed effects compare "
    "women and men in the same town.", widths=[3400, 600, 800, 900, 700, 1000, 900, 700], landscape=True)

hm = pd.read_csv(R / "07_holm_main_tests.csv")
add("appendix", "A12", "Multiple-testing correction across the five main tests", ["Test", "p", "p (Holm)"],
    [[r.hypothesis, pf(r.p), pf(r.p_holm)] for _, r in hm.iterrows()], "Holm correction. H5 remains significant using the R p-value (.018).",
    widths=[6000, 1500, 1500])

r2 = (R / "R_checkpoint2_output.txt").read_text()
rep = re.findall(r"^(.+?)\s+R est\s+(-?[0-9.]+) \| Py est\s+(-?[0-9.]+) \| R se\s+([0-9.]+) \| Py se\s+([0-9.]+)", r2, re.M)
add("appendix", "A13", "Independent replication in R", ["Model term", "Estimate (R)", "Estimate (Python)", "SE (R)", "SE (Python)"],
    [[a.strip(), b, c, d, e] for a, b, c, d, e in rep] + [["Town fixed effects: IRR woman", "2.411", "2.411", "", ""]],
    "Python: maximum likelihood with dispersion estimated jointly in the cluster-robust covariance. R: MASS::glm.nb, which holds the dispersion "
    "parameter fixed in the sandwich, so standard errors differ slightly. No conclusion changes.", widths=[3600, 1350, 1350, 1350, 1350])

cv = pd.read_csv(R / "07_ml_cv.csv"); imp = pd.read_csv(R / "07_ml_permutation_importance.csv")
add("appendix", "A14", "Machine-learning check: predictors of \"don't know\" answers",
    ["Rank", "Feature", "Permutation importance", "SD across folds"],
    [[str(r["rank"]), r.feature, f"{r.importance_mean:.3f}", f"{r.importance_sd_across_folds:.3f}"] for _, r in imp.iterrows()],
    f"Histogram gradient boosting (Poisson loss); 5-fold cross-validation grouped by sampling unit; out-of-sample deviance explained "
    f"{cv['deviance_explained_%'].mean():.1f}% on average. Importance = increase in Poisson deviance when a feature is permuted on held-out folds.",
    widths=[800, 3200, 2500, 2500])

prov = pd.read_csv(R / "05_dk_by_province_sex.csv")
g = wide.groupby("psu").female.mean()
rows = [[r.province, str(int(r.n)), f2(r.men), f2(r.women), f2(r.gap_pp)] for _, r in prov.iterrows()]
rows += [["Sampling units: all men / all women / mixed", f"{int((g==0).sum())} / {int((g==1).sum())} / {int(((g>0)&(g<1)).sum())}", "", "", ""]]
add("appendix", "A15", "Province pattern and sex composition of sampling units",
    ["Province", "N", "Men DK (%)", "Women DK (%)", "Gap (pp)"], rows,
    "Raw percentages of domestic political items answered \"don't know\". Men and women were interviewed in separate sampling units, very likely by "
    "same-sex interviewer teams; interviewer sex is not recorded. Intraclass correlation (sampling unit) of the domestic \"don't know\" share = 0.33.",
    widths=[3600, 1500, 1300, 1300, 1300])



# ================================================================ INTERPRETATIONS (numbers read from the same files)
def put(num, text):
    for where in T:
        for t in T[where]:
            if t["number"] == num: t["interpretation"] = text; return
    raise KeyError(num)
r1 = t1.set_index("item_set")
put("1", f"Women answer \"don't know\" to {r1.loc['domestic_political','women_dk_%']:.1f}% of domestic political questions, against "
    f"{r1.loc['domestic_political','men_dk_%']:.1f}% for men: {r1.loc['domestic_political','ratio_w_to_m']:.1f} times as often. On non-political "
    f"questions the two are virtually identical ({r1.loc['non_political','women_dk_%']:.2f}% vs {r1.loc['non_political','men_dk_%']:.2f}%), which rules out "
    "general explanations such as literacy, difficulty with the questionnaire or a tendency to say \"don't know\" to everything. International and "
    "knowledge items produce high \"don't know\" rates for both sexes, where genuine unfamiliarity is plausible; they serve as a contrast. The "
    "within-person comparison gives the largest effect: the gap is specific to politics.")
st_ = step
put("2", f"Without controls, women give {mods['M0'].loc['female','exp_b']:.2f} times as many \"don't know\" answers as men (M0). Controlling for "
    f"education, income and other resources reduces this to {mods['M1'].loc['female','exp_b']:.2f} (M1): part of the gap reflects women's "
    f"lower schooling and income. Adding political awareness (M2) leaves the gap essentially unchanged ({mods['M2'].loc['female','exp_b']:.2f}), even "
    f"though attention (IRR {mods['M2'].loc['attention_z','exp_b']:.2f}) and TV news (IRR {mods['M2'].loc['tv_z','exp_b']:.2f}) strongly reduce "
    "\"don't know\" answers for everyone. Awareness lowers \"don't know\" answers, but it does not explain why women give more of them. Interview "
    f"conditions (M3) change little ({mods['M3'].loc['female','exp_b']:.2f}). In practical terms, women give about "
    f"{st_.loc['M3','AME_count']:.1f} more \"don't know\" answers out of 59 than comparable men. This supports H1 and H3.")
put("3", f"The key term, woman × political item, has an odds ratio of {h2a['female:political']['OR']:.2f}: compared with their own answers on "
    f"non-political questions, women's odds of saying \"don't know\" rise about {h2a['female:political']['OR']:.1f} times more than men's on political "
    f"questions. The main effect for women (OR {h2a['female']['OR']:.2f}) shows that on non-political questions women are, if anything, slightly less "
    "likely than men to say \"don't know\". Because each respondent is compared with herself, stable personal traits such as literacy or response "
    f"style cancel out. In Model 2, letting education, literacy and awareness affect political items differently barely changes the key term "
    f"(OR {h2b['female:political']['OR']:.2f}). Education does reduce political \"don't know\" answers (OR {h2b['political:educ_z']['OR']:.2f}), but it "
    "does not explain the gender gap. This supports H2.")
put("A1", "The outcome (59 items) and the placebo set (62 items) are of similar size, so the comparison between them is balanced. Removing Q4 from "
    "the placebo set keeps it free of political content.")
put("A2", "Interest, discussion and importance of politics form an acceptable attention index for three items. News sources split into a "
    "digital factor and TV news; newspapers, TV and radio do not form a reliable \"traditional media\" scale, so TV news is used on its own.")
put("A3", "Missing data are low (under 3%) for all predictors except income. The multiple-imputation check (Table A11) shows that "
    "excluding incomplete cases does not change the results.")
put("A4", "Parametric and non-parametric tests agree. The effect for domestic political items is moderate (Cohen's d about 0.5) and negligible "
    "for non-political items.")
top3 = dom.head(3)["label"].str.replace(r"\s+", " ", regex=True).tolist()
put("A5", "The gap is close to universal: women are higher on 58 of 59 items. The largest gaps involve judging elections, the media, parliament, "
    "local authorities and the civil service. The smallest or no gaps involve the police, the government, the army and television, institutions "
    "that are either part of everyday experience or have a clear public image, about which most people readily hold an opinion.")
ka = khb.set_index("mediators"); oa = oax
put("A6", f"Political awareness explains only about {ka.iloc[0]['pct_of_total_explained']:.0f}% of the gender gap once resources are held "
    f"constant, and its confidence interval includes zero. Resources and awareness together explain about {ka.iloc[1]['pct_of_total_explained']:.0f}%. "
    f"The Oaxaca–Blinder decomposition gives the same picture: about {oa.loc['unexplained','pct_of_gap']:.0f}% of the gap is not explained by "
    "differences in resources or awareness. This residual is consistent with gendered silence, but it is a residual: it shows what the gap is not "
    "explained by, not directly what it is.")
sb_ = sub.set_index("group")
put("A7", f"Among highly aware respondents the gap is smaller (IRR {sb_.loc['high awareness','IRR_female']:.2f}) and not statistically significant, "
    "partly because this group is only a third of the sample. The woman × attention interaction is close to one and not significant: awareness "
    "reduces \"don't know\" answers at a similar rate for both sexes, and women remain higher at every level (Figure 2). The gap narrows among "
    "the most engaged but does not disappear.")
put("A8", "The gender gap differs significantly across topics. It is concentrated in electoral integrity, government surveillance and corruption, "
    "all judgements about whether those in power act fairly or honestly. It is small for the meaning of democracy and not significant for "
    "regime preferences or confidence in institutions. Women's silence is strongest where answering means passing judgement on authority. An "
    "alternative to consider: several electoral-integrity items ask about things observed at polling stations, which women may have experienced "
    "less directly. Table A16 tests this: the gap persists among women who vote regularly.")
pd_ = pdesc.set_index(["female", "others_present"])["mean"]
put("A9", f"When others could follow the interview, women's \"don't know\" answers rise (raw mean {pd_.loc[(1,0)]:.2f} to {pd_.loc[(1,1)]:.2f}) while "
    f"men's do not ({pd_.loc[(0,0)]:.2f} to {pd_.loc[(0,1)]:.2f}). The interaction is significant (IRR {pint.loc['female:others_present','exp_b']:.2f}), "
    "which fits the idea that women hold back political opinions when they may be overheard. But the effect among women alone is not significant "
    "in either the model or the matched comparison, and privacy was not randomly assigned. This is suggestive, not conclusive, evidence for H5.")
put("A10", "Women attempt the quiz questions less often than men, and in the raw data are less accurate on two of the three items. After controls, "
    "neither difference is significant. This is consistent with women being less willing to guess, as earlier research suggests, but these three "
    "items cannot settle the question; they play a supporting role only.")
tw = rob[rob.variant.str.startswith("town")].iloc[0]
put("A11", f"The female effect and the placebo interaction hold across every variant except two with little statistical power: the binary "
    f"outcome (same direction, p = .06) and sampling-unit fixed effects (67 respondents). The most demanding check, comparing women and men living "
    f"in the same town, gives a larger effect (IRR {tw.H1_IRR_female:.2f}; placebo OR {tw.H2_OR_female_x_political:.2f}). The results do not depend "
    "on coding choices, item selection, interview quality, missing data or the age imbalance of the sample.")
put("A12", "All five main tests remain significant after correcting for multiple testing, so the findings are not an artefact of running many tests.")
put("A13", "Two independent implementations, custom maximum-likelihood code in Python and R's established MASS package, give identical coefficients "
    "and nearly identical standard errors. The results do not depend on the software or on coding errors.")
put("A14", f"A model that selects predictors on its own, without any theory, ranks political attention first and sex second, ahead of education, "
    f"income and province. Sex therefore matters independently of the variables the theory controls for. The model explains about "
    f"{cv['deviance_explained_%'].mean():.0f}% of out-of-sample variation, so most differences between individuals remain unexplained, as is usual "
    "for survey responses.")
pv = prov.set_index("province")
put("A15", f"The gender gap appears in Punjab, Sindh and especially Khyber Pakhtunkhwa ({pv.loc['KP','gap_pp']:.1f} percentage points), which fits "
    "more restrictive gender norms there. Balochistan shows the reverse, but with only 91 respondents in 10 sampling units this is too small to "
    "interpret. Because men and women were interviewed in separate sampling units, respondent sex cannot be fully separated from interviewer team. "
    "Same-sex interviewing should make women more comfortable, which works against the finding, and the within-person and within-town checks "
    "make an interviewer explanation unlikely, but it is a limitation to report.")


# ================================================================ A16-A17: extra checks (11_extra_checks.py)
ex = pd.read_csv(R / "11_extra_checks.csv")
vd = pd.read_csv(R / "11_vote_descriptives.csv"); ndsc = pd.read_csv(R / "11_norm_descriptives.csv")
smry = (R / "11_summary.txt").read_text()
TERM = {"female": "Woman", "never_votes": "Never votes (ref.: regular voter)", "not_allowed": "Not allowed to vote (ref.: regular voter)",
        "female:political": "Woman × political item", "Q29_z": "Men better political leaders (Q29, z)",
        "female:Q29_z": "Woman × Q29", "Q29_z:political": "Q29 × political item", "leaders_agree": "Agrees men better leaders (binary)",
        "norms_z": "Gender-norms index Q29–Q31 (z)"}
def exrows(checks):
    return [[f"{r.model.replace('female x', 'woman ×').replace(' x ', ' × ')}: {TERM.get(r.term, r.term)}", f"{r.estimate:.2f}{stars(r.p)}", f"{r.ci_low:.2f}–{r.ci_high:.2f}", pf(r.p), str(r.n)]
            for _, r in ex[ex.check.isin(checks)].iterrows()]
rows = [[f"{r.vote_status}, {'women' if r.female else 'men'} (n = {int(r.n)})", f"{r.dk_domestic_pct:.2f}%", f"{r.dk_electoral_mean:.2f}", "", ""] for _, r in vd.iterrows()]
rows += exrows(["1a", "1b", "1c"])
add("appendix", "A16", "Check 1: does direct electoral experience explain the gap? Voting in national elections (Q222)",
    ["Quantity", "Estimate", "95% CI", "p", "N"], rows,
    "Descriptive rows: % of domestic items answered \"don't know\" and mean \"don't know\" answers on the 12 electoral-integrity items. Model rows: "
    "incidence rate ratios (negative binomial) or odds ratio (fractional logit, placebo), with resource, awareness and province controls; "
    "cluster-robust SEs. Regular voter = always or usually votes. * p < .05, ** p < .01, *** p < .001.",
    widths=[5200, 1000, 1200, 800, 800])
vv = ex.set_index(["check", "model", "term"])
g = lambda c, m, t: vv.loc[(c, m, t)]
put("A16", f"If women said \"don't know\" about elections mainly because they see less of them, the gap should shrink once voting is "
    f"controlled, and vanish among women who vote. It does neither: controlling for voting leaves the gap on electoral-integrity items unchanged "
    f"(IRR {g('1a','Electoral-integrity DK: base model','female').estimate:.2f} to {g('1a','Electoral-integrity DK: + voting status','female').estimate:.2f}), "
    f"and among regular voters women still give {g('1b','Regular voters only: electoral-integrity DK','female').estimate:.2f} times as many "
    f"\"don't know\" answers on these items. The placebo interaction also holds among voters (OR {g('1c','Regular voters only: placebo (fractional logit)','female:political').estimate:.2f}). "
    f"The direct-experience explanation is therefore weak. Respondents who say they are \"not allowed to vote\" give more \"don't know\" answers "
    f"(IRR {g('1a','Electoral-integrity DK: + voting status','not_allowed').estimate:.2f}), and women in this group have the highest rate of all, "
    "which is consistent with household gatekeeping of women's political participation.")
rows = [[f"{'Agrees' if r.leaders_agree else 'Disagrees'}: men better political leaders, {'women' if r.female else 'men'} (n = {int(r.n)})",
         f"{r.dk_domestic_pct:.2f}%", f"{r.dk_nonpolitical_pct:.2f}%", "", ""] for _, r in ndsc.iterrows()]
rows += exrows(["2a", "2b", "2c", "2d"])
tn = pd.read_csv(R / "12_town_norms.csv")
rows += [[f"Surrounding norms, {r.check[3:]} [{r.outcome}]", f"{r.IRR_per_SD:.2f}{stars(r.p)}",
          f"{r.ci_low:.2f}–{r.ci_high:.2f}", pf(r.p), f"{r.n} ({r.towns} towns)"] for _, r in tn.iterrows()]
add("appendix", "A17", "Check 2: the gender-norm mechanism. Endorsing \"men make better political leaders\" (Q29)",
    ["Quantity", "Estimate", "95% CI", "p", "N"], rows,
    "Descriptive rows: % of domestic political and of non-political items answered \"don't know\". Model rows: incidence rate ratios per standard "
    "deviation of traditional endorsement (negative binomial) or odds ratio (fractional logit, within-person placebo), with resource, awareness and "
    "province controls; cluster-robust SEs. Respondents answering \"don't know\" to Q29 are excluded. Gender-norms index: Q29, Q30 (university more "
    "important for boys), Q31 (men better business executives). Surrounding norms: mean gender-role traditionalism (Q29–Q31) of the men "
    "interviewed in the respondent's town (towns with at least 5 men; for men, the other men in the town); SEs clustered by town. "
    "* p < .05, ** p < .01, *** p < .001.",
    widths=[5200, 1000, 1200, 800, 800])
t_w = tn[tn.check.str.contains(r"\+ own")].iloc[0]; t_m = tn[tn.check.str.startswith("2g")].iloc[0]
put("A17", f"Among women, stronger agreement that men make better political leaders goes with more political \"don't know\" answers "
    f"(IRR {g('2a','Women only: domestic DK on Q29 (traditional, z)','Q29_z').estimate:.2f} per standard deviation) but fewer non-political ones "
    f"(IRR {g('2b','Women only: NON-political DK on Q29 (placebo)','Q29_z').estimate:.2f}). The within-person test is clearest: for women, endorsing the "
    f"norm raises \"don't know\" answers on political items relative to their own non-political answers (OR {g('2b','Women only: Q29 x political item (within-person placebo)','Q29_z:political').estimate:.2f}, p = .001). "
    f"Among men the norm is unrelated to \"don't know\" answers (IRR {g('2a','Men only: domestic DK on Q29 (comparison)','Q29_z').estimate:.2f}). "
    "The broader gender-norms index gives the same pattern. This is direct evidence for the norms mechanism behind \"silence\": women who see politics "
    "as a male domain are the ones who withhold political judgements. Two cautions: the effect is modest (the binary agree/disagree contrast, p = .08, and the "
    "full-sample woman × Q29 interaction, p = .055, are not significant), and norms and answers are measured in the same survey, so the direction of the "
    "relationship cannot be established. The test of surrounding norms is inconclusive: women in towns where men hold more traditional views give "
    f"somewhat more \"don't know\" answers (IRR {t_w.IRR_per_SD:.2f}), but this is not significant with only {t_w.towns} towns, and the same "
    f"association appears among men (IRR {t_m.IRR_per_SD:.2f}), so it is not specific to women.")


# ================================================================ PAPER LAYOUT (Final Draft v2, 11 Oct 2026)
# The paper reports a sample table first and adds a town fixed-effects column to the count models.
# Main tables are renumbered: sample characteristics = 1, "don't know" by item set = 2, count models = 3, placebo = 4.
EMPTY = "Empty cells indicate that a variable is not included in that model."
m1, m2, m3 = [next(x for x in T["main"] if x["number"] == n) for n in ("1", "2", "3")]
m1["note"] = ("Mean percentage of items answered \"don't know\" per respondent. N = 1,995 (1,037 men, 958 women). p from Welch t-tests on "
              "respondent-level counts, Holm-adjusted across the five comparisons. Within-person difference = each respondent's domestic share minus "
              "her or his non-political share.")
# count models: add M4 (town fixed effects) from the robustness results
rb = pd.read_csv(R / "07_robustness.csv")
tw = rb[rb.variant.str.startswith("town fixed effects")].iloc[0]
lo, hi = tw.H1_CI.split("-")
included = {LAB[k] for k in ["age_z", "educ_z", "literate", "income_z", "employed", "urban"]}
m2["columns"] = m2["columns"] + ["M4: Town fixed effects"]
new_rows = []
for row in m2["rows"]:
    lab = row[0]
    if lab == LAB["female"]: row = row + [f"{tw.H1_IRR_female:.2f}{stars(tw.H1_p)}\n({float(lo):.2f}–{float(hi):.2f})"]
    elif lab in included: row = row + ["Included"]
    elif lab.startswith("Respondents"): row = ["Respondents"] + ["1,845"] * 4 + [f"{int(tw.n):,}"]
    else: row = row + [""]
    if lab.startswith("Dispersion"): new_rows.append(["Town fixed effects", "No", "No", "No", "No", "Yes"])
    new_rows.append(row)
m2["rows"] = new_rows
m2["widths"] = [2580, 1300, 1300, 1300, 1300, 1300]
m2["note"] = ("Incidence rate ratios with 95% confidence intervals from standard errors clustered by sampling unit (197 units). Reference province: "
              "Punjab. " + EMPTY + " M0–M3 use the same complete-case sample. Predicted difference = average marginal effect of being a woman on the "
              "number of \"don't know\" answers (out of 59), with 95% simulation interval. M4 includes the resource controls of M1 (coefficients not "
              "shown) and replaces province with fixed effects for the " + str(int(rb.variant.str.extract(r"(\d+) towns")[0].dropna().iloc[0])) +
              " towns in which both women and men were interviewed; its sample is all complete cases on these variables in those towns. "
              "* p < .05, ** p < .01, *** p < .001.")
m3["note"] = ("Odds ratios with 95% confidence intervals. " + EMPTY + " Each respondent contributes two observations: the share of \"don't know\" "
              "answers on 59 domestic political items and on 62 non-political items (weighted by number of items). Multilevel logistic models "
              "(penalised quasi-likelihood, R MASS::glmmPQL) with random intercepts for sampling unit and respondent. The key term is Woman × political "
              "item. * p < .05, ** p < .01, *** p < .001.")
for x, n in ((m3, "4"), (m2, "3"), (m1, "2")): x["number"] = n
# new Table 1: sample characteristics by sex (python/13_sample_descriptives.py)
sd = pd.read_csv(R / "13_sample_descriptives.csv")
fmt1 = lambda v: f"{v:,.0f}" if float(v).is_integer() and v > 100 else f"{v:.2f}".replace("-", "−")
T["main"].insert(0, {"number": "1", "title": "Sample characteristics by sex, WVS-7 Pakistan 2018", "columns": ["Variable", "Men", "Women"],
    "rows": [[r.variable, fmt1(r.men), fmt1(r.women)] for _, r in sd.iterrows()],
    "note": "Complete-case sample used in the main models (N = 1,845).", "widths": [5000, 2000, 2000], "landscape": False})

(P / "tables_spec.json").write_text(json.dumps(T, indent=1, ensure_ascii=False))
print("Main tables:", len(T["main"]), " Appendix tables:", len(T["appendix"]))
