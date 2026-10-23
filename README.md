# Silence, or ignorance? Gender, "don't know" Responses and Political Awareness in Pakistan

Replication materials. All scripts, results, figures and tables for the paper.

**Data:** World Values Survey Wave 7, Pakistan 2018 (N = 1,995), Pakistan national data file, Excel version 5.1 (F00013210-WVS_Wave_7_Pakistan_Excel_v5.1).
**The data are not included** (WVS terms of use do not allow redistribution). Download them free from
[worldvaluessurvey.org](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp) (Wave 7 → Pakistan) and follow the steps below. Please cite the WVS as its terms require.

## How to reproduce everything

| Step | Command | What it does |
|---|---|---|
| 1 | `python python/01_prepare_data.py path/to/WVS_Pak_Data.xlsx` | Audit, recoding, item sets, awareness measures; rebuilds `data/` |
| 2 | `python python/02_descriptives.py` | Descriptives and bivariate tests (Phase A) |
| 3 | `python python/03_write_phaseA_report.py` | Phase A report |
| 4 | `Rscript R/checkpoint1_baseR.R` | Multilevel models for H2 and H4 (base R: MASS, nlme) |
| 5 | `python python/04_validate_estimator.py` | Validates the Python estimators against step 4 |
| 6 | `python python/05_phaseB_models.py` | Diagnostics, H1, H3 (stepwise, KHB, Oaxaca, subgroup, interaction) |
| 7 | `python python/06_phaseB_privacy_knowledge.py path/to/WVS_Pak_Data.xlsx` | H5 interview privacy; knowledge items |
| 8 | `python python/07_phaseB_robustness_ml.py path/to/WVS_Pak_Data.xlsx` | Robustness checks; machine-learning check |
| 9 | `python python/08_write_phaseB_report.py` | Phase B report |
| 10 | `Rscript R/checkpoint2_verify_baseR.R` | Independent R re-estimation of key models, printed next to the Python results |
| 11 | `python python/09_figures.py` | Publication figures (`figures/`) |
| 11a | `python python/11_extra_checks.py path/to/WVS_Pak_Data.xlsx` | Extra check: voting and own gender-norm attitudes |
| 11b | `python python/12_town_norms.py path/to/WVS_Pak_Data.xlsx` | Extra check: surrounding norms (men's traditionalism in the town) |
| 11c | `python python/13_sample_descriptives.py` | Sample characteristics by sex (paper Table 1) |
| 12 | `python python/10_tables_spec.py` then `node paper/build_docx.js` | Word files of main tables/figures and online appendix (`paper/`) |

Run all commands from this folder. R scripts expect the working directory to be this folder.

**Software:** Python 3.12 (`requirements.txt`); R 4.6.1 with built-in MASS and nlme only; Node.js with the `docx` package for step 12.
`R/checkpoint1_multilevel.R` is an lme4 version of step 4 for computers where lme4 loads.

## Contents
- `python/`, `R/`: scripts (`python/stats_utils.py` holds the estimators)
- `figures/`: figures at 300 dpi. Paper Figure 1 = `Figure2_attention.png`, Figure 2 = `Figure3_topic_gradient.png`, Figure 3 = `Figure4_privacy.png`; `Figure1_item_gaps.png`.
- `paper/`: `Main_Tables_Figures.docx`, `Online_Appendix.docx`, `tables_spec.json`, `build_docx.js`

## Main results
Women give about 1.8 times as many "don't know" answers as men on domestic political questions (with controls), but not on non-political questions. Political awareness does not explain the gap, which is concentrated on judgements about electoral integrity, corruption and surveillance. Key models were re-estimated in R with identical coefficients.

## Citation

If you use these materials, please cite the article above. Citation metadata are in `CITATION.cff`.

## Contact

Saima Naseem, saimanaseemhere@gmail.com


## Table numbering in the paper
Table 1 sample characteristics; Table 2 "don't know" by item set; Table 3 negative binomial models (M0–M3 and M4 town fixed effects); Table 4 multilevel placebo models. Online Appendix Tables A1–A17.

## Licence
Code: MIT (see `LICENSE`). The WVS data are not covered by this licence.
