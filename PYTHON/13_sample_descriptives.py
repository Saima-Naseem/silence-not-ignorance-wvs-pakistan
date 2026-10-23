"""
Sample characteristics by sex for the paper (new Table 1).
Sample: the same complete-case sample as the negative binomial models (script 05): complete on all model variables.
Output: results/13_sample_descriptives.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
w = pd.read_csv(ROOT / "data" / "analysis_wide.csv")
model_vars = ["female", "age", "education_isced", "literate", "income_scale", "employed", "urban", "province",
              "attention_index", "news_digital", "news_tv", "others_present", "resp_interest_low", "duration_min",
              "dk_count_domestic_political"]
d = w[w[model_vars].notna().all(axis=1)]

rows = [("Age (mean, years)", "age", "mean"),
        ("Education (mean, ISCED level 0–8)", "education_isced", "mean"),
        ("Literate (%)", "literate", "pct"),
        ("Income scale (mean, 1–10)", "income_scale", "mean"),
        ("Employed (%)", "employed", "pct"),
        ("Urban (%)", "urban", "pct"),
        ("Political attention index (mean of z-scores)", "attention_index", "mean"),
        ("Watches TV news at least weekly (%)", "news_tv_weekly", "pct"),
        ("Others could follow the interview (%)", "others_present", "pct")]
out = []
for label, col, kind in rows:
    vals = {}
    for sex, g in d.groupby("female"):
        x = g[col].dropna()
        vals[sex] = x.mean() * (100 if kind == "pct" else 1)
    out.append({"variable": label, "men": round(vals[0], 2), "women": round(vals[1], 2)})
out.append({"variable": "N", "men": int((d.female == 0).sum()), "women": int((d.female == 1).sum())})
pd.DataFrame(out).to_csv(R / "13_sample_descriptives.csv", index=False)
print(pd.DataFrame(out).to_string(index=False))
