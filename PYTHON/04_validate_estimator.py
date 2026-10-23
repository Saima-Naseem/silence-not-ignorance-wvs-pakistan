"""
Validates stats_utils against independent results before Phase B uses it.

Test 1: reproduce the R checkpoint 1 cross-check (quasi-binomial GLM, H2-M1, cluster-robust SE by psu)
        from results/R_checkpoint1_output.txt. Coefficients and SEs should match to ~3-4 decimals.
Test 2: unpenalized logit vs scikit-learn LogisticRegression (coefficients should match).
Output: results/04_validation.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from stats_utils import make_X, fit, PROVINCES

ROOT = Path(__file__).resolve().parents[1]
wide = pd.read_csv(ROOT / "data" / "analysis_wide.csv")
z = lambda s: (s - s.mean()) / s.std()
d = pd.DataFrame({"id": wide.id, "psu": wide.psu, "female": wide.female, "age_z": z(wide.age),
                  "educ_z": z(wide.education_isced), "literate": wide.literate, "income_z": z(wide.income_scale),
                  "employed": wide.employed, "urban": wide.urban, "province": wide.province,
                  "attention_z": z(wide.attention_index), "digital_z": z(wide.news_digital), "tv_z": z(wide.news_tv)})
keep = d.notna().all(axis=1); d, w = d[keep], wide[keep]
h2 = pd.concat([d.assign(political=1, share=w.dk_share_domestic_political.values, n_items=w.n_items_domestic_political.values),
                d.assign(political=0, share=w.dk_share_non_political.values, n_items=w.n_items_non_political.values)])
num = ["female", "political", "age_z", "educ_z", "literate", "income_z", "employed", "urban"]
X = make_X(h2, num, {"province": PROVINCES}, [("female", "political")])
r = fit("logit", h2.share, X, w=h2.n_items, cluster=h2.psu)

R_vals = {"(Intercept)": (-4.6885, 0.2003), "female": (-0.2830, 0.2027), "political": (1.5257, 0.1488),
          "educ_z": (-0.2588, 0.0726), "income_z": (-0.1607, 0.0604), "province[Balochistan]": (-1.3312, 0.6060),
          "female:political": (0.9091, 0.2094)}
lines = [f"Test 1: R cross-check replication (n respondents = {len(d)}, rows = {len(h2)}, clusters = {r['n_clusters']})",
         f"{'term':24s} {'R est':>9s} {'Py est':>9s} {'R se':>8s} {'Py se':>8s}"]
ok = True
for k, (be, se) in R_vals.items():
    i = r["names"].index(k)
    lines.append(f"{k:24s} {be:9.4f} {r['coef'][i]:9.4f} {se:8.4f} {r['se'][i]:8.4f}")
    ok &= abs(be - r["coef"][i]) < 0.001 and abs(se - r["se"][i]) < 0.001
lines.append(f"Test 1 {'PASSED' if ok else 'FAILED'} (tolerance 0.001)\n")

# Test 2: binary logit vs scikit-learn
yb = (w.dk_count_domestic_political > 0).astype(int)
Xb = make_X(d, ["female", "age_z", "educ_z", "literate", "income_z"], {"province": PROVINCES})
r2 = fit("logit", yb, Xb)
sk = LogisticRegression(penalty=None, max_iter=5000, tol=1e-10).fit(Xb.drop(columns="(Intercept)"), yb)
skc = np.r_[sk.intercept_, sk.coef_.ravel()]
diff = np.abs(skc - r2["coef"]).max()
lines.append(f"Test 2: logit vs scikit-learn, max coefficient difference = {diff:.2e} -> {'PASSED' if diff < 1e-4 else 'FAILED'}")
txt = "\n".join(lines); print(txt)
(ROOT / "results" / "04_validation.txt").write_text(txt)
