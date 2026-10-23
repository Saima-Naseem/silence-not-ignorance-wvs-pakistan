"""
Extra check 2e: surrounding norms.
The silence argument concerns the norms around a woman, not only her own attitudes. Men and women were
interviewed in separate sampling units, but towns contain both. For each town, the average gender-role
traditionalism of the MEN interviewed there (index of Q29, Q30, Q31) measures the local normative climate.
  2e  Women only: domestic DK count ~ men's traditionalism in town (+ own traditionalism, resources, awareness, province)
  2f  Same on the non-political DK count (placebo)
  2g  Men only (comparison): own domestic DK ~ traditionalism of the OTHER men in the town (leave-one-out)
Cluster-robust SEs by town. Towns with fewer than 5 men interviewed are excluded.
Outputs: results/12_town_norms.csv, results/12_town_norms_info.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from stats_utils import make_X, fit, table, PROVINCES

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
XLSX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/project/WVS_Pak_Data.xlsx")
wide = pd.read_csv(ROOT / "data" / "analysis_wide.csv"); town = pd.read_csv(ROOT / "data" / "town.csv")
raw = pd.read_excel(XLSX, sheet_name="original", usecols=lambda c: c.split(":")[0] in ("D_INTERVIEW", "Q29", "Q30", "Q31"))
raw.columns = [c.split(":")[0] for c in raw.columns]; raw = raw.rename(columns={"D_INTERVIEW": "id"})
w = wide.merge(town, on="id").merge(raw, on="id")
z = lambda s: (s - s.mean()) / s.std()
trad = pd.concat({q: (5 - w[q]).where(w[q] > 0) for q in ["Q29", "Q30", "Q31"]}, axis=1)
w["trad"] = trad.mean(axis=1).where(trad.notna().sum(axis=1) >= 2)

men = w[(w.female == 0) & w.trad.notna()]
tsum = men.groupby("town").trad.agg(["sum", "count"])
w = w.merge(tsum, left_on="town", right_index=True, how="left")
# women: mean of all men in town; men: leave-one-out mean of the other men
w["town_men_trad"] = np.where(w.female == 1, w["sum"] / w["count"],
                              (w["sum"] - w.trad.fillna(0)) / (w["count"] - w.trad.notna().astype(int)))
w["men_in_town"] = np.where(w.female == 1, w["count"], w["count"] - w.trad.notna().astype(int))

RES = ["age_z", "educ_z", "literate", "income_z", "employed", "urban"]; AWR = ["attention_z", "digital_z", "tv_z"]
d = pd.DataFrame({"id": w.id, "town": w.town, "female": w.female, "y": w.dk_count_domestic_political,
    "y_non": w.dk_count_non_political, "age_z": z(w.age), "educ_z": z(w.education_isced), "literate": w.literate,
    "income_z": z(w.income_scale), "employed": w.employed, "urban": w.urban, "province": w.province,
    "attention_z": z(w.attention_index), "digital_z": z(w.news_digital), "tv_z": z(w.news_tv),
    "own_trad_z": z(w.trad), "town_men_trad": w.town_men_trad, "men_in_town": w.men_in_town})
d = d[d.men_in_town >= 5].dropna().reset_index(drop=True)
d["town_men_trad_z"] = z(d.town_men_trad)
PF = {"province": PROVINCES}
rows = []
def run(label, df, y, vars_):
    levels = [p_ for p_ in PROVINCES if (df.province == p_).any()]      # drop provinces absent from this subsample
    r = fit("negbin", df[y], make_X(df, vars_ + RES + AWR, {"province": levels}), cluster=df.town); t = table(r).loc["town_men_trad_z"]
    rows.append({"check": label, "outcome": "59 domestic items" if y == "y" else "62 non-political items (placebo)",
                 "n": len(df), "towns": df.town.nunique(), "IRR_per_SD": round(t.exp_b, 3), "ci_low": round(t.exp_ci_low, 3),
                 "ci_high": round(t.exp_ci_high, 3), "p": t.p})
wo = d[d.female == 1].reset_index(drop=True); mo = d[d.female == 0].reset_index(drop=True)
run("2e women: men's traditionalism in town", wo, "y", ["town_men_trad_z"])
run("2e women: men's traditionalism in town + own traditionalism", wo, "y", ["town_men_trad_z", "own_trad_z"])
run("2f women: placebo", wo, "y_non", ["town_men_trad_z", "own_trad_z"])
run("2g men (comparison): other men's traditionalism in town", mo, "y", ["town_men_trad_z", "own_trad_z"])
out = pd.DataFrame(rows); out.to_csv(R / "12_town_norms.csv", index=False)
info = (f"Towns with >= 5 men interviewed: women n = {len(wo)} in {wo.town.nunique()} towns; men n = {len(mo)} in {mo.town.nunique()} towns.\n"
        f"Men's town traditionalism (1-4 scale): mean {d.town_men_trad.mean():.2f}, SD across respondents {d.town_men_trad.std():.2f}.")
(R / "12_town_norms_info.txt").write_text(info)
print(out.assign(p=out.p.round(4)).to_string(index=False)); print(info)
