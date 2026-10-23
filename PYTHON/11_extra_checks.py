"""
Extra checks requested before writing.

CHECK 1 (voting / direct experience). Several electoral-integrity items ask about things seen at polling stations.
  If women's "don't know" reflects less direct experience, the gap should shrink or vanish among women who vote.
  1a  Electoral-integrity DK count (12 items): female effect before/after controlling for voting status
  1b  Female effect among regular voters only (electoral-integrity count and all 59 domestic items)
  1c  Placebo interaction (female x political) among regular voters only
CHECK 2 (gender-norm mechanism). If silence reflects norms that politics is a male domain, women who endorse
  "men make better political leaders" (Q29) should give more political "don't know" answers, but not more
  non-political ones.
  2a  Women only: domestic DK count on Q29 endorsement (+ resources, awareness, province)
  2b  Women only: same on non-political DK count (placebo)
  2c  Full sample: female x Q29 interaction on domestic DK count
  2d  Robustness: gender-norms index (Q29, Q30, Q31)
Outputs: results/11_*.csv, results/11_summary.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from stats_utils import make_X, fit, table, ame_binary, PROVINCES

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
XLSX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/project/WVS_Pak_Data.xlsx")
wide = pd.read_csv(ROOT / "data" / "analysis_wide.csv")
long = pd.read_csv(ROOT / "data" / "analysis_long.csv")
raw = pd.read_excel(XLSX, sheet_name="original",
                    usecols=lambda c: c.split(":")[0] in ("D_INTERVIEW", "Q222", "Q29", "Q30", "Q31"))
raw.columns = [c.split(":")[0] for c in raw.columns]
raw = raw.rename(columns={"D_INTERVIEW": "id"})
w = wide.merge(raw, on="id")
# electoral-integrity block count
items = pd.read_csv(ROOT / "data" / "item_sets.csv")
ei = items.item[items.item_block == "electoral_integrity"]
eic = long[long.item.isin(ei)].groupby("id").dk.agg(["sum", "count"]).rename(columns={"sum": "dk_ei", "count": "n_ei"})
w = w.merge(eic, left_on="id", right_index=True)

z = lambda s: (s - s.mean()) / s.std()
vote = w.Q222.where(w.Q222 > 0)
w["vote_status"] = vote.map({1: "regular voter", 2: "regular voter", 3: "never votes", 4: "not allowed to vote"})
w["regular_voter"] = (vote.isin([1, 2])).astype(float).where(vote.notna())
w["never_votes"] = (vote == 3).astype(float).where(vote.notna())
w["not_allowed"] = (vote == 4).astype(float).where(vote.notna())
for q in ["Q29", "Q30", "Q31"]:
    w[q + "_trad"] = (5 - w[q]).where(w[q] > 0)          # 4 = agree strongly (traditional) ... 1 = disagree strongly
w["leaders_agree"] = (w.Q29_trad >= 3).astype(float).where(w.Q29_trad.notna())
d = pd.DataFrame({
    "id": w.id, "psu": w.psu, "female": w.female, "y": w.dk_count_domestic_political, "y_ei": w.dk_ei,
    "y_non": w.dk_count_non_political, "share_dom": w.dk_share_domestic_political, "share_non": w.dk_share_non_political,
    "n_dom": w.n_items_domestic_political, "n_non": w.n_items_non_political,
    "age_z": z(w.age), "educ_z": z(w.education_isced), "literate": w.literate, "income_z": z(w.income_scale),
    "employed": w.employed, "urban": w.urban, "province": w.province, "attention_z": z(w.attention_index),
    "digital_z": z(w.news_digital), "tv_z": z(w.news_tv), "vote_status": w.vote_status, "regular_voter": w.regular_voter,
    "never_votes": w.never_votes, "not_allowed": w.not_allowed, "Q29_trad": w.Q29_trad, "Q29_z": z(w.Q29_trad),
    "leaders_agree": w.leaders_agree, "Q30_trad": w.Q30_trad, "Q31_trad": w.Q31_trad})
RES = ["age_z", "educ_z", "literate", "income_z", "employed", "urban"]
AWR = ["attention_z", "digital_z", "tv_z"]
PF = {"province": PROVINCES}
core = ["y", "female"] + RES + AWR + ["province"]
out, lines = [], []
def rec(check, model, term, r, n, extra=""):
    t = table(r).loc[term]
    out.append({"check": check, "model": model, "term": term, "estimate": t.exp_b, "ci_low": t.exp_ci_low,
                "ci_high": t.exp_ci_high, "p": t.p, "n": n, "note": extra})
    lines.append(f"{check:4s} {model:62s} {term:24s} {t.exp_b:6.3f} [{t.exp_ci_low:.2f}, {t.exp_ci_high:.2f}] p = {t.p:.4f}  n = {n}")

# ---------------------------------------------------------------- descriptives
desc = (d.dropna(subset=["vote_status"]).groupby(["vote_status", "female"])
        .agg(n=("y", "size"), dk_domestic_pct=("share_dom", "mean"), dk_electoral_mean=("y_ei", "mean")))
desc["dk_domestic_pct"] *= 100; desc = desc.round(2); desc.to_csv(R / "11_vote_descriptives.csv")
nd = (d.dropna(subset=["leaders_agree"]).groupby(["leaders_agree", "female"])
      .agg(n=("y", "size"), dk_domestic_pct=("share_dom", "mean"), dk_nonpolitical_pct=("share_non", "mean")))
nd[["dk_domestic_pct", "dk_nonpolitical_pct"]] *= 100; nd = nd.round(2); nd.to_csv(R / "11_norm_descriptives.csv")
lines += ["DESCRIPTIVES: voting status x sex", desc.to_string(), "",
          "DESCRIPTIVES: agrees men make better political leaders (Q29) x sex", nd.to_string(), ""]

# ---------------------------------------------------------------- CHECK 1
d1 = d.dropna(subset=core + ["regular_voter", "y_ei"]).reset_index(drop=True)
b0 = lambda df: make_X(df, ["female"] + RES + AWR, PF)
b1 = lambda df: make_X(df, ["female"] + RES + AWR + ["never_votes", "not_allowed"], PF)
r0 = fit("negbin", d1.y_ei, b0(d1), cluster=d1.psu); rec("1a", "Electoral-integrity DK: base model", "female", r0, len(d1))
r1 = fit("negbin", d1.y_ei, b1(d1), cluster=d1.psu); rec("1a", "Electoral-integrity DK: + voting status", "female", r1, len(d1))
rec("1a", "Electoral-integrity DK: + voting status", "never_votes", r1, len(d1))
rec("1a", "Electoral-integrity DK: + voting status", "not_allowed", r1, len(d1))
rd = fit("negbin", d1.y, b1(d1), cluster=d1.psu); rec("1a", "All 59 domestic items: + voting status", "female", rd, len(d1))
v = d1[d1.regular_voter == 1].reset_index(drop=True)
rv = fit("negbin", v.y_ei, b0(v), cluster=v.psu); rec("1b", "Regular voters only: electoral-integrity DK", "female", rv, len(v),
     f"women {int(v.female.sum())}")
a_ei = ame_binary(rv, v, b0)
rv2 = fit("negbin", v.y, b0(v), cluster=v.psu); rec("1b", "Regular voters only: all 59 domestic items", "female", rv2, len(v))
h = pd.concat([v.assign(political=1, share=v.share_dom, n=v.n_dom), v.assign(political=0, share=v.share_non, n=v.n_non)], ignore_index=True)
rp = fit("logit", h.share, make_X(h, ["female", "political"] + RES + AWR, PF, [("female", "political")]), w=h.n, cluster=h.psu)
rec("1c", "Regular voters only: placebo (fractional logit)", "female:political", rp, len(v))
lines.append(f"     Regular voters, predicted electoral-integrity DK answers (of 12): men {a_ei['pred0']:.2f}, women {a_ei['pred1']:.2f}, "
             f"difference {a_ei['ame']:.2f} [{a_ei['ci_low']:.2f}, {a_ei['ci_high']:.2f}]")

# ---------------------------------------------------------------- CHECK 2
d2 = d.dropna(subset=core + ["Q29_z"]).reset_index(drop=True)
wo = d2[d2.female == 1].reset_index(drop=True)
bn = lambda df: make_X(df, ["Q29_z"] + RES + AWR, PF)
ra = fit("negbin", wo.y, bn(wo), cluster=wo.psu); rec("2a", "Women only: domestic DK on Q29 (traditional, z)", "Q29_z", ra, len(wo))
rb = fit("negbin", wo.y_non, bn(wo), cluster=wo.psu); rec("2b", "Women only: NON-political DK on Q29 (placebo)", "Q29_z", rb, len(wo))
mo = d2[d2.female == 0].reset_index(drop=True)
rm = fit("negbin", mo.y, bn(mo), cluster=mo.psu); rec("2a", "Men only: domestic DK on Q29 (comparison)", "Q29_z", rm, len(mo))
rc = fit("negbin", d2.y, make_X(d2, ["female", "Q29_z"] + RES + AWR, PF, [("female", "Q29_z")]), cluster=d2.psu)
rec("2c", "Full sample: female x Q29 on domestic DK", "female", rc, len(d2))
rec("2c", "Full sample: female x Q29 on domestic DK", "Q29_z", rc, len(d2))
rec("2c", "Full sample: female x Q29 on domestic DK", "female:Q29_z", rc, len(d2))
# women: within-person placebo version (does Q29 raise political DK relative to own non-political DK?)
hw = pd.concat([wo.assign(political=1, share=wo.share_dom, n=wo.n_dom), wo.assign(political=0, share=wo.share_non, n=wo.n_non)], ignore_index=True)
rw = fit("logit", hw.share, make_X(hw, ["Q29_z", "political"] + RES + AWR, PF, [("Q29_z", "political")]), w=hw.n, cluster=hw.psu)
rec("2b", "Women only: Q29 x political item (within-person placebo)", "Q29_z:political", rw, len(wo))
# predicted DK for women who agree vs disagree (binary version)
bb = lambda df: make_X(df, ["leaders_agree"] + RES + AWR, PF)
wob = wo.dropna(subset=["leaders_agree"]).reset_index(drop=True)
rbb = fit("negbin", wob.y, bb(wob), cluster=wob.psu); rec("2a", "Women only: agrees men better leaders (binary)", "leaders_agree", rbb, len(wob))
ab = ame_binary(rbb, wob, bb, var="leaders_agree")
lines.append(f"     Women, predicted domestic DK answers (of 59): disagree {ab['pred0']:.2f}, agree {ab['pred1']:.2f}, "
             f"difference {ab['ame']:.2f} [{ab['ci_low']:.2f}, {ab['ci_high']:.2f}]")
# 2d gender-norms index
idx = d2[["Q29_trad", "Q30_trad", "Q31_trad"]]
cc = idx.dropna(); k = 3
alpha = k / (k - 1) * (1 - cc.var().sum() / cc.sum(axis=1).var())
d2["norms_z"] = z(idx.mean(axis=1).where(idx.notna().sum(axis=1) >= 2))
wo2 = d2[(d2.female == 1) & d2.norms_z.notna()].reset_index(drop=True)
ri = fit("negbin", wo2.y, make_X(wo2, ["norms_z"] + RES + AWR, PF), cluster=wo2.psu)
rec("2d", f"Women only: gender-norms index Q29-31 (alpha = {alpha:.2f})", "norms_z", ri, len(wo2))
ri2 = fit("negbin", wo2.y_non, make_X(wo2, ["norms_z"] + RES + AWR, PF), cluster=wo2.psu)
rec("2d", "Women only: gender-norms index on NON-political DK (placebo)", "norms_z", ri2, len(wo2))

res = pd.DataFrame(out); res.to_csv(R / "11_extra_checks.csv", index=False)
# variables for the R verification script (respondent-level; stays in data/, excluded from GitHub)
w[["id", "Q222", "Q29", "Q30", "Q31", "dk_ei"]].to_csv(ROOT / "data" / "extra_vars.csv", index=False)
txt = "\n".join(lines); (R / "11_summary.txt").write_text(txt); print(txt)
