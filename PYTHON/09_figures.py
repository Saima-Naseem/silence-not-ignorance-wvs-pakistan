"""
Phase C: publication figures (300 dpi, readable in black and white).
Figure 1  Gender gap on each of the 59 domestic political items (Phase A)
Figure 2  Predicted "don't know" answers by political attention and sex (H3, Phase B)
Figure 3  Model-based gender gap by topic block (H4, R checkpoint 1)
Figure 4  Predicted "don't know" answers by interview privacy and sex (H5, Phase B)
Outputs: figures/Figure1-4.png
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"; F = ROOT / "figures"; F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
MEN = dict(color="#8c8c8c", marker="o", ms=5, ls="none")
WOMEN = dict(color="black", marker="^", ms=5, ls="none")

def r_block(text, header):
    """Parse a whitespace table printed by R after a header line."""
    lines = text.split(header, 1)[1].strip("\n").splitlines()
    rows = []
    for ln in lines[1:]:
        if not ln.strip() or not re.match(r"^\s*\d+\s", ln): break
        rows.append(ln.split())
    return rows

def r_gaps():
    txt = (R / "R_checkpoint1_output.txt").read_text()
    rows = r_block(txt, "Predicted 'don't know' % and gender gap by block:")
    return pd.DataFrame([r[1:] for r in rows], columns=["block", "men_pct", "women_pct", "gap_pp", "ci_low", "ci_high"]).astype(
        {"men_pct": float, "women_pct": float, "gap_pp": float, "ci_low": float, "ci_high": float})

# ---------------------------------------------------------------- Figure 1
it = pd.read_csv(R / "02_item_level_tests.csv")
dom = it[it.item_set == "domestic_political"].sort_values("gap_pp").reset_index(drop=True)
short = lambda s: re.sub(r"\s+", " ", s.replace("How often in country's elections:", "Elections:")
                         .replace("Involved in corruption:", "Corruption:").replace("Confidence:", "Confidence:")
                         .replace("Government has the right:", "Govt may:").replace("Political system:", "System:"))[:58]
fig, ax = plt.subplots(figsize=(6.5, 9.2))
y = np.arange(len(dom))
ax.hlines(y, dom["men_dk_%"], dom["women_dk_%"], color="#c8c8c8", lw=1)
ax.plot(dom["men_dk_%"], y, label="Men", **MEN); ax.plot(dom["women_dk_%"], y, label="Women", **WOMEN)
ax.set_yticks(y); ax.set_yticklabels([f"{a}  {short(b)}" for a, b in zip(dom["item"], dom["label"])], fontsize=6)
ax.set_xlabel("Respondents answering \"don't know\" (%)"); ax.set_ylim(-1, len(dom))
ax.grid(axis="x", color="#e6e6e6", lw=.6); ax.legend(loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(F / "Figure1_item_gaps.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 2
pp = pd.read_csv(R / "05_pred_by_attention.csv")
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for f_, col, ls, lab in [(0, "#8c8c8c", "--", "Men"), (1, "black", "-", "Women")]:
    s = pp[pp.female == f_]
    ax.plot(s.attention_z, s.pred, color=col, ls=ls, lw=1.6, label=lab)
    ax.fill_between(s.attention_z, s.lo, s.hi, color=col, alpha=.15, lw=0)
ax.set_xlabel("Political attention index (standard deviations)")
ax.set_ylabel("Predicted \"don't know\" answers (of 59)")
ax.grid(color="#e6e6e6", lw=.6); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(F / "Figure2_attention.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 3
g = r_gaps()
names = {"democracy": "Meaning of democracy (13)", "regime_preferences": "Regime preferences (5)",
         "confidence_institutions": "Confidence in institutions (17)", "corruption": "Corruption (9)",
         "surveillance": "Government surveillance (3)", "electoral_integrity": "Electoral integrity (12)"}
g["label"] = g.block.map(names); g = g.sort_values("gap_pp").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(5.5, 3.0))
yy = np.arange(len(g))
ax.hlines(yy, g.ci_low, g.ci_high, color="black", lw=1.2)
ax.plot(g.gap_pp, yy, "s", color="black", ms=5)
ax.axvline(0, color="#8c8c8c", lw=.8, ls=":")
ax.set_yticks(yy); ax.set_yticklabels(g.label)
ax.set_xlabel("Gender gap in \"don't know\"\n(women − men, percentage points)")
ax.grid(axis="x", color="#e6e6e6", lw=.6)
fig.tight_layout(); fig.savefig(F / "Figure3_topic_gradient.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 4
pe = pd.read_csv(R / "06_privacy_effect_by_sex.csv").set_index("group")
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for i, (grp, mk) in enumerate([("men", MEN), ("women", WOMEN)]):
    vals = [pe.loc[grp, "pred_private"], pe.loc[grp, "pred_others_present"]]
    ax.plot([0, 1], vals, color=mk["color"], marker=mk["marker"], ms=6, lw=1.4,
            ls="--" if grp == "men" else "-", label=grp.capitalize())
    ax.annotate(f"{pe.loc[grp,'difference']:+.2f} [{pe.loc[grp,'ci_low']:.2f}, {pe.loc[grp,'ci_high']:.2f}]",
                (1.04, vals[1]), fontsize=7.5, va="center")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Interview private", "Others could follow"])
ax.set_xlim(-.25, 1.75); ax.set_ylim(0, 6); ax.set_ylabel("Predicted \"don't know\" answers (of 59)")
ax.grid(axis="y", color="#e6e6e6", lw=.6); ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(F / "Figure4_privacy.png"); plt.close(fig)
print("Saved figures:", sorted(p.name for p in F.glob("*.png")))
