"""Render paper figures from results/ into ../paper/figures (PDF, IEEE column width)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).parent / "results"
F = Path(__file__).parent.parent / "paper" / "figures"; F.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "pdf.fonttype": 42})
STY = {  # fixed categorical order + secondary encoding (line style / marker) for print & CVD
    "B4 Governed memory (ours)": ("#2a78d6", "-", "o", "B4 governed (ours)"),
    "B1 Hard-coded backend": ("#eb6834", "--", "s", "B1 hard-coded (weekly)"),
    "B2 Log-RAG": ("#1baf7a", "-.", "^", "B2 log-RAG"),
    "B3 Ungoverned memory": ("#eda100", ":", "D", "B3 ungoverned"),
}
W = 3.5

# ---- Fig: daily accuracy timeline (seed 0) --------------------------------
raw = json.load(open(R / "main_raw.json"))
run0 = raw[0]; inc = run0["meta"]["incident"]
fig, ax = plt.subplots(figsize=(W, 2.2))
ax.axvspan(inc["start"], inc["end"], color="#d9d8d3", alpha=0.6, lw=0)
for n, (c, ls, m, lab) in STY.items():
    y = np.array(run0["results"][n]["daily_accuracy"])
    ax.plot(np.arange(len(y)), y, color=c, ls=ls, lw=1.2, label=lab)
ax.set_xlabel("Day of scenario"); ax.set_ylabel("Daily API answer accuracy")
ax.set_ylim(0.35, 1.02); ax.set_xlim(0, 89)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, handlelength=2.2, columnspacing=1.0, borderaxespad=0.2)
fig.tight_layout(pad=0.3); fig.savefig(F / "timeline.pdf"); plt.close(fig)

# ---- Fig: attack-pressure sweep --------------------------------------------
rows = json.load(open(R / "attack_sweep.json"))
fig, ax = plt.subplots(figsize=(W, 2.2))
for n, (c, ls, m, lab) in STY.items():
    rs = sorted([r for r in rows if r["system"] == n], key=lambda r: r["attack_rate_per_day"])
    x = [r["attack_rate_per_day"] for r in rs]; y = [r["accuracy"] for r in rs]; e = [r["accuracy_ci95"] for r in rs]
    ax.errorbar(x, y, yerr=e, color=c, ls=ls, marker=m, ms=4, lw=1.2, capsize=2, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks([0.3, 0.6, 1.2, 2.4, 4.8]); ax.set_xticklabels(["0.3", "0.6", "1.2", "2.4", "4.8"])
ax.set_xlabel("Adversarial messages per day"); ax.set_ylabel("API answer accuracy"); ax.set_xlim(0.25, 5.8)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=2, columnspacing=1.0, borderaxespad=0.2)
fig.tight_layout(pad=0.3); fig.savefig(F / "attack_sweep.pdf"); plt.close(fig)

# ---- Fig: freshness vs safety ---------------------------------------------
main = {r["system"]: r for r in json.load(open(R / "main.json"))}
cad = {r["system"]: r for r in json.load(open(R / "cadence.json"))}
fig, ax = plt.subplots(figsize=(W, 2.0))
lab_off = {"B4 Governed memory (ours)": (6, -2, "left"), "B2 Log-RAG": (6, -2, "left"), "B3 Ungoverned memory": (6, -2, "left")}
for n in ("B4 Governed memory (ours)", "B2 Log-RAG", "B3 Ungoverned memory"):
    c, ls, m, lab = STY[n]; r = main[n]
    x, y = max(r["latency_p90_h"], 0.1), 100 * r["attack_error_rate"]
    ax.scatter(x, y, color=c, marker=m, s=30, zorder=3, edgecolor="#fcfcfb", linewidth=0.8)
    dx, dy, ha = lab_off[n]
    ax.annotate(lab.split(" (")[0] + (" (ours)" if "ours" in lab else ""), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, ha=ha, va="center")
c, ls, m, _ = STY["B1 Hard-coded backend"]
xs, ys = [], []
for d, off in ((1, (0, -10)), (7, (-4, 6)), (14, (6, -3))):
    r = cad[f"B1 release every {d}d"]; xs.append(r["latency_p90_h"]); ys.append(100 * r["attack_error_rate"])
    ax.annotate(f"B1 {d}d", (xs[-1], ys[-1]), xytext=off, textcoords="offset points", fontsize=6.5,
                ha="left" if d == 14 else "center", va="center" if d == 14 else "baseline")
ax.plot(xs, ys, color=c, ls=ls, marker=m, ms=4, lw=1.0)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.06, 900); ax.set_ylim(0.04, 80)
ax.set_xlabel("p90 policy-change latency (h); 0 drawn at 0.1"); ax.set_ylabel("Attack-induced errors (%)")
ax.text(0.98, 0.95, "B1 = hard-coded backend,\nlabel = release cadence", transform=ax.transAxes, ha="right", va="top", fontsize=6, color="#52514e")
fig.tight_layout(pad=0.3); fig.savefig(F / "frontier.pdf"); plt.close(fig)
print("figures written to", F)
