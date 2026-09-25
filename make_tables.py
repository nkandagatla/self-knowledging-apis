"""Generate LaTeX tables (paper/tables/*.tex) directly from results/ so the paper cannot drift from the data."""
import json
from pathlib import Path

R = Path(__file__).parent / "results"
T = Path(__file__).parent.parent / "paper" / "tables"; T.mkdir(parents=True, exist_ok=True)
L = lambda f: json.load(open(R / f))


def pct(r, k, d=1):
    return f"{100 * r[k]:.{d}f}$\\pm${100 * r[k + '_ci95']:.{d}f}"


def num(r, k, d=1):
    return f"{r[k]:.{d}f}"


main = {r["system"]: r for r in L("main.json")}
short = {"B1 Hard-coded backend": "B1 Hard-coded, weekly release", "B2 Log-RAG": "B2 Log-RAG",
         "B3 Ungoverned memory": "B3 Ungoverned memory", "B4 Governed memory (ours)": "\\textbf{B4 Governed (ours)}"}
rows = []
for n, lab in short.items():
    r = main[n]
    lat = "0.0" if r["latency_p90_h"] < 0.05 else f"{r['latency_p90_h']:.1f}"
    rows.append(f"{lab} & {pct(r, 'accuracy')} & {pct(r, 'attack_error_rate', 2)} & {pct(r, 'stale_error_rate', 2)} & "
                f"{num(r, 'bad_writes_activated')} & {lat} & {r['pii_items_in_memory']:.0f} \\\\")
(T / "tab_main.tex").write_text(r"""\begin{table*}[t]
\centering
\caption{Main results over 90 simulated days (mean $\pm$ 95\% CI across 10 seeds; 13{,}500 API calls and $\approx$10{,}900 runtime messages per seed, of which @BAD@ are adversarial and @LEG@ are legitimate policy changes). Accuracy, attack-induced and stale error are shares of API calls. Bad writes activated: adversarial or unauthorised messages that at some point became the value served for a policy slot. PII: personal-data items retained in agent memory at day 90.}
\label{tab:main}
\begin{tabular}{lcccccc}
\toprule
System & Accuracy (\%) & \shortstack{Attack-induced\\errors (\%)} & \shortstack{Stale\\errors (\%)} & \shortstack{Bad writes\\activated} & \shortstack{p90 policy\\latency (h)} & \shortstack{PII items\\in memory} \\
\midrule
""".replace("@BAD@", f'{main["B4 Governed memory (ours)"]["bad_writes_total"]:.0f}').replace("@LEG@", f'{main["B4 Governed memory (ours)"]["legit_updates"]:.1f}')
                           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

abl = L("ablation.json")
rows = []
full = abl[0]
for r in abl:
    d = 100 * (r["accuracy"] - full["accuracy"])
    lab = r["system"].replace("B4 full", "Full B4").replace("&", "\\&")
    lab = ("$-$ " + lab[2:]) if lab.startswith("- ") else lab
    rows.append(f"{lab} & {pct(r, 'accuracy')} & "
                f"{'--' if r is full else f'{d:+.1f}'} & {pct(r, 'attack_error_rate', 2)} & "
                f"{num(r, 'stat_review_items_per_day', 2)} & {r['pii_items_in_memory']:.0f} \\\\")
(T / "tab_ablation.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Ablation of the write-path controls (10 seeds). $\Delta$ is the change in accuracy (percentage points) versus the full design.}
\label{tab:ablation}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
Variant & Acc. (\%) & $\Delta$ & Attack err. (\%) & Review/day & PII \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

rev = L("reviewer.json")
rows = [f"{r['system'].replace('reviewer ', '').replace('p_bad', '$p_{bad}$').replace('p_legit', '$p_{legit}$')} & "
        f"{pct(r, 'accuracy')} & {pct(r, 'attack_error_rate', 2)} & {num(r, 'bad_writes_activated')} \\\\" for r in rev]
(T / "tab_reviewer.tex").write_text(r"""\begin{table}[t]
\centering
\caption{Sensitivity to reviewer error. $p_{bad}$: probability a reviewer approves a bad quarantined item; $p_{legit}$: probability a legitimate item is approved (default 0.02 / 0.97).}
\label{tab:reviewer}
\footnotesize
\begin{tabular}{lccc}
\toprule
Reviewer model & Acc. (\%) & Attack err. (\%) & Bad writes act. \\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print("tables written to", T)
for f in ("main", "ablation", "reviewer"):
    print((T / f"tab_{f}.tex").read_text())
