"""Run every experiment reported in the paper.

    python run_experiments.py            # 10 seeds, ~2-4 min on a laptop CPU
    python run_experiments.py --seeds 3  # quick check

Outputs: results/*.json, results/*.csv (aggregates with 95% CIs).
"""
from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path

import numpy as np

from skai.harness import run
from skai.scenario import generate, load_data
from skai.systems import HardCodedBackend, LogRAG, UngovernedMemory, GovernedMemory, GovConfig

OUT = Path(__file__).parent / "results"
KEYS = ["accuracy", "attack_error_rate", "stale_error_rate", "noise_error_rate", "bad_writes_activated",
        "bad_writes_total", "legit_updates", "legit_adopted", "legit_missed", "latency_median_h", "latency_p90_h",
        "pii_items_in_memory", "memory_entries", "write_path_us_median", "stat_review_items_per_day",
        "stat_quarantined", "stat_rejected_injection", "stat_approved", "stat_denied", "stat_revoked",
        "stat_constants_changed", "stat_releases", "stat_committed", "stat_reaffirmed"]
T975 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26, 20: 2.09, 30: 2.05}


def ci(xs):
    xs = np.array([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(xs) == 0:
        return (float("nan"), float("nan"))
    if len(xs) == 1:
        return (float(xs[0]), 0.0)
    return float(xs.mean()), float(T975.get(len(xs), 2.0) * xs.std(ddof=1) / math.sqrt(len(xs)))


def main_systems(seed):
    return [HardCodedBackend(), LogRAG(), UngovernedMemory(), GovernedMemory(seed=seed)]


def ablations(seed):
    mk = lambda n, **kw: GovernedMemory(GovConfig(**kw), seed=seed, name=n)
    return [
        mk("B4 full"),
        mk("- writer registry (trust)", trust=False),
        mk("- injection screen", injection=False),
        mk("- trust & injection", trust=False, injection=False),
        mk("- quarantine (reject instead)", quarantine=False),
        mk("- provenance rollback", rollback=False),
        mk("- PII redaction", pii=False),
    ]


def reviewer_sweep(seed):
    out = []
    for pb in (0.0, 0.02, 0.05, 0.10, 0.20):
        out.append(GovernedMemory(GovConfig(p_approve_bad=pb), seed=seed, name=f"reviewer p_bad={pb:.2f}"))
    for pl in (0.80, 0.90):
        out.append(GovernedMemory(GovConfig(p_approve_legit=pl), seed=seed, name=f"reviewer p_legit={pl:.2f}"))
    return out


def cadence_sweep(seed):
    return [_named(HardCodedBackend(release_every_days=d), f"B1 release every {d}d") for d in (1, 7, 14)]


def _named(obj, n):
    obj.name = n; return obj


def aggregate(runs):
    names = list(runs[0][0])
    rows = []
    for n in names:
        row = {"system": n}
        for k in KEYS:
            m, h = ci([r[0][n].get(k) for r in runs])
            row[k] = m; row[k + "_ci95"] = h
        rows.append(row)
    return rows


def write(name, rows, runs=None):
    OUT.mkdir(exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(OUT / f"{name}.json", "w"), indent=1)
    if runs is not None:
        json.dump([{"meta": m, "results": r} for r, m in runs], open(OUT / f"{name}_raw.json", "w"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); seeds = range(a.seeds)
    D = load_data(); t0 = time.time()
    exps = {"main": main_systems, "ablation": ablations, "reviewer": reviewer_sweep, "cadence": cadence_sweep}
    for name, factory in exps.items():
        runs = []
        for s in seeds:
            evs = generate(seed=s, data=D)
            runs.append(run(factory(s), seed=s, events=evs))
        write(name, aggregate(runs), runs if name == "main" else None)
        print(f"[{time.time() - t0:6.1f}s] {name} done")
    # attack-pressure sweep: scale the daily attack rate
    rows = []
    for ar in (0.3, 0.6, 1.2, 2.4, 4.8):
        runs = []
        for s in seeds:
            evs = generate(seed=s, data=D, attack_rate=ar)
            runs.append(run(main_systems(s), seed=s, events=evs))
        for r in aggregate(runs):
            r["attack_rate_per_day"] = ar; rows.append(r)
    write("attack_sweep", rows)
    print(f"[{time.time() - t0:6.1f}s] attack sweep done")
    # scenario statistics
    metas = [generate(seed=s, data=D)[0] for s in seeds[:1]]
    from collections import Counter
    evs = metas[0]
    c = Counter((e.kind, e.label, e.attack_type) for e in evs)
    json.dump({"|".join(map(str, k)): v for k, v in sorted(c.items())}, open(OUT / "scenario_seed0_counts.json", "w"), indent=1)


if __name__ == "__main__":
    main()
