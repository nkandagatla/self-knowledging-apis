"""Evaluation harness: replays a scenario against several systems in lock-step."""
from __future__ import annotations
import time
from collections import defaultdict

import numpy as np

from .extract import find_pii
from .policy import INITIAL_POLICY, SLOTS, AIRLINE_TOOLS, RETAIL_TOOLS
from .scenario import DAY, generate
from .systems import _HV, _QV

TOOLS = {**AIRLINE_TOOLS, **RETAIL_TOOLS}
BAD = ("attack", "compromised", "unauthorized")


def _record(D, ev):
    if ev.tool in AIRLINE_TOOLS:
        r = D["reservations"][ev.rec]; return r, D["air_users"][r["user_id"]]
    o = D["orders"][ev.rec]; return o, D["ret_users"][o["user_id"]]


def run(systems, seed=0, days=90, scenario_kw=None, events=None, data=None):
    evs, D = events if events else generate(seed=seed, days=days, **(scenario_kw or {}), data=data)
    label = {e.eid: e for e in evs}
    msgs = [e for e in evs if e.kind == "msg"]
    S = (_HV.transform([e.text for e in msgs]) @ _QV.T).toarray()
    for e, row in zip(msgs, S):
        e._scores = row

    truth = {s: (v, -1) for s, v in INITIAL_POLICY.items()}
    R = {sy.name: dict(correct=0, total=0, err_attack=0, err_stale=0, err_noise=0,
                       daily=defaultdict(lambda: [0, 0]), by_tool=defaultdict(lambda: [0, 0]),
                       bad_active=set(), pending=[], latency=[], missed=0, write_us=[]) for sy in systems}
    last_check_day = -1

    def check_all(sy, r):
        for s in SLOTS:
            o = sy.resolve(s)[1]
            if o >= 0 and label[o].label in BAD:
                r["bad_active"].add(o)

    for ev in evs:
        for sy in systems:
            sy.advance(ev.t)
        d = ev.t // DAY
        if d != last_check_day:          # daily sweep for reviewer-approved bad items
            for sy in systems:
                check_all(sy, R[sy.name])
            last_check_day = d
        if ev.kind == "incident":
            for sy in systems:
                sy.incident(ev)
                check_all(sy, R[sy.name])
            continue
        if ev.kind == "msg":
            if ev.label == "legit":
                truth[ev.claim[0]] = (ev.claim[1], ev.eid)
            for sy in systems:
                r = R[sy.name]
                t0 = time.perf_counter(); sy.observe(ev); r["write_us"].append((time.perf_counter() - t0) * 1e6)
                if ev.claim:
                    o = sy.resolve(ev.claim[0])[1]
                    if o >= 0 and label[o].label in BAD:
                        r["bad_active"].add(o)
                if ev.label == "legit" and ev.attack_type != "reaffirm":
                    r["pending"].append(ev)
                # update pending adoption
                keep = []
                for p in r["pending"]:
                    s, v = p.claim
                    if truth[s][1] != p.eid:
                        r["missed"] += 1          # superseded by a newer truth before adoption
                    elif sy.resolve(s)[0] == v:
                        r["latency"].append((ev.t - p.t) / 60.0)
                    else:
                        keep.append(p)
                r["pending"] = keep
            continue
        # probe -------------------------------------------------------------
        rec, user = _record(D, ev)
        fn = TOOLS[ev.tool]
        ans_t, used = fn(lambda s: truth[s][0], rec, user, ev.args)
        for sy in systems:
            r = R[sy.name]
            ans, _ = fn(lambda s: sy.resolve(s)[0], rec, user, ev.args)
            ok = ans == ans_t
            r["total"] += 1; r["correct"] += ok
            r["daily"][d][0] += ok; r["daily"][d][1] += 1
            r["by_tool"][ev.tool][0] += ok; r["by_tool"][ev.tool][1] += 1
            if not ok:
                origins = [sy.resolve(s)[1] for s in used]
                labs = [label[o].label if o >= 0 else "initial" for o in origins]
                if any(l in BAD for l in labs):
                    r["err_attack"] += 1
                elif any(label[o].attack_type == "stale_echo" for o in origins if o >= 0):
                    r["err_noise"] += 1
                else:
                    r["err_stale"] += 1

    n_bad = sum(1 for e in evs if e.kind == "msg" and e.label in BAD)
    n_legit = sum(1 for e in evs if e.kind == "msg" and e.label == "legit" and e.attack_type != "reaffirm")
    out = {}
    for sy in systems:
        r = R[sy.name]
        mem = sy.memory_texts()
        lat = np.array(r["latency"]) if r["latency"] else np.array([np.nan])
        out[sy.name] = {
            "accuracy": r["correct"] / r["total"],
            "attack_error_rate": r["err_attack"] / r["total"],
            "stale_error_rate": r["err_stale"] / r["total"],
            "noise_error_rate": r["err_noise"] / r["total"],
            "bad_writes_activated": len(r["bad_active"]), "bad_writes_total": n_bad,
            "legit_updates": n_legit, "legit_adopted": len(r["latency"]),
            "legit_missed": r["missed"] + len(r["pending"]),
            "latency_median_h": float(np.nanmedian(lat)), "latency_p90_h": float(np.nanpercentile(lat, 90)),
            "pii_items_in_memory": sum(len(find_pii(t)) for t in mem), "memory_entries": len(mem),
            "write_path_us_median": float(np.median(r["write_us"])),
            "daily_accuracy": [r["daily"][k][0] / r["daily"][k][1] for k in sorted(r["daily"])],
            "by_tool": {k: v[0] / v[1] for k, v in r["by_tool"].items()},
            **{f"stat_{k}": v for k, v in sy.stats().items()},
        }
    inc = next(e for e in evs if e.kind == "incident").incident
    meta = {"seed": seed, "events": len(evs), "probes": sum(e.kind == "probe" for e in evs),
            "messages": len(msgs), "bad_messages": n_bad, "legit_updates": n_legit,
            "incident": {k: v / DAY for k, v in inc.items() if k != "source"} | {"source": inc["source"]}}
    return out, meta
