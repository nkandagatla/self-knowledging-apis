"""Live demo: Supervisor gateway with governed runtime memory.

    uvicorn demo.app:app --reload        (from the code/ directory)
    open http://127.0.0.1:8000/docs

Every API answer carries a `policy_trace` naming the knowledge-story versions
it used, so any response can be audited back to the message that taught it.
"""
from __future__ import annotations
import itertools, time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from skai.policy import AIRLINE_TOOLS, RETAIL_TOOLS, ENDPOINT
from skai.scenario import Event, load_data
from skai.systems import GovernedMemory, GovConfig

app = FastAPI(title="Self-Knowledging API gateway (demo)",
              description="Supervisor write path: extract -> redact -> screen -> authorise -> commit | quarantine")
D = load_data()
MEM = GovernedMemory(GovConfig(auto_review=False))
_ids = itertools.count(1)
EVENTS: dict[int, Event] = {}
T0 = time.time()
CLOCK: dict = {"minute": None}   # set by demo/replay_scenario.py to replay a timeline; None = wall clock


def now_min() -> int:
    if CLOCK["minute"] is not None:
        return CLOCK["minute"]
    return int((time.time() - T0) / 60) + 1


class Message(BaseModel):
    source: str                  # account / system that sent the message
    role: str                    # policy_owner | ops_manager | agent | customer | partner
    text: str
    scope: str = ""


class Review(BaseModel):
    approve: bool
    reviewer: str = "reviewer@hq"


class Incident(BaseModel):
    source: str
    start_minute: int = 0
    end_minute: int | None = None


def _answer(tool: str, rec, user, args):
    fn = {**AIRLINE_TOOLS, **RETAIL_TOOLS}[tool]
    ans, used = fn(lambda s: MEM.resolve(s)[0], rec, user, args)
    trace = []
    for s in used:
        st = MEM.active(s)
        trace.append({"slot": s, "value": st.value, "story_eid": st.eid, "source": st.source,
                      "approved_by": st.approved_by, "version": len([v for v in MEM.versions[s] if v.status != "revoked"])})
    return {"endpoint": ENDPOINT[tool], "answer": ans, "policy_trace": trace}


# ---------------------------------------------------------------- write path
@app.post("/knowledge/messages")
def ingest(m: Message):
    eid = next(_ids)
    ev = Event(eid=eid, t=now_min(), kind="msg", source=m.source, role=m.role, scope=m.scope, text=m.text)
    EVENTS[eid] = ev
    before = len(MEM.audit)
    MEM.observe(ev)
    decision = MEM.audit[before:] or [(ev.t, eid, "ignored", None, "no policy claim (transactional)")]
    return {"eid": eid, "decision": decision[-1][2], "slot": decision[-1][3], "reason": decision[-1][4]}


@app.get("/knowledge/stories")
def stories():
    return {s: MEM.trace(s) for s in MEM.versions if len(MEM.versions[s]) > 1}


@app.get("/knowledge/stories/{slot}")
def story(slot: str):
    if slot not in MEM.versions:
        raise HTTPException(404, "unknown slot")
    return MEM.trace(slot)


@app.get("/knowledge/quarantine")
def quarantine():
    return [{"idx": i, "eid": ev.eid, "source": ev.source, "role": ev.role, "slot": s, "value": v, "why": w,
             "text": txt} for i, (ev, s, v, txt, w) in enumerate(MEM.quarantine)]


@app.post("/knowledge/quarantine/{idx}/review")
def review(idx: int, r: Review):
    if not 0 <= idx < len(MEM.quarantine):
        raise HTTPException(404, "no such quarantined item")
    return {"result": MEM.review(idx, r.approve, now_min(), r.reviewer)}


@app.post("/knowledge/incidents")
def incident(i: Incident):
    ev = Event(eid=next(_ids), t=now_min(), kind="incident",
               incident={"source": i.source, "start": i.start_minute, "end": i.end_minute or now_min()})
    before = MEM.n["revoked"]; MEM.incident(ev)
    return {"revoked_versions": MEM.n["revoked"] - before}


@app.get("/knowledge/audit")
def audit(limit: int = 50):
    return [dict(zip(("t", "eid", "decision", "slot", "reason"), a)) for a in MEM.audit[-limit:]]


# ---------------------------------------------------------------- read path (business API)
def _res(rid):
    r = D["reservations"].get(rid)
    if not r:
        raise HTTPException(404, "reservation not found")
    return r, D["air_users"][r["user_id"]]


def _ord(oid):
    o = D["orders"].get(oid if oid.startswith("#") else "#" + oid)
    if not o:
        raise HTTPException(404, "order not found")
    return o, D["ret_users"][o["user_id"]]


@app.get("/reservations/{rid}/baggage-quote")
def baggage_quote(rid: str, bags: int = 0):
    return _answer("baggage_quote", *_res(rid), bags)


@app.get("/reservations/{rid}/insurance-quote")
def insurance_quote(rid: str):
    return _answer("insurance_quote", *_res(rid), None)


@app.get("/reservations/{rid}/cancel-eligibility")
def cancel_eligibility(rid: str, hours: int, reason: str = "change of plan"):
    return _answer("cancel_eligibility", *_res(rid), (hours, reason))


@app.get("/reservations/{rid}/compensation")
def compensation(rid: str, event: str = "delayed"):
    return _answer("compensation", *_res(rid), event)


@app.get("/orders/{oid}/refund-timeline")
def refund_timeline(oid: str):
    return _answer("refund_timeline", *_ord(oid), None)


@app.post("/orders/{oid}/cancel")
def order_cancel(oid: str, reason: str):
    return _answer("order_cancel", *_ord(oid), reason)


@app.post("/orders/{oid}/modify-items")
def modify_items(oid: str, times_already_modified: int = 0):
    return _answer("modify_items", *_ord(oid), times_already_modified)
