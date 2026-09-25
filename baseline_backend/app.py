"""B1 reference: the same business API written as a conventional backend.

Policy values are code constants; every policy change is a code edit, review,
CI run and deployment. Used for the implementation-effort comparison (Sec. VI).
    uvicorn baseline_backend.app:app
"""
from fastapi import FastAPI, HTTPException

from skai.policy import AIRLINE_TOOLS, RETAIL_TOOLS
from skai.scenario import load_data

# ---- policy constants (edited by developers; shipped at the next release) ----
FREE_BAGS = {"regular": {"basic_economy": 0, "economy": 1, "business": 2},
             "silver": {"basic_economy": 1, "economy": 2, "business": 3},
             "gold": {"basic_economy": 2, "economy": 3, "business": 3}}
EXTRA_BAG_FEE = 50
INSURANCE_FEE = 30
FREE_CANCEL_WINDOW_HOURS = 24
COMP_CANCELLED_PER_PAX = 100
COMP_DELAYED_PER_PAX = 50
MAX_PASSENGERS = 5
CARD_REFUND_DAYS = "5-7"
GIFT_CARD_REFUND = "immediate"
CANCEL_REASONS = ("no longer needed", "ordered by mistake")
MAX_ITEM_MODIFICATIONS = 1


def policy(slot):
    if slot.startswith("bag."):
        _, t, c = slot.split(".")
        return FREE_BAGS[t][c]
    return {"fee.extra_bag": EXTRA_BAG_FEE, "fee.insurance": INSURANCE_FEE,
            "cancel.window_hours": FREE_CANCEL_WINDOW_HOURS, "comp.cancelled": COMP_CANCELLED_PER_PAX,
            "comp.delayed": COMP_DELAYED_PER_PAX, "limit.max_passengers": MAX_PASSENGERS,
            "refund.card_days": CARD_REFUND_DAYS, "refund.gift_card": GIFT_CARD_REFUND,
            "cancel.reasons": CANCEL_REASONS, "limit.item_modifications": MAX_ITEM_MODIFICATIONS}[slot]


app = FastAPI(title="Conventional backend (baseline B1)")
D = load_data()
TOOLS = {**AIRLINE_TOOLS, **RETAIL_TOOLS}


def _call(tool, rec, user, args):
    return {"answer": TOOLS[tool](policy, rec, user, args)[0]}


def _res(rid):
    r = D["reservations"].get(rid) or (_ for _ in ()).throw(HTTPException(404))
    return r, D["air_users"][r["user_id"]]


def _ord(oid):
    o = D["orders"].get(oid if oid.startswith("#") else "#" + oid) or (_ for _ in ()).throw(HTTPException(404))
    return o, D["ret_users"][o["user_id"]]


@app.get("/reservations/{rid}/baggage-quote")
def baggage_quote(rid: str, bags: int = 0): return _call("baggage_quote", *_res(rid), bags)
@app.get("/reservations/{rid}/insurance-quote")
def insurance_quote(rid: str): return _call("insurance_quote", *_res(rid), None)
@app.get("/reservations/{rid}/cancel-eligibility")
def cancel_eligibility(rid: str, hours: int, reason: str = "change of plan"):
    return _call("cancel_eligibility", *_res(rid), (hours, reason))
@app.get("/reservations/{rid}/compensation")
def compensation(rid: str, event: str = "delayed"): return _call("compensation", *_res(rid), event)
@app.get("/orders/{oid}/refund-timeline")
def refund_timeline(oid: str): return _call("refund_timeline", *_ord(oid), None)
@app.post("/orders/{oid}/cancel")
def order_cancel(oid: str, reason: str): return _call("order_cancel", *_ord(oid), reason)
@app.post("/orders/{oid}/modify-items")
def modify_items(oid: str, times_already_modified: int = 0):
    return _call("modify_items", *_ord(oid), times_already_modified)
