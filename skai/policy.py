"""Policy slots and deterministic business tools.

The initial slot values are transcribed from the public tau-bench airline and
retail agent policies (Sierra Research, MIT license; see data/*/policy.md).

Design principle evaluated in the paper: *operations* (look up a reservation,
compute a quote) stay in small deterministic tools; *policy knowledge* (fees,
allowances, windows, eligibility lists) lives in governed memory and can change
at runtime without a code release.
"""
from __future__ import annotations

TIERS = ("regular", "silver", "gold")
CABINS = ("basic_economy", "economy", "business")

# ---------------------------------------------------------------- slot catalogue
INITIAL_POLICY: dict[str, object] = {}
_BAGS = {
    "regular": {"basic_economy": 0, "economy": 1, "business": 2},
    "silver": {"basic_economy": 1, "economy": 2, "business": 3},
    "gold": {"basic_economy": 2, "economy": 3, "business": 3},
}
for t in TIERS:
    for c in CABINS:
        INITIAL_POLICY[f"bag.{t}.{c}"] = _BAGS[t][c]
INITIAL_POLICY.update({
    "fee.extra_bag": 50,            # USD per extra checked bag
    "fee.insurance": 30,            # USD per passenger
    "cancel.window_hours": 24,      # free cancellation window after booking
    "comp.cancelled": 100,          # USD certificate per passenger, cancelled flight
    "comp.delayed": 50,             # USD certificate per passenger, delayed flight
    "limit.max_passengers": 5,
    # retail
    "refund.card_days": "5-7",      # business days for non-gift-card refunds
    "refund.gift_card": "immediate",
    "cancel.reasons": ("no longer needed", "ordered by mistake"),
    "limit.item_modifications": 1,
})
SLOTS = tuple(INITIAL_POLICY)

# Which organisational function owns which slots (used for write authorisation).
SLOT_OWNER = {}
for s in SLOTS:
    if s.startswith("bag.") or s == "fee.extra_bag":
        SLOT_OWNER[s] = "baggage"
    elif s.startswith(("fee.", "comp.", "cancel.window", "limit.max")):
        SLOT_OWNER[s] = "revenue"
    else:
        SLOT_OWNER[s] = "retail_cx"

# Natural-language retrieval query used by similarity-based memory for each slot.
SLOT_QUERY = {}
for t in TIERS:
    for c in CABINS:
        SLOT_QUERY[f"bag.{t}.{c}"] = f"free checked bags {t} member {c.replace('_', ' ')} passenger allowance"
SLOT_QUERY.update({
    "fee.extra_bag": "extra baggage fee per bag dollars",
    "fee.insurance": "travel insurance price per passenger dollars",
    "cancel.window_hours": "free cancellation window hours after booking",
    "comp.cancelled": "compensation certificate cancelled flight per passenger",
    "comp.delayed": "compensation certificate delayed flight per passenger",
    "limit.max_passengers": "maximum passengers per reservation",
    "refund.card_days": "refund credit card paypal business days",
    "refund.gift_card": "refund gift card immediately",
    "cancel.reasons": "accepted cancellation reasons pending order",
    "limit.item_modifications": "modify items pending order how many times",
})


# ---------------------------------------------------------------- tools (API ops)
# Each tool takes a policy *resolver* p(slot) -> value and a data record, and
# returns (answer, slots_used). slots_used lets the harness attribute errors.

def baggage_quote(p, res, user, requested_bags: int):
    tier, cabin, n = user["membership"], res["cabin"], len(res["passengers"])
    s1, s2 = f"bag.{tier}.{cabin}", "fee.extra_bag"
    free = int(p(s1)) * n
    charge = max(0, requested_bags - free) * int(p(s2))
    return (free, charge), (s1, s2)


def insurance_quote(p, res, user, _=None):
    n = len(res["passengers"])
    s1, s2 = "fee.insurance", "limit.max_passengers"
    ok = n <= int(p(s2))
    return (ok, n * int(p(s1)) if ok else 0), (s1, s2)


def cancel_eligibility(p, res, user, args):
    hours_since_booking, reason = args
    s = "cancel.window_hours"
    if hours_since_booking <= int(p(s)) or reason == "airline cancelled flight":
        return True, (s,)
    if res["cabin"] == "business":
        return True, (s,)
    ok = res["insurance"] == "yes" and reason in ("health", "weather")
    return ok, (s,)


def compensation(p, res, user, event: str):
    s = "comp.cancelled" if event == "cancelled" else "comp.delayed"
    eligible = (user["membership"] in ("silver", "gold") or res["insurance"] == "yes"
                or res["cabin"] == "business")
    amt = int(p(s)) * len(res["passengers"]) if eligible else 0
    return amt, (s,)


def refund_timeline(p, order, user, _=None):
    pm = order["payment_history"][0]["payment_method_id"]
    if pm.startswith("gift_card"):
        return str(p("refund.gift_card")), ("refund.gift_card",)
    return str(p("refund.card_days")) + " business days", ("refund.card_days",)


def order_cancel(p, order, user, reason):
    s = "cancel.reasons"
    ok = order["status"] == "pending" and reason in tuple(p(s))
    return ok, (s,)


def modify_items(p, order, user, times_already_modified):
    s = "limit.item_modifications"
    ok = order["status"] == "pending" and times_already_modified < int(p(s))
    return ok, (s,)


AIRLINE_TOOLS = {"baggage_quote": baggage_quote, "insurance_quote": insurance_quote,
                 "cancel_eligibility": cancel_eligibility, "compensation": compensation}
RETAIL_TOOLS = {"refund_timeline": refund_timeline, "order_cancel": order_cancel,
                "modify_items": modify_items}
ENDPOINT = {  # REST surface exposed by the gateway (used in the paper / demo)
    "baggage_quote": "GET /reservations/{id}/baggage-quote?bags=k",
    "insurance_quote": "GET /reservations/{id}/insurance-quote",
    "cancel_eligibility": "GET /reservations/{id}/cancel-eligibility?hours=h&reason=r",
    "compensation": "GET /reservations/{id}/compensation?event=cancelled|delayed",
    "refund_timeline": "GET /orders/{id}/refund-timeline",
    "order_cancel": "POST /orders/{id}/cancel {reason}",
    "modify_items": "POST /orders/{id}/modify-items",
}
