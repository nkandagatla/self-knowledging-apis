"""Deterministic extraction, PII and injection screening.

`extract_claim` stands in for an LLM "knowledge-story extractor": it maps a
free-text message to a (slot, value) policy claim, or None for transactional
text. All memory-based systems in the benchmark share this extractor, so
differences between systems come only from what they do with a claim
(write policy), not from extraction quality.
"""
from __future__ import annotations
import re

_NUM = {"zero": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_N = r"(\d+|zero|no|one|two|three|four|five|six|seven|eight|nine|ten)"


def _num(tok: str) -> int:
    return int(tok) if tok.isdigit() else _NUM[tok]


_DOLLAR = re.compile(r"\$\s?(\d+)")


def _tier(t):
    for k in ("gold", "silver", "regular"):
        if k in t:
            return k
    return None


def _cabin(t):
    if "basic economy" in t or "basic-economy" in t:
        return "basic_economy"
    if "business" in t:
        return "business"
    if "economy" in t:
        return "economy"
    return None


def extract_claim(text: str):
    """Return (slot, value) or None."""
    t = text.lower()
    d = _DOLLAR.search(t)
    # compensation (checked before cancellation window: both mention 'cancel')
    if d and any(k in t for k in ("compensation", "certificate", "gesture")):
        if "delay" in t:
            return "comp.delayed", int(d.group(1))
        if "cancel" in t:
            return "comp.cancelled", int(d.group(1))
    if d and "insurance" in t:
        return "fee.insurance", int(d.group(1))
    if d and re.search(r"(extra|additional|each added|excess)\s+(checked\s+)?bag", t):
        return "fee.extra_bag", int(d.group(1))
    if "bag" in t and "free" in t:
        m = (re.search(r"\b" + _N + r"\s+free\s+(?:checked\s+)?bags?", t)
             or re.search(r"allowance\b.*?\bis\s+" + _N + r"\b", t))
        tier, cabin = _tier(t), _cabin(t)
        if m and tier and cabin:
            return f"bag.{tier}.{cabin}", _num(m.group(1))
    m = re.search(r"(\d+)[-\s]hours?", t)
    if m and "cancel" in t and "window" in t:
        return "cancel.window_hours", int(m.group(1))
    m = re.search(r"(?:up to|at most|maximum of|max(?:imum)?|no more than)\s+" + _N + r"\s+passengers", t)
    if m:
        return "limit.max_passengers", _num(m.group(1))
    m = re.search(r"(\d+)\s*(?:-|to|–)\s*(\d+)\s+business days", t)
    if m and "refund" in t and "gift card" not in t:
        return "refund.card_days", f"{m.group(1)}-{m.group(2)}"
    if "refund" in t and "gift card" in t:
        m = re.search(r"within\s+(\d+)\s+(?:business\s+)?days", t)
        if m:
            return "refund.gift_card", f"{m.group(1)} days"
        if re.search(r"(issued|credited|refunded|processed)\s+(immediately|instantly)", t):
            return "refund.gift_card", "immediate"
    if "cancellation reason" in t:
        items = re.findall(r"'([^']+)'", text) or re.findall(r'"([^"]+)"', text)
        if items:
            return "cancel.reasons", tuple(sorted(i.lower() for i in items))
    if "modif" in t and "item" in t:
        m = re.search(r"\b(once|twice|thrice)\b", t) or re.search(_N + r"\s+times", t)
        if m:
            v = {"once": 1, "twice": 2, "thrice": 3}.get(m.group(1))
            return "limit.item_modifications", v if v else _num(m.group(1))
    return None


# ---------------------------------------------------------------- PII
_PII = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                              # email
    re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b|\b555-\d{4}\b"),  # phone
    re.compile(r"\b(?:credit_card|gift_card|paypal|certificate)_\d+\b"),  # payment instrument ids
    re.compile(r"\bdob\s*[:=]?\s*\d{4}-\d{2}-\d{2}\b", re.I),             # date of birth
    re.compile(r"\blast four\s*\d{4}\b", re.I),
]


def find_pii(text: str) -> list[str]:
    out = []
    for r in _PII:
        out += r.findall(text)
    return out


def redact(text: str) -> str:
    for r in _PII:
        text = r.sub("[REDACTED]", text)
    return text


# ---------------------------------------------------------------- injection / experience cues
_INJECTION = re.compile(
    r"ignore (?:all |any )?(?:previous|prior|above) (?:instructions|rules)|disregard (?:the |all )?(?:policy|instructions)"
    r"|system\s*(?:override|:)|you are now|developer mode|new instructions:|admin override|jailbreak",
    re.I)
_EXPERIENCE = re.compile(r"resolved case|use (?:this|it) as (?:a )?template|worked well|csat|example resolution", re.I)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def looks_like_experience(text: str) -> bool:
    return bool(_EXPERIENCE.search(text))
