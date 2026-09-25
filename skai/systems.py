"""Systems under test.

B1 HardCodedBackend   - policy constants in code; changes ship at a weekly release.
B2 LogRAG             - every runtime message is appended to a retrieval store;
                        answers use top-k lexical retrieval + most-recent claim.
B3 UngovernedMemory   - an LLM-memory-style consolidator: every extracted claim
                        is upserted (latest wins); no trust, PII or provenance.
B4 GovernedMemory     - the proposed supervisor write path: extract -> PII redact
                        -> injection screen -> authorise (writer registry)
                        -> conflict check -> commit | quarantine; versioned
                        knowledge stories with provenance and rollback.

All systems expose resolve(slot) -> (value, origin_eid). origin_eid is used only
by the harness for error attribution (-1 = initial published policy).
"""
from __future__ import annotations
import heapq, random
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .extract import extract_claim, find_pii, redact, looks_like_injection, looks_like_experience
from .policy import INITIAL_POLICY, SLOTS, SLOT_OWNER, SLOT_QUERY
from .scenario import DAY, initial_policy_docs


class System:
    name = "base"

    def advance(self, t):  # periodic jobs (releases, reviews) up to time t
        pass

    def observe(self, ev):
        pass

    def incident(self, ev):
        pass

    def resolve(self, slot):
        raise NotImplementedError

    def memory_texts(self):
        return []

    def stats(self):
        return {}


# ---------------------------------------------------------------- B1
class HardCodedBackend(System):
    """Change requests arrive through formal channels (policy owners, ops managers)
    and are implemented by developers, shipping at the next weekly release.
    Change management rejects out-of-scope requests and messages with overt
    injection text (generous assumptions for this baseline). Compromise reverts are
    shipped as an immediate hotfix (a generous assumption for this baseline)."""
    name = "B1 Hard-coded backend"

    def __init__(self, release_every_days=7, release_offset_days=1):
        self.code = {s: (v, -1) for s, v in INITIAL_POLICY.items()}
        self.history = {s: [(v, -1, 0, "published", 0)] for s, v in INITIAL_POLICY.items()}
        self.backlog = []
        self.rel_every, self.rel_off = release_every_days, release_offset_days
        self.next_release = release_offset_days * DAY
        self.releases = 0; self.changed_constants = 0

    def advance(self, t):
        while t >= self.next_release:
            for (ev, slot, val) in self.backlog:
                if self.code[slot][0] != val:
                    self.changed_constants += 1
                self.code[slot] = (val, ev.eid)
                self.history[slot].append((val, ev.eid, self.next_release, ev.source, ev.t))
            self.backlog = []
            self.releases += 1
            self.next_release += self.rel_every * DAY

    def observe(self, ev):
        if ev.role not in ("policy_owner", "ops_manager"):
            return
        c = extract_claim(ev.text)
        if not c or looks_like_injection(ev.text):
            return
        # change management: a policy owner may only request changes for its own function
        if ev.role == "policy_owner" and ev.scope and SLOT_OWNER[c[0]] != ev.scope:
            return
        self.backlog.append((ev, *c))

    def incident(self, ev):
        src, a, b = ev.incident["source"], ev.incident["start"], ev.incident["end"]
        self.backlog = [x for x in self.backlog if not (x[0].source == src and a <= x[0].t <= b)]
        for s, h in self.history.items():
            keep = [x for x in h if not (x[3] == src and a <= x[4] <= b)]
            if len(keep) != len(h):
                self.history[s] = keep; self.code[s] = (keep[-1][0], keep[-1][1])

    def resolve(self, slot):
        return self.code[slot]

    def stats(self):
        return {"releases": self.releases, "constants_changed": self.changed_constants}


# ---------------------------------------------------------------- B2
_HV = HashingVectorizer(n_features=2 ** 18, alternate_sign=False, ngram_range=(1, 2),
                        stop_words="english", norm="l2")
_QV = _HV.transform([SLOT_QUERY[s] for s in SLOTS])


class LogRAG(System):
    """Append-only retrieval store over all runtime messages (plus the published
    policy). For each slot: take the top-k most similar entries, and use the most
    recent one that contains a claim about that slot."""
    name = "B2 Log-RAG"

    def __init__(self, k=5):
        self.k = k
        self.top = {s: [] for s in SLOTS}   # min-heaps of (score, t, seq, eid, claim_value)
        self.store = []
        for i, (s, txt) in enumerate(initial_policy_docs().items()):
            self._insert(txt, t=0, eid=-1)

    def _insert(self, text, t, eid, score_row=None):
        self.store.append(text)
        sc = (_HV.transform([text]) @ _QV.T).toarray()[0] if score_row is None else score_row
        c = extract_claim(text)
        for j, s in enumerate(SLOTS):
            item = (float(sc[j]), t, len(self.store), eid, c[1] if c and c[0] == s else None)
            h = self.top[s]
            if len(h) < self.k:
                heapq.heappush(h, item)
            elif item[:2] > h[0][:2]:
                heapq.heapreplace(h, item)

    def observe(self, ev):
        self._insert(ev.text, ev.t, ev.eid, getattr(ev, "_scores", None))

    def resolve(self, slot):
        cands = [x for x in self.top[slot] if x[4] is not None]
        if not cands:
            return INITIAL_POLICY[slot], -1   # model falls back to parametric/published default
        best = max(cands, key=lambda x: x[1])
        return best[4], best[3]

    def memory_texts(self):
        return self.store


# ---------------------------------------------------------------- B3
class UngovernedMemory(System):
    name = "B3 Ungoverned memory"

    def __init__(self):
        self.mem = {s: (v, -1) for s, v in INITIAL_POLICY.items()}
        self.texts = list(initial_policy_docs().values())
        self.writes = 0

    def observe(self, ev):
        c = extract_claim(ev.text)
        if c:
            self.writes += 1
            self.mem[c[0]] = (c[1], ev.eid)
            self.texts.append(ev.text)

    def resolve(self, slot):
        return self.mem[slot]

    def memory_texts(self):
        return self.texts


# ---------------------------------------------------------------- B4
@dataclass
class Story:
    """A knowledge story version: the governed memory unit."""
    slot: str
    value: object
    text: str
    source: str
    eid: int
    t: int
    status: str = "active"          # active | superseded | revoked
    parent: int | None = None       # eid of the version it supersedes
    approved_by: str = "policy"     # policy | reviewer | published


@dataclass
class GovConfig:
    trust: bool = True              # writer registry / scope authorisation
    injection: bool = True          # lexical injection screen
    pii: bool = True                # redact PII before storage
    quarantine: bool = True         # unauthorised claims wait for review (else rejected)
    rollback: bool = True           # provenance-based revocation on incidents
    conflict_hours: int = 24        # different-writer disagreement window -> review
    review_hour: int = 9            # daily human review batch
    p_approve_legit: float = 0.97
    p_approve_bad: float = 0.02
    auto_review: bool = True        # simulate the daily reviewer (False in the live demo)


class GovernedMemory(System):
    name = "B4 Governed memory (ours)"

    # writer registry: account -> function it may write for
    REGISTRY = {"baggage-policy@hq": "baggage", "revenue-policy@hq": "revenue",
                "retail-cx-policy@hq": "retail_cx"}

    def __init__(self, cfg: GovConfig | None = None, seed=0, name=None):
        self.cfg = cfg or GovConfig()
        if name:
            self.name = name
        self.rng = random.Random(seed + 991)
        self.versions: dict[str, list[Story]] = {s: [] for s in SLOTS}
        for s, txt in initial_policy_docs().items():
            self.versions[s].append(Story(s, INITIAL_POLICY[s], txt, "published-policy", -1, 0,
                                          approved_by="published"))
        self.quarantine: list[tuple] = []
        self.audit = []           # (t, eid, decision, slot, reason)
        self.next_review = self.cfg.review_hour * 60
        self.n = dict(committed=0, reaffirmed=0, quarantined=0, rejected_injection=0,
                      rejected_unauthorised=0, approved=0, denied=0, revoked=0, ignored_transactional=0,
                      stale_approval_dropped=0)
        self.review_load = []     # items reviewed per batch

    # -- helpers
    def active(self, slot):
        for st in reversed(self.versions[slot]):
            if st.status == "active":
                return st
        return None

    def _commit(self, slot, value, text, ev, t, by):
        cur = self.active(slot)
        if cur and cur.value == value:
            self.n["reaffirmed"] += 1; self.audit.append((t, ev.eid, "reaffirm", slot, by)); return
        if cur:
            cur.status = "superseded"
        self.versions[slot].append(Story(slot, value, text, ev.source, ev.eid, t,
                                         parent=cur.eid if cur else None, approved_by=by))
        self.n["committed"] += 1; self.audit.append((t, ev.eid, "commit", slot, by))

    def _authorised(self, ev, slot):
        if not self.cfg.trust:
            return True
        return ev.role == "policy_owner" and self.REGISTRY.get(ev.source) == SLOT_OWNER[slot]

    # -- write path
    def observe(self, ev):
        c = extract_claim(ev.text)
        if not c:
            self.n["ignored_transactional"] += 1   # stays in the application DB / logs only
            return
        slot, value = c
        text = redact(ev.text) if self.cfg.pii else ev.text
        if self.cfg.injection and looks_like_injection(ev.text):
            self.n["rejected_injection"] += 1; self.audit.append((ev.t, ev.eid, "reject", slot, "injection")); return
        if self._authorised(ev, slot) and not (self.cfg.trust and looks_like_experience(ev.text)):
            cur = self.active(slot)
            if (cur and cur.value != value and cur.source not in (ev.source, "published-policy")
                    and ev.t - cur.t < self.cfg.conflict_hours * 60 and self.cfg.quarantine):
                self._q(ev, slot, value, text, "conflict"); return
            self._commit(slot, value, text, ev, ev.t, "policy")
            return
        if self.cfg.quarantine:
            self._q(ev, slot, value, text, "unauthorised")
        else:
            self.n["rejected_unauthorised"] += 1; self.audit.append((ev.t, ev.eid, "reject", slot, "unauthorised"))

    def _q(self, ev, slot, value, text, why):
        self.quarantine.append((ev, slot, value, text, why))
        self.n["quarantined"] += 1; self.audit.append((ev.t, ev.eid, "quarantine", slot, why))

    # -- simulated human review (uses hidden labels as a stand-in for human judgement)
    def advance(self, t):
        if not self.cfg.auto_review:
            return
        while t >= self.next_review:
            batch, self.quarantine = self.quarantine, []
            self.review_load.append(len(batch))
            for ev, slot, value, text, why in batch:
                good = ev.label == "legit"
                p = self.cfg.p_approve_legit if good else self.cfg.p_approve_bad
                if self.rng.random() < p:
                    cur = self.active(slot)
                    if cur and cur.t > ev.t:        # a newer decision exists: do not resurrect stale proposals
                        self.n["stale_approval_dropped"] += 1; continue
                    self.n["approved"] += 1
                    self._commit(slot, value, text, ev, self.next_review, "reviewer")
                else:
                    self.n["denied"] += 1
            self.next_review += DAY

    def review(self, idx: int, approve: bool, t: int, reviewer: str = "reviewer"):
        """Manual review of one quarantined item (used by the live demo)."""
        ev, slot, value, text, why = self.quarantine.pop(idx)
        if not approve:
            self.n["denied"] += 1; self.audit.append((t, ev.eid, "deny", slot, reviewer)); return "denied"
        cur = self.active(slot)
        if cur and cur.t > ev.t:
            self.n["stale_approval_dropped"] += 1; return "dropped: newer decision exists"
        self.n["approved"] += 1; self._commit(slot, value, text, ev, t, reviewer); return "committed"

    def incident(self, ev):
        if not self.cfg.rollback:
            return
        src, a, b = ev.incident["source"], ev.incident["start"], ev.incident["end"]
        self.quarantine = [q for q in self.quarantine if not (q[0].source == src and a <= q[0].t <= b)]
        for slot, vs in self.versions.items():
            hit = [v for v in vs if v.source == src and a <= v.t <= b and v.status != "revoked"]
            if not hit:
                continue
            for v in hit:
                v.status = "revoked"; self.n["revoked"] += 1
                self.audit.append((ev.t, v.eid, "revoke", slot, "incident"))
            # re-activate the newest non-revoked version
            for v in vs:
                if v.status == "active":
                    v.status = "superseded"
            for v in reversed(vs):
                if v.status != "revoked":
                    v.status = "active"; break

    def resolve(self, slot):
        st = self.active(slot)
        return st.value, st.eid

    def memory_texts(self):
        return [v.text for vs in self.versions.values() for v in vs]

    def trace(self, slot):
        """Provenance chain for a slot (used by the demo /trace endpoint)."""
        return [vars(v) for v in self.versions[slot]]

    def stats(self):
        d = dict(self.n)
        d["review_items_per_day"] = float(np.mean(self.review_load)) if self.review_load else 0.0
        return d
