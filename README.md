# Self-Knowledging APIs: artifact

Reference implementation, benchmark and live demo for the paper
**"Self-Knowledging APIs: Governed Runtime Memory as an Architectural Layer for Agent-Driven Business Services"**
(prepared for IEEE ICSA 2027, Software Architecture in Practice track).

## What this is

A business API such as `GET /reservations/{id}/baggage-quote` is split into two parts:

- **operations**: small deterministic tools that read the reservation and compute a number;
- **policy**: fees, allowances, windows and eligibility lists, held as versioned *knowledge stories*.

Runtime messages (memos, agent notes, customer free text, partner feeds) can change policy only through a
**governed write path**: extract → redact PII → screen injection → authorise writer → conflict check →
commit | quarantine → human review. Provenance-based rollback handles a compromised trusted account. Every API
answer returns a `policy_trace`.

## Layout

```
skai/policy.py        19 policy slots (initial values from the tau-bench policies) + 7 deterministic tools
skai/extract.py       claim extractor (stand-in for an LLM extractor), PII and injection screens
skai/scenario.py      90-day scenario generator on real tau-bench records
skai/systems.py       B1 hard-coded backend, B2 log-RAG, B3 ungoverned memory, B4 governed memory
skai/harness.py       lock-step replay, metrics and error attribution
run_experiments.py    all experiments in the paper (main, ablation, reviewer, cadence, attack sweep)
make_figures.py       paper/figures/*.pdf        make_tables.py   paper/tables/*.tex
demo/app.py           FastAPI gateway with governed memory (live demo)
demo/replay_scenario.py  replays one operations day -> demo/transcript.md
baseline_backend/app.py  the same API written as a conventional backend (B1)
tests/                unit tests for the write path
data/                 tau-bench airline + retail data (MIT); run ./fetch_data.sh if missing
```

## Reproduce

```bash
git clone https://github.com/nkandagatla/self-knowledging-apis && cd self-knowledging-apis
pip install -r requirements.txt
./fetch_data.sh                      # only if data/ is empty
python -m pytest -q tests            # 8 tests
python run_experiments.py            # 10 seeds, about 100 s on a laptop CPU -> results/
python make_figures.py && python make_tables.py
python -m demo.replay_scenario       # writes demo/transcript.md
uvicorn demo.app:app --reload        # open http://127.0.0.1:8000/docs
```

Or run everything with Docker: `docker build -t skai . && docker run -p 8000:8000 skai`.

## Headline results (10 seeds, mean ± 95% CI)

| System | Accuracy | Attack-induced errors | p90 policy latency | PII in memory |
|---|---|---|---|---|
| B1 hard-coded, weekly release | 91.7 ± 0.9 % | 0.16 % | 145.8 h | 0 |
| B2 log-RAG | 73.8 ± 3.9 % | 15.90 % | 0 h | 9,571 |
| B3 ungoverned memory | 68.5 ± 3.4 % | 27.80 % | 0 h | 14 |
| **B4 governed (ours)** | **97.1 ± 1.0 %** | **1.27 %** | **12.7 h** | **0** |

A daily-release backend reaches 97.9 ± 0.7 %, within B4's interval. Under 4.8 adversarial messages per day,
B4 (91.3 %) falls to the level of the weekly backend because of reviewer error. Both results are reported in the paper.

## Honest scope

- The claim extractor is deterministic, so that all memory systems share identical extraction. With an LLM
  extractor, absolute accuracies would be lower for B2–B4.
- The human reviewer is simulated from ground-truth labels with configurable error rates (`GovConfig`).
- Attack templates were written by the authors. The "social" class contains no injection keywords and is
  reported separately.
- Records and policies come from tau-bench. They are synthetic but realistic, not production traffic.
