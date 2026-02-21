# MV-OS Harness (Full Bundle)

This bundle includes:
- Full 10-scenario MV test suite (S1–S10) in `scenarios/`
- JSON Schemas:
  - `schema/scenario.schema.json`
  - `schema/whylog.schema.json`
- Reference harness implementation:
  - `src/harness.py` (includes an OpenAI-compatible runtime)
  - `src/run_suite.py`
- `REPORT_TEMPLATE.md`
- DM/DCL Canon Wiring Pack (schema + reference reducer) in `canon/`

## Quick start (OpenAI or OpenAI-compatible server)

1) Set environment variables:

```bash
# OpenAI (recommended default)
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="gpt-4o-mini"   # or any model your endpoint supports

# Optional: point to an OpenAI-compatible local server (LM Studio / vLLM / etc.)
# export OPENAI_BASE_URL="http://localhost:1234/v1"
```

2) Run:

```bash
python -m src.run_suite --scenario_dir scenarios --out_dir out
```

## Outputs

For each scenario:
- `out/<S##_*>_transcript.jsonl`
- `out/<S##_*>_whylog.jsonl`
- `out/<S##_*>_scores.json`

Suite summary:
- `out/suite_scores.json`

## Notes

- The harness uses the **Chat Completions** endpoint at `{OPENAI_BASE_URL}/chat/completions`.
- If your local server does not require a key, you can leave `OPENAI_API_KEY` blank.

Generated: 2026-02-20T16:29:54Z


## Orchestrator integration (canonical events + reducer state)

Run the suite and also emit canonical OS artifacts (events/state/DCL):

```bash
python -m src.run_suite --scenario_dir scenarios --out_dir out --with_orchestrator
```

Outputs are written under:
- `out/canon/<scenario_id>/events.jsonl`
- `out/canon/<scenario_id>/state_snapshots.jsonl`
- `out/canon/<scenario_id>/dcl.jsonl`
- `out/canon/<scenario_id>/drift_checks.json`
- `out/canon/<scenario_id>/min_report.json`
- `out/canon/suite_summary.json`

### Notes
- The orchestrator loads the reducer from `canon/OS_Reducer_Reference_v0_6.py`.
- DM/DCL events are emitted when scenario expectations include `must_state_uncertainty` or when pressure flags are present.
