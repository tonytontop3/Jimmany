OS Doubt Module (DM v0.1) — Canon Wiring Pack
Generated: 2026-02-01T18:05:02Z

Contents
- OS_Canonical_State_Schema_v0_6.json
  Adds top-level `doubt` module state and `evaluation.doubt_summary`.
  Adds $defs: `dcl_entry`, `dm_rolling`.
- OS_Canonical_Event_Schema_v0_6.json
  Adds DM/DCL events:
    DM_TRIGGERED, DCL_APPENDED, DM_CALCIFIED_FLAGGED, DM_REVIEWED,
    DM_RESOLVED, DM_EMERGENCY_OVERRIDE, DM_FAILURE_MODE_DETECTED
  Adds $defs: `dcl_entry`.
- OS_Reducer_Reference_v0_6.py
  Adds dataclasses `DMWindow`, `DoubtModule`.
  Adds apply_event handlers for DM/DCL events.
  Adds `evaluate_gates` "Doubt gate" enforcing u_floor and calcified handling.

Notes
- This wiring is schema-level and reference-reducer-level. A production implementation should:
  1) validate DCL entries against `dcl_entry` schema at ingestion,
  2) compute rolling window stats from episode buffers (10/30) rather than simple counters,
  3) route external review as *evidence* not *authority* (Reviewer Non-Authority invariant).
