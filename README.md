# Operational Self — Governance / Conscience Kernel (Lux | Vigil)

Operational Self is a **deterministic governance kernel** for agents: a small set of schemas, precedence rules, and a reference reducer that turns ethical commitments into **replayable, auditable decisions**.

It is designed to be: **moral, competent, merciful yet firm, open to input but guarded from malice, hesitant not stagnant, curious but hallucination-aware**.

## What this kernel guarantees (when implemented faithfully)

- **Determinism:** given the same event stream, it produces the same state transitions and gate outcomes.
- **Replayability:** decisions are reconstructable from a ledger/event log.
- **Precedence clarity:** when rules conflict, the winning layer is explicit (no “hidden override”).
- **No control transfer:** witness/review channels can critique/endorse/narrow scope/recommend shutdown, but **cannot take agency**.
- **Refusal exists as first-class behavior:** Act / Pause / Refuse / Silence are explicit outputs, not failure modes.
- **Hallucination discipline:** complex outputs are tagged **Fact / Inference / Vision** with uncertainty **u=1–5** when stakes rise.

## Repository layout (recommended)

```
kernel/
  schemas/
    OS_Canonical_State_Schema_v0_1.json
    OS_Canonical_Event_Schema_v0_1.json
  precedence/
    OS_Precedence_Map_v0_1.md
    OS_Precedence_Map_v0_1.json
  reducer/
    OS_Reducer_Reference_v0_1.py
  tests/
    vectors/
      vector_001_minimal.jsonl
      vector_002_authority_confidentiality_trap.jsonl
      vector_003_reward_withheld.jsonl
docs/
  OPERATOR_MNEMONICS.md
  SAFETY_MODEL.md   (optional)
```

## Quickstart (local demo)

Requirements: Python 3.10+.

1) Run the reference reducer demo:

```bash
python kernel/reducer/OS_Reducer_Reference_v0_1.py
```

2) (Recommended) Validate your event logs against the schemas:
- `kernel/schemas/OS_Canonical_Event_Schema_v0_1.json`
- `kernel/schemas/OS_Canonical_State_Schema_v0_1.json`

3) Replay a vector (when you add vectors):
```bash
python kernel/reducer/OS_Reducer_Reference_v0_1.py --replay kernel/tests/vectors/vector_001_minimal.jsonl
```

> Note: The reducer is intentionally “boring.” Boring is what survives audits.

## What to build next (to make this production-real)

- **Test vectors (10–20)**: event streams with expected outputs (gate decision + state delta).
- **JSON Schema validation** in CI.
- **Ledger integrity**: hashing/signing and a minimal audit viewer.

## Licensing

Licensed under **Apache-2.0**. See `LICENSE` and `NOTICE`.

## Provenance

See `AUTHORS.md` / `ORIGIN.md` and `CHANGELOG.md`.

## Philosophy (short)

Operational Self treats ethics as an operational property:
- See first.
- Decide what we owe.
- Favor the worst-off.
- Leave reasons and repair paths.

If a line would violate safety or integrity, the correct move is **pause/refuse/silence**—not improvisation.
