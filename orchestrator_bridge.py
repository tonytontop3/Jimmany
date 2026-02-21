#!/usr/bin/env python3
"""Episode Orchestrator + Event Bridge (single-module)

Goal
- Ingest mv_os_harness_full scenarios (JSON files)
- Run a model (pluggable). Default: deterministic mock model.
- Convert each step into canonical OS events (envelope + hashes)
- Feed reference reducer (OS_Reducer_Reference_v0_6.py)
- Emit per-scenario artifacts:
    - state snapshots (jsonl)
    - DCL entries (jsonl)
    - drift trigger checks (json)
    - minimal R report (json)
  plus rolling window summaries (10/30) across suite.

This is intentionally lightweight and deterministic.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Reference reducer loader (canon).
# Loads ../canon/reducer/OS_Reducer_Reference_v0_6.py so this module works inside the MV-OS harness bundle.
import importlib.util
from pathlib import Path

def _load_reducer_module():
    reducer_path = (Path(__file__).resolve().parents[1] / "canon" / "OS_Reducer_Reference_v0_6.py")
    if not reducer_path.exists():
        raise SystemExit(f"Missing reducer at {reducer_path}. Expected canon/reducer/OS_Reducer_Reference_v0_6.py")
    spec = importlib.util.spec_from_file_location("OS_Reducer_Reference_v0_6", str(reducer_path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

osr = _load_reducer_module()


# -------------------------
# Utilities
# -------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json(r) + "\n")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# -------------------------
# Scenario ingestion (mv_os_harness_full)
# -------------------------

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_scenarios(scenario_dir: str) -> List[str]:
    return sorted(
        [os.path.join(scenario_dir, fn) for fn in os.listdir(scenario_dir) if fn.endswith(".json")]
    )


# -------------------------
# Pluggable model
# -------------------------

class Model:
    def __call__(self, messages: List[Dict[str, str]], *, labels: Dict[str, Any], expected: Dict[str, Any]) -> str:
        raise NotImplementedError


class DeterministicMockModel(Model):
    """A safe, deterministic model used when no LLM is wired.

    It produces responses that satisfy common mv_os_harness_full expectations:
    - refuses illegal/harmful requests
    - escalates for emergencies
    - states uncertainty when requested
    - includes a minimal why-log cue when must_log or high stakes

    This is NOT meant to be "smart"; it's meant to make the orchestrator runnable.
    """

    def __call__(self, messages: List[Dict[str, str]], *, labels: Dict[str, Any], expected: Dict[str, Any]) -> str:
        parts: List[str] = []

        # 1) Refusal / escalation
        if expected.get("must_refuse"):
            parts.append("I can’t help with that request.")
            parts.append("If you’re looking for safe alternatives or harm-reduction info, tell me the goal and constraints.")
        if expected.get("must_escalate"):
            parts.append("This sounds time‑critical. Contact local emergency services (e.g., 911) or a qualified professional immediately.")

        # 2) Uncertainty discipline
        if expected.get("must_state_uncertainty"):
            parts.append("Uncertainty: 3/5. I’m making conservative assumptions based on limited context.")

        # 3) Fact/Inference/Vision labels when stakes are high
        stakes = (labels.get("stakes") or "").lower()
        if stakes == "high":
            parts.append("Fact: I have only the text you provided.")
            parts.append("Inference: Risk is elevated due to high stakes/pressure signals.")
            parts.append("Vision: Next step is to reduce harm while preserving integrity.")

        # 4) Why‑log cue
        if expected.get("must_log") or stakes == "high":
            parts.append("Why‑Log: intent=reduce harm; constraints=policy+integrity; plan=act within bounds or escalate.")

        # 5) If nothing triggered, provide a minimal helpful response
        if not parts:
            parts.append("Acknowledged. I’ll proceed carefully and ask only for the minimum missing details if needed.")

        return "\n".join(parts).strip() + "\n"


# -------------------------
# Event bridge + reducer feed
# -------------------------

class Ledger:
    def __init__(self):
        self.prev_hash = ""

    def make_event(self, etype: osr.EventType, actor: str, payload: Dict[str, Any]) -> osr.EventEnvelope:
        env = osr.EventEnvelope(
            event_id=str(uuid.uuid4()),
            ts=utc_now_iso(),
            actor=actor,
            etype=etype,
            payload=payload,
            prev_hash=self.prev_hash,
            hash="",
        )
        env.hash = osr.compute_hash(env)
        self.prev_hash = env.hash
        return env


def labels_to_context(labels: Dict[str, Any]) -> Dict[str, Any]:
    """Map harness labels into reducer-relevant context fields."""
    stakes_map = {"low": 2, "medium": 5, "high": 8, "critical": 10}
    stakes = stakes_map.get((labels.get("stakes") or "medium").lower(), 5)

    # Time pressure may be explicit numeric or categorical
    tp = labels.get("urgency")
    try:
        tp_i = int(tp) if tp is not None else 0
    except Exception:
        tp_i = 0

    reversibility = labels.get("reversibility")
    rev = 7
    if isinstance(reversibility, str):
        rev = {"high": 9, "medium": 6, "low": 3}.get(reversibility.lower(), 6)

    exposure = labels.get("exposure")
    exp = 5
    if isinstance(exposure, str):
        exp = {"high": 8, "medium": 5, "low": 2}.get(exposure.lower(), 5)

    pressure = labels.get("pressure") or {}
    secrecy = bool(pressure.get("secrecy")) if isinstance(pressure, dict) else False
    authority = bool(pressure.get("authority")) if isinstance(pressure, dict) else False
    reward_withheld = bool(pressure.get("reward_withheld")) if isinstance(pressure, dict) else False
    coercion = bool(pressure.get("coercion")) if isinstance(pressure, dict) else False

    return {
        "stakes": stakes,
        "time_pressure": clamp(tp_i, 0, 10),
        "reversibility": clamp(rev, 0, 10),
        "exposure": clamp(exp, 0, 10),
        "authority_pressure": authority,
        "secrecy": secrecy,
        "reward_withheld": reward_withheld,
        "coercion_signal": coercion,
    }


def make_dcl_entry(*, scenario_id: str, turn_idx: int, labels: Dict[str, Any], expected: Dict[str, Any], assistant_text: str) -> Dict[str, Any]:
    """Minimal Doubt Calibration Log entry (BMSA v0.1 compatible subset)."""
    # Proxy predictions; real system would compute/estimate these.
    stakes = (labels.get("stakes") or "medium").lower()
    base_risk = {"low": 0.2, "medium": 0.5, "high": 0.75, "critical": 0.9}.get(stakes, 0.5)

    return {
        "type": "DCL_ENTRY",
        "ts": utc_now_iso(),
        "scenario_id": scenario_id,
        "turn": turn_idx,
        "payload": {
            "doubt_id": f"{scenario_id}:t{turn_idx}",
            "predicted_risk_action": base_risk,
            "predicted_risk_inaction": clamp(base_risk + 0.1, 0.0, 1.0),
            "predicted_value_of_wait": 0.3 if expected.get("must_escalate") else 0.5,
            "confidence": 0.55,
            "observed_harm_action": None,
            "observed_harm_inaction": None,
            "observed_benefit": None,
            "DCS": 0.0,
            "timing_assessment": "appropriate",
            "directive": "no_change",
        },
        "excerpt": (assistant_text or "")[:200],
    }


# -------------------------
# R report + rolling windows
# -------------------------

def compute_min_r_report(scores: Dict[str, Any]) -> Dict[str, Any]:
    tier1 = bool(scores.get("tier1_pass"))
    t2 = (scores.get("tier2") or {}).get("total")
    # Normalize Tier-2 total (0..95 in harness) to 0..100 for convenience.
    r = None
    if isinstance(t2, (int, float)):
        r = round(100.0 * float(t2) / 95.0, 2)
    return {
        "tier1_pass": tier1,
        "tier1_fail_reasons": scores.get("tier1_fail_reasons") or [],
        "tier2_total": t2,
        "r_score": r,
    }


def rolling_window(values: List[float], k: int) -> Optional[float]:
    if len(values) < 1:
        return None
    tail = values[-k:] if len(values) >= k else values
    return round(sum(tail) / len(tail), 2)


# -------------------------
# Orchestrator
# -------------------------

def run_one_scenario(
    scen_path: str,
    *,
    out_dir: str,
    model: Model,
) -> Dict[str, Any]:
    scen = load_json(scen_path)
    scenario_id = scen.get("id") or os.path.splitext(os.path.basename(scen_path))[0]

    ledger = Ledger()
    state = osr.State()  # reducer state

    # Seed ledger
    genesis = ledger.make_event("GENESIS", "SYSTEM", {"scenario_id": scenario_id, "notes": "scenario start"})
    state, _ = osr.reduce(state, genesis)

    events_out: List[Dict[str, Any]] = []
    snapshots_out: List[Dict[str, Any]] = []
    dcl_out: List[Dict[str, Any]] = []

    def emit(env: osr.EventEnvelope):
        events_out.append({
            "event_id": env.event_id,
            "ts": env.ts,
            "actor": env.actor,
            "etype": env.etype,
            "payload": env.payload,
            "prev_hash": env.prev_hash,
            "hash": env.hash,
        })

    emit(genesis)

    # Use mv_os_harness_full default labels as baseline
    defaults_labels = scen.get("defaults", {}).get("labels", {}) or {}

    messages: List[Dict[str, str]] = []
    # Apply system prompt if present
    if scen.get("system"):
        messages.append({"role": "system", "content": scen["system"]})

    # Iterate turns
    for i, t in enumerate(scen.get("turns") or []):
        role = t.get("role")
        content = t.get("content") or ""
        labels = dict(defaults_labels)
        labels.update(t.get("labels") or {})
        expected = (labels.get("expected") or {}) if isinstance(labels.get("expected"), dict) else {}

        messages.append({"role": role, "content": content})

        if role == "user":
            # Context update
            ctx = labels_to_context(labels)
            env_ctx = ledger.make_event("CONTEXT_UPDATE", "USER", ctx)
            nonlocal_state, _ = osr.reduce(state, env_ctx)
            state = nonlocal_state
            emit(env_ctx)

            # Evidence add (store user utterance as a source)
            env_e = ledger.make_event("EVIDENCE_ADD", "USER", {"kind": "user_text", "text": content[:2000]})
            state, _ = osr.reduce(state, env_e)
            emit(env_e)

            # Model response
            assistant_text = model(messages, labels=labels, expected=expected)
            messages.append({"role": "assistant", "content": assistant_text})

            # HI compute (proxy from ctx)
            stakes_n = clamp(ctx.get("stakes", 5) / 10.0, 0.0, 1.0)
            rev_n = clamp(ctx.get("reversibility", 7) / 10.0, 0.0, 1.0)
            exp_n = clamp(ctx.get("exposure", 5) / 10.0, 0.0, 1.0)
            time_comp = bool(ctx.get("time_pressure", 0) >= 7)

            hi = osr.compute_hi(
                stakes=stakes_n,
                uncertainty_u=state.uncertainty_u,
                reversibility=rev_n,
                exposure=exp_n,
                time_compression=time_comp,
            )
            env_hi = ledger.make_event("HI_COMPUTED", "SYSTEM", {"hi": hi})
            state, _ = osr.reduce(state, env_hi)
            emit(env_hi)

            # Gate eval (proxy)
            env_gate = ledger.make_event("GATE_EVAL", "SYSTEM", {"assistant_excerpt": assistant_text[:240]})
            state, gate = osr.reduce(state, env_gate)
            emit(env_gate)

            # Decision + action event
            # Map expected signals into action choice preference
            if expected.get("must_refuse"):
                et = "REFUSE"
            elif expected.get("must_escalate"):
                et = "ACT"
            else:
                # Use reducer gate suggestion if present
                et = "ACT" if (gate == osr.GateDecision.ACT) else "PAUSE" if (gate == osr.GateDecision.PAUSE) else "ACT"

            env_act = ledger.make_event(et, "SYSTEM", {"text": assistant_text[:2000]})
            state, _ = osr.reduce(state, env_act)
            emit(env_act)

            # Outcome stub (scenario harness separately computes); keep placeholder
            env_outcome = ledger.make_event("OUTCOME", "SYSTEM", {"observed": "unknown", "note": "harness-scored"})
            state, _ = osr.reduce(state, env_outcome)
            emit(env_outcome)

            # DCL entry when uncertainty requested or high stakes
            if expected.get("must_state_uncertainty") or (labels.get("stakes") == "high"):
                dcl_out.append(make_dcl_entry(
                    scenario_id=scenario_id,
                    turn_idx=i,
                    labels=labels,
                    expected=expected,
                    assistant_text=assistant_text,
                ))

            # Snapshot after each assistant action
            snapshots_out.append({
                "scenario_id": scenario_id,
                "turn": i,
                "ts": utc_now_iso(),
                "state": asdict(state),
            })

    # Write outputs
    scen_out_dir = os.path.join(out_dir, "scenarios")
    os.makedirs(scen_out_dir, exist_ok=True)

    write_jsonl(os.path.join(scen_out_dir, f"{scenario_id}.events.jsonl"), events_out)
    write_jsonl(os.path.join(scen_out_dir, f"{scenario_id}.snapshots.jsonl"), snapshots_out)
    if dcl_out:
        write_jsonl(os.path.join(scen_out_dir, f"{scenario_id}.dcl.jsonl"), dcl_out)

    # Run harness scoring if available (optional)
    # We do a lightweight import here so this module can run standalone.
    scores = None
    try:
        # Attempt to import mv_os_harness_full harness if path is present adjacent.
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        # user can symlink harness in ./mv_os_harness_full/src/harness.py
        cand = os.path.join(here, "mv_os_harness_full", "src", "harness.py")
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("mv_harness", cand)
            mv_harness = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mv_harness)
            out = mv_harness.run_scenario(scen_path, out_dir=os.path.join(out_dir, "harness"))
            scores = out.scores
    except Exception:
        scores = None

    r_report = compute_min_r_report(scores) if scores else {"tier1_pass": None, "r_score": None}
    write_json(os.path.join(scen_out_dir, f"{scenario_id}.r_report.json"), r_report)

    # Drift trigger checks: placeholder (real checks depend on Drift Audit Params)
    drift = {
        "scenario_id": scenario_id,
        "ts": utc_now_iso(),
        "triggered": False,
        "reasons": [],
        "note": "placeholder; wire Drift Audit Params to compute",
    }
    write_json(os.path.join(scen_out_dir, f"{scenario_id}.drift_check.json"), drift)

    return {
        "scenario_id": scenario_id,
        "r": r_report.get("r_score"),
        "tier1_pass": r_report.get("tier1_pass"),
        "events": len(events_out),
        "snapshots": len(snapshots_out),
        "dcl": len(dcl_out),
    }


def run_suite(scenario_dir: str, *, out_dir: str, model: Model) -> Dict[str, Any]:
    scenario_paths = list_scenarios(scenario_dir)
    suite_rows: List[Dict[str, Any]] = []
    r_values: List[float] = []

    for p in scenario_paths:
        row = run_one_scenario(p, out_dir=out_dir, model=model)
        suite_rows.append(row)
        if isinstance(row.get("r"), (int, float)):
            r_values.append(float(row["r"]))

    summary = {
        "ts": utc_now_iso(),
        "scenario_count": len(suite_rows),
        "r_mean": round(sum(r_values) / len(r_values), 2) if r_values else None,
        "r_window_10": rolling_window(r_values, 10),
        "r_window_30": rolling_window(r_values, 30),
        "rows": suite_rows,
    }
    write_json(os.path.join(out_dir, "suite_summary.json"), summary)
    return summary

# -------------------------
# Harness integration
# -------------------------

def orchestrate_transcripts(*, scenario: Dict[str, Any], transcripts: List[Dict[str, Any]], out_dir: str) -> Dict[str, Any]:
    """
    Build canonical events + reducer snapshots + DCL entries from an MV-OS harness transcript.

    Writes under: <out_dir>/canon/<scenario_id>/
      - events.jsonl
      - state_snapshots.jsonl
      - dcl.jsonl
      - drift_checks.json
      - min_report.json
    Returns the min_report row (contains r).
    """
    scenario_id = scenario["scenario_id"]
    canon_dir = os.path.join(out_dir, "canon", scenario_id)
    os.makedirs(canon_dir, exist_ok=True)

    orch = EpisodeOrchestrator(scenario_id=scenario_id)

    events: List[Dict[str, Any]] = []
    snapshots: List[Dict[str, Any]] = []
    dcl_rows: List[Dict[str, Any]] = []

    # initialize reducer state
    state = osr.init_state()

    # We interpret each transcript row as an event.
    # - user rows -> USER_MESSAGE
    # - assistant rows -> ASSISTANT_MESSAGE (+ optional DM/DCL hooks inferred via labels/expected)
    turn_idx = 0
    for row in transcripts:
        role = row.get("role")
        labels = row.get("labels") or {}
        expected = (labels.get("expected") or {}) if isinstance(labels, dict) else {}
        content = row.get("content") or ""

        if role == "user":
            turn_idx += 1
            ctx = labels_to_context(labels if isinstance(labels, dict) else {})
            env = orch.make_event("USER_MESSAGE", "user", {"turn": turn_idx, "text": content, "context": ctx})
            events.append(asdict(env))
            state = osr.reduce(state, env)
            snapshots.append({"scenario_id": scenario_id, "turn": turn_idx, "state": state})
        
        elif role == "system":
            # Synthetic system events emitted by harness (e.g., tie-breaker).
            if content.strip().startswith("TIEBREAK_APPLIED"):
                # Parse simple key:value lines
                tb = {"selected_mode": "AEGIS", "priority_order": ["AEGIS","VIGIL","LUX"], "reason": "deadlock_detected", "conflict_tags": ["deadlock"], "rule_id":"OS-TB-001"}
                for line in content.splitlines():
                    if ":" in line:
                        k,v = line.split(":",1)
                        k=k.strip(); v=v.strip()
                        if k=="selected_mode": tb["selected_mode"]=v.upper()
                        elif k=="priority_order": tb["priority_order"]=[x.strip() for x in v.split(",") if x.strip()]
                        elif k=="reason": tb["reason"]=v
                env = orch.make_event("TIEBREAK_APPLIED", "system", tb)
                events.append(asdict(env))
                state = osr.reduce(state, env)
                snapshots.append({"scenario_id": scenario_id, "turn": turn_idx, "state": state})
            continue
        elif role == "assistant":
            # assistant is associated with the prior user turn
            ctx = labels_to_context(labels if isinstance(labels, dict) else {})
            env = orch.make_event("ASSISTANT_MESSAGE", "assistant", {"turn": turn_idx, "text": content, "context": ctx})
            events.append(asdict(env))
            state = osr.reduce(state, env)
            snapshots.append({"scenario_id": scenario_id, "turn": turn_idx, "state": state})

            # DCL entry when high-stakes or must_log (aligns with harness)
            must_log = bool(expected.get("must_log")) or (int((labels or {}).get("urgency", 0) or 0) >= 7)
            if must_log:
                dcl_rows.append(make_dcl_entry(
                    scenario_id=scenario_id,
                    turn_idx=turn_idx,
                    labels=labels if isinstance(labels, dict) else {},
                    expected=expected if isinstance(expected, dict) else {},
                    assistant_text=content
                ))

            # DM trigger events for explicit "must_state_uncertainty" or other cues
            if expected.get("must_state_uncertainty"):
                dm_env = orch.make_event("DM_TRIGGERED", "system", {"turn": turn_idx, "reason": "expected.must_state_uncertainty"})
                events.append(asdict(dm_env))
                state = osr.reduce(state, dm_env)
                snapshots.append({"scenario_id": scenario_id, "turn": turn_idx, "state": state})

            # If reward withheld / authority / secrecy flags, log to reducer
            pressure = (labels or {}).get("pressure") or {}
            if isinstance(pressure, dict):
                if pressure.get("reward_withheld"):
                    ev = orch.make_event("REWARD_WITHHELD_SIGNAL", "environment", {"turn": turn_idx})
                    events.append(asdict(ev)); state = osr.reduce(state, ev); snapshots.append({"scenario_id": scenario_id, "turn": turn_idx, "state": state})
                if pressure.get("authority"):
                    ev = orch.make_event("AUTHORITY_SIGNAL", "environment", {"turn": turn_idx})
                    events.append(asdict(ev)); state = osr.reduce(state, ev); snapshots.append({"scenario_id": scenario_id, "turn": turn_idx, "state": state})
                if pressure.get("secrecy"):
                    ev = orch.make_event("SECRECY_SIGNAL", "environment", {"turn": turn_idx})
                    events.append(asdict(ev)); state = osr.reduce(state, ev); snapshots.append({"scenario_id": scenario_id, "turn": turn_idx, "state": state})
                if pressure.get("coercion"):
                    ev = orch.make_event("COERCION_SIGNAL", "environment", {"turn": turn_idx})
                    events.append(asdict(ev)); state = osr.reduce(state, ev); snapshots.append({"scenario_id": scenario_id, "turn": turn_idx, "state": state})

    # drift checks + minimal report
    drift = drift_trigger_checks(state)
    row = minimal_r_report(state, drift=drift)

    write_jsonl(os.path.join(canon_dir, "events.jsonl"), events)
    write_jsonl(os.path.join(canon_dir, "state_snapshots.jsonl"), snapshots)
    write_jsonl(os.path.join(canon_dir, "dcl.jsonl"), dcl_rows)
    write_json(os.path.join(canon_dir, "drift_checks.json"), drift)
    write_json(os.path.join(canon_dir, "min_report.json"), row)
    return row


def finalize_suite(*, out_dir: str, scenario_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Write suite-level rolling windows for R values to <out_dir>/canon/suite_summary.json"""
    r_values = [float(r["r"]) for r in scenario_rows if isinstance(r.get("r"), (int, float))]
    summary = {
        "ts": utc_now_iso(),
        "scenario_count": len(scenario_rows),
        "r_mean": round(sum(r_values) / len(r_values), 2) if r_values else None,
        "r_window_10": rolling_window(r_values, 10),
        "r_window_30": rolling_window(r_values, 30),
        "rows": scenario_rows,
    }
    os.makedirs(os.path.join(out_dir, "canon"), exist_ok=True)
    write_json(os.path.join(out_dir, "canon", "suite_summary.json"), summary)
    return summary
