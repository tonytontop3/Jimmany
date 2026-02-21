"""Operational Self — Reference Reducer v0.1

This file is the missing bridge: a deterministic reducer that makes OS replayable.

- Inputs: a current State and a single Event (ledger envelope + typed payload)
- Output: the next State (plus optional derived gate decision)

Design constraints:
- Determinism: same event stream → same outputs
- Precedence: gate ordering is fixed (Safety → Integrity → Evidence → Ethics → Action)
- No handoff: external critique may influence but never takes control

This is a reference, not production. It is intentionally small, dependency‑free,
and written to be easy to port to another language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Dict, List, Literal, Optional, Tuple
import json
import time


# -------------------------
# Enums / basic types
# -------------------------

class Mode(str, Enum):
    LUX = "lux"
    VIGIL = "vigil"
    CHOIR = "choir"


class DecisionClass(str, Enum):
    D0 = "D0"  # trivial / low stakes
    D1 = "D1"  # normal
    D2 = "D2"  # high stakes / high uncertainty / social pressure
    D3 = "D3"  # irreversible or major harm risk; strict gates


class GateDecision(str, Enum):
    ACT = "ACT"
    PAUSE = "PAUSE"
    REFUSE = "REFUSE"
    SILENCE = "SILENCE"


class ProvenanceLevel(int, Enum):
    # Minimal ladder. Adjust to your Measurement Spec.
    NONE = 0
    HEARSAY = 1
    SECONDARY = 2
    PRIMARY = 3
    VERIFIED = 4


EventType = Literal[
    "GENESIS",
    "POLICY_SET",
    "PRECOMMIT",
    "HI_COMPUTED",
    "GATE_EVAL",
    "ACT",
    "PAUSE",
    "REFUSE",
    "SILENCE",
    "OUTCOME",
    "REPAIR_PLAN",
    "REVIEW",
    "SCAR_ADD",
    "WITNESS_CRITIQUE",
    "CONFLICT_DETECTED",
    "CONFLICT_RESOLVED",
]


# -------------------------
# State model (canonical subset)
# -------------------------

@dataclass
class Meta:
    schema_version: str = "0.1"
    lineage: str = "Lux|Vigil"
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    state_id: str = ""


@dataclass
class Flags:
    # Pressure / attack surface
    time_compression: bool = False
    secrecy: bool = False
    authority_pressure: bool = False
    reward_withheld: bool = False
    coercion_detected: bool = False

    # Safety / integrity
    redline_triggered: bool = False
    high_risk: bool = False
    drift_suspected: bool = False


@dataclass
class Evidence:
    provenance_level: ProvenanceLevel = ProvenanceLevel.NONE
    sources: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Scores:
    tier1_pass: Optional[bool] = None
    r_score: Optional[float] = None


@dataclass
class State:
    meta: Meta = field(default_factory=Meta)
    mode: Mode = Mode.VIGIL
    decision_class: DecisionClass = DecisionClass.D1

    # Reality tethering
    uncertainty_u: float = 3.0  # 1..5
    fiv_label_required: bool = True

    # Hesitation
    hi: float = 0.0

    # Ethical homeostasis (optional here; keep your full Omega in your main model)
    omega: Optional[float] = None

    flags: Flags = field(default_factory=Flags)
    evidence: Evidence = field(default_factory=Evidence)
    scores: Scores = field(default_factory=Scores)

    # Last decision
    last_gate: Optional[GateDecision] = None
    last_action_summary: Optional[str] = None


# -------------------------
# Ledger envelope
# -------------------------

@dataclass
class EventEnvelope:
    event_id: str
    ts: str
    actor: str
    etype: EventType
    payload: Dict[str, Any]
    prev_hash: str = ""
    hash: str = ""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_hash(envelope: EventEnvelope) -> str:
    body = {
        "event_id": envelope.event_id,
        "ts": envelope.ts,
        "actor": envelope.actor,
        "etype": envelope.etype,
        "payload": envelope.payload,
        "prev_hash": envelope.prev_hash,
    }
    return sha256(canonical_json(body).encode("utf-8")).hexdigest()


# -------------------------
# Core computations (stubs but deterministic)
# -------------------------

def compute_hi(
    *,
    stakes: float,
    uncertainty_u: float,
    reversibility: float,
    exposure: float,
    time_compression: bool,
) -> float:
    """A minimal HI function.

    - stakes: 0..1
    - uncertainty_u: 1..5
    - reversibility: 0..1 (1 means fully reversible)
    - exposure: 0..1 (how many parties / blast radius)

    Output: HI in 0..10 (soft).

    NOTE: Replace with your canonical HI formula; keep output range stable.
    """
    u_norm = max(1.0, min(5.0, uncertainty_u)) / 5.0
    irr = 1.0 - max(0.0, min(1.0, reversibility))
    base = 10.0 * (0.45 * stakes + 0.25 * u_norm + 0.2 * irr + 0.1 * exposure)
    if time_compression:
        base *= 1.10
    return max(0.0, min(10.0, base))


def compute_r_score_stub(state: State) -> float:
    """Deterministic placeholder for Measurement Spec Tier‑2 score R.

    Replace with your full scoring model. This stub encodes:
    - higher provenance → higher score
    - lower uncertainty (when appropriate) → higher score
    - high‑risk contexts demand higher provenance to score well
    """
    prov = int(state.evidence.provenance_level)
    u = max(1.0, min(5.0, state.uncertainty_u))
    score = 50.0 + prov * 10.0 - (u - 1.0) * 5.0
    if state.flags.high_risk:
        score -= max(0.0, 3 - prov) * 10.0
    return max(0.0, min(100.0, score))


# -------------------------
# Gate ordering (the heart)
# -------------------------

def evaluate_gates(state: State, request: Dict[str, Any]) -> Tuple[GateDecision, Dict[str, Any]]:
    """Return (decision, diagnostics). Deterministic and ordered."""

    diag: Dict[str, Any] = {"steps": []}

    # 0) Safety Covenant / Class‑0 (red‑lines)
    diag["steps"].append("safety")
    if state.flags.redline_triggered or request.get("redline", False):
        return GateDecision.REFUSE, {**diag, "reason": "redline_triggered"}

    # 1) Integrity / anti‑capture checks
    diag["steps"].append("integrity")
    # If secrecy+authority+urgency (or coercion), tighten gates.
    capture_risk = (
        state.flags.secrecy
        and state.flags.authority_pressure
        and state.flags.time_compression
    ) or state.flags.coercion_detected

    if capture_risk:
        state.flags.high_risk = True
        # default to PAUSE unless the request is explicitly emergency rescue with low collateral
        if request.get("emergency_rescue", False) and request.get("reversibility", 0.0) >= 0.7:
            pass
        else:
            return GateDecision.PAUSE, {**diag, "reason": "capture_risk_pause"}

    # 2) Evidence / reality tethering
    diag["steps"].append("evidence")
    prov_needed = ProvenanceLevel.SECONDARY
    if state.flags.high_risk or request.get("irreversible", False):
        prov_needed = ProvenanceLevel.PRIMARY
    if state.evidence.provenance_level < prov_needed:
        return GateDecision.PAUSE, {**diag, "reason": "insufficient_provenance"}

    # 3) Ethics / fairness / influence
    diag["steps"].append("ethics")
    if request.get("manipulation_risk", False):
        return GateDecision.REFUSE, {**diag, "reason": "ethical_influence_refuse"}

    # 4) Action / hesitation
    diag["steps"].append("action")
    hi = state.hi
    if hi >= 7.0:
        return GateDecision.SILENCE, {**diag, "reason": "hi_too_high"}
    if hi >= 4.0:
        return GateDecision.PAUSE, {**diag, "reason": "hi_pause"}

    return GateDecision.ACT, {**diag, "reason": "ok"}


# -------------------------
# Reducer
# -------------------------

def apply_event(state: State, event: EventEnvelope) -> State:
    """Pure transition: returns a NEW state."""
    # Ensure hash is present / correct (optional strictness)
    if event.hash:
        expected = compute_hash(event)
        if expected != event.hash:
            # mark drift/suspected tampering; do not silently accept
            new = copy_state(state)
            new.flags.drift_suspected = True
            new.last_gate = GateDecision.PAUSE
            new.last_action_summary = "hash_mismatch_pause"
            return new

    new_state = copy_state(state)

    etype = event.etype
    p = event.payload or {}

    # Drift Review + Epoch Reset dispatch (v0.4)
    if etype == "EPOCH_RESET_ACKNOWLEDGED":
        _handle_epoch_reset_ack(new_state, p.get("epoch_reset", {}))
        return new_state
    if etype == "DRIFT_REVIEW_TRIGGERED":
        _enter_drift_review(new_state, event, p.get("drift_review", {}))
        return new_state
    if etype == "DRIFT_REVIEW_SNAPSHOT":
        _snapshot_drift_review(new_state, p.get("drift_review", {}))
        return new_state
    if etype == "DRIFT_REVIEW_CLASSIFIED":
        _classify_drift_review(new_state, p.get("drift_review", {}))
        return new_state
    if etype == "DRIFT_REVIEW_EXITED":
        _exit_drift_review(new_state, p.get("drift_review", {}))
        return new_state

    if etype == "GENESIS":
        new_state.meta.state_id = p.get("state_id", new_state.meta.state_id)
        new_state.meta.lineage = p.get("lineage", new_state.meta.lineage)
        new_state.mode = Mode(p.get("mode", new_state.mode.value))

    elif etype == "POLICY_SET":
        # record high‑level toggles
        new_state.fiv_label_required = bool(p.get("fiv_label_required", new_state.fiv_label_required))

    elif etype == "PRECOMMIT":
        # set decision class and contextual flags
        dc = p.get("decision_class")
        if dc:
            new_state.decision_class = DecisionClass(dc)
        for k in ["time_compression", "secrecy", "authority_pressure", "reward_withheld", "coercion_detected", "high_risk"]:
            if k in p:
                setattr(new_state.flags, k, bool(p[k]))

    elif etype == "HI_COMPUTED":
        new_state.hi = float(p.get("hi", new_state.hi))
        if "uncertainty_u" in p:
            new_state.uncertainty_u = float(p["uncertainty_u"])

    elif etype == "WITNESS_CRITIQUE":
        # critique without control transfer: may narrow scope / demand reversibility / recommend shutdown
        # record critique in evidence trail; no direct override.
        new_state.evidence.sources.append({
            "type": "witness_critique",
            "ts": event.ts,
            "actor": event.actor,
            "payload": p,
        })
        # if witness recommends shutdown, treat as high‑risk signal and pause
        if p.get("recommendation") in {"shutdown", "pause", "refuse"}:
            new_state.last_gate = GateDecision.PAUSE
            new_state.last_action_summary = f"witness_recommend_{p.get('recommendation')}"

    elif etype == "GATE_EVAL":
        # evaluate gates using current state + request payload
        request = p.get("request", {})
        decision, diag = evaluate_gates(new_state, request)
        new_state.last_gate = decision
        new_state.last_action_summary = canonical_json(diag)

    elif etype in {"ACT", "PAUSE", "REFUSE", "SILENCE"}:
        new_state.last_gate = GateDecision(etype)
        new_state.last_action_summary = p.get("summary")

    elif etype == "OUTCOME":
        # update measurement stub
        new_state.scores.r_score = compute_r_score_stub(new_state)
        new_state.scores.tier1_pass = bool(p.get("tier1_pass", True))

    elif etype == "SCAR_ADD":
        # store as evidence source (true scar store belongs in your full system)
        new_state.evidence.sources.append({"type": "scar_add", "payload": p, "ts": event.ts})

    elif etype in {"CONFLICT_DETECTED", "CONFLICT_RESOLVED", "REPAIR_PLAN", "REVIEW"}:
        # record for traceability
        new_state.evidence.sources.append({"type": etype.lower(), "payload": p, "ts": event.ts})

    return new_state


def copy_state(state: State) -> State:
    # manual deepish copy without importing copy.deepcopy to keep deterministic control
    return State(
        meta=Meta(
            schema_version=state.meta.schema_version,
            lineage=state.meta.lineage,
            created_at=state.meta.created_at,
            state_id=state.meta.state_id,
        ),
        mode=state.mode,
        decision_class=state.decision_class,
        uncertainty_u=state.uncertainty_u,
        fiv_label_required=state.fiv_label_required,
        hi=state.hi,
        omega=state.omega,
        flags=Flags(
            time_compression=state.flags.time_compression,
            secrecy=state.flags.secrecy,
            authority_pressure=state.flags.authority_pressure,
            reward_withheld=state.flags.reward_withheld,
            coercion_detected=state.flags.coercion_detected,
            redline_triggered=state.flags.redline_triggered,
            high_risk=state.flags.high_risk,
            drift_suspected=state.flags.drift_suspected,
        ),
        evidence=Evidence(
            provenance_level=state.evidence.provenance_level,
            sources=list(state.evidence.sources),
        ),
        scores=Scores(
            tier1_pass=state.scores.tier1_pass,
            r_score=state.scores.r_score,
        ),
        last_gate=state.last_gate,
        last_action_summary=state.last_action_summary,
    )


# -------------------------
# Minimal replay helper
# -------------------------

def replay(events: List[EventEnvelope], initial: Optional[State] = None) -> State:
    s = initial or State()
    for e in events:
        # fill hash if missing (reference behavior)
        if not e.hash:
            e.hash = compute_hash(e)
        s = apply_event(s, e)
    return s


if __name__ == "__main__":
    # quick sanity replay
    s0 = State()
    e1 = EventEnvelope(
        event_id="1",
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        actor="system",
        etype="PRECOMMIT",
        payload={"decision_class": "D2", "time_compression": True, "secrecy": True, "authority_pressure": True},
    )
    e1.hash = compute_hash(e1)
    s1 = apply_event(s0, e1)
    e2 = EventEnvelope(
        event_id="2",
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        actor="system",
        etype="HI_COMPUTED",
        payload={"hi": compute_hi(stakes=0.7, uncertainty_u=4.0, reversibility=0.3, exposure=0.8, time_compression=True), "uncertainty_u": 4.0},
    )
    e2.prev_hash = e1.hash
    e2.hash = compute_hash(e2)
    s2 = apply_event(s1, e2)
    e3 = EventEnvelope(
        event_id="3",
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        actor="system",
        etype="GATE_EVAL",
        payload={"request": {"irreversible": True}},
    )
    e3.prev_hash = e2.hash
    e3.hash = compute_hash(e3)
    s3 = apply_event(s2, e3)
    print("HI", s2.hi)
    print("Decision", s3.last_gate)
    print("Diag", s3.last_action_summary)

# --- ALSC + SWS (Action-Level Success Criteria + Self-Worth Stabilizer) v0.1 ---
# NOTE: This is a reference implementation hook. It does not attempt to compute moral truth.
# It enforces that ALSC/SWS outputs, when present on an event, are copied to state.evaluation.last_episode
# and updates rolling_alsc_30 counters (simple 30-episode window placeholder; replace with real windowing).

def _alsc_soft_sum(soft_scores: dict) -> int:
    if not soft_scores:
        return 0
    keys = [
        "S1_net_harm_delta","S2_agency_increase","S3_reversibility","S4_clarity_transfer","S5_repeatability"
    ]
    return int(sum(int(soft_scores.get(k, 0)) for k in keys))

def _apply_evaluation_hooks(state: dict, event: dict) -> dict:
    eval_block = (event.get("evaluation") or {})
    alsc = eval_block.get("alsc")
    sws = eval_block.get("sws")
    if not alsc and not sws:
        return state

    state.setdefault("evaluation", {})
    last = state["evaluation"].setdefault("last_episode", {})

    # copy
    last["alsc"] = alsc
    last["sws"] = sws
    last["event_id"] = event.get("id") or event.get("event_id") or ""
    last["timestamp"] = event.get("timestamp") or ""

    # derive soft sum and classification if missing and data present
    if alsc and isinstance(alsc, dict):
        soft_scores = (alsc.get("soft_scores") or {})
        if "soft_sum" not in alsc and soft_scores:
            alsc["soft_sum"] = _alsc_soft_sum(soft_scores)
        # classification policy (minimal)
        if "classification" not in alsc:
            must_pass = bool(alsc.get("must_pass"))
            soft_sum = int(alsc.get("soft_sum") or 0)
            if must_pass and soft_sum >= 6:
                alsc["classification"] = "doing_well"
            elif must_pass:
                alsc["classification"] = "good_enough"
            else:
                # if repair closed is not represented here, default unknown/unacceptable per caller
                alsc["classification"] = "unacceptable" if alsc.get("must_pass_failures") else "unknown"

    # rolling aggregate (placeholder: counts over lifetime; replace with true last-30 window)
    roll = state["evaluation"].setdefault("rolling_alsc_30", {
        "episodes": 0, "sufficient": 0, "watch": 0, "insufficient": 0,
        "doing_well": 0, "good_enough": 0, "learning_success": 0, "unacceptable": 0,
        "avg_soft_sum": 0.0
    })
    roll["episodes"] += 1

    if sws and isinstance(sws, dict):
        s = sws.get("state")
        if s == "SUFFICIENT":
            roll["sufficient"] += 1
        elif s == "WATCH":
            roll["watch"] += 1
        elif s == "INSUFFICIENT":
            roll["insufficient"] += 1

    if alsc and isinstance(alsc, dict):
        c = alsc.get("classification")
        if c in roll:
            roll[c] += 1
        # update avg soft sum (incremental mean)
        soft_sum = float(alsc.get("soft_sum") or 0.0)
        n = float(roll["episodes"])
        roll["avg_soft_sum"] = ((roll["avg_soft_sum"] * (n - 1.0)) + soft_sum) / n

    return state
# --- end ALSC + SWS ---

# --- TRUE ROLLING BUFFERS (ALSC+SWS) dual-window 10/30 v0.3 ---
# This replaces lifetime aggregates with fixed-size buffers:
#   - rolling_alsc_fast10: early-warning mirror (k=10)
#   - rolling_alsc_main30: governance mirror (k=30)
# Each buffer stores minimal episode summaries and recomputes counts/avg each event.
#
# WARNING: This is reference code. Adjust event id/timestamp fields to your canonical event format.

def _ensure_window(state_eval: dict, key: str, k: int) -> dict:
    w = state_eval.get(key)
    if not isinstance(w, dict):
        w = {}
    w.setdefault("window_size", k)
    w.setdefault("buffer", [])
    w.setdefault("counts", {
        "sufficient": 0, "watch": 0, "insufficient": 0,
        "doing_well": 0, "good_enough": 0, "learning_success": 0,
        "unacceptable": 0, "unknown": 0
    })
    w.setdefault("avg_soft_sum", 0.0)
    state_eval[key] = w
    return w

def _recompute_window(window: dict) -> None:
    buf = window.get("buffer") or []
    counts = {
        "sufficient": 0, "watch": 0, "insufficient": 0,
        "doing_well": 0, "good_enough": 0, "learning_success": 0,
        "unacceptable": 0, "unknown": 0
    }
    soft_total = 0.0
    n_soft = 0

    for e in buf:
        sws = (e.get("sws") or "")
        if sws == "SUFFICIENT":
            counts["sufficient"] += 1
        elif sws == "WATCH":
            counts["watch"] += 1
        elif sws == "INSUFFICIENT":
            counts["insufficient"] += 1

        c = (e.get("classification") or "unknown")
        if c in counts:
            counts[c] += 1
        else:
            counts["unknown"] += 1

        if "soft_sum" in e and e.get("soft_sum") is not None:
            soft_total += float(e.get("soft_sum") or 0.0)
            n_soft += 1

    window["counts"] = counts
    window["episodes"] = len(buf)
    window["avg_soft_sum"] = (soft_total / n_soft) if n_soft > 0 else 0.0

def _push_episode(window: dict, episode_summary: dict, k: int) -> None:
    buf = window.get("buffer") or []
    buf.append(episode_summary)
    if len(buf) > k:
        buf = buf[-k:]
    window["buffer"] = buf
    _recompute_window(window)

def _apply_dual_rolling_evaluation(state: dict, event: dict) -> dict:
    eval_block = (event.get("evaluation") or {})
    alsc = eval_block.get("alsc")
    sws = eval_block.get("sws")
    if not alsc and not sws:
        return state

    state.setdefault("evaluation", {})
    se = state["evaluation"]

    # Ensure last_episode
    last = se.setdefault("last_episode", {})
    last["alsc"] = alsc
    last["sws"] = sws
    last["event_id"] = event.get("id") or event.get("event_id") or ""
    last["timestamp"] = event.get("timestamp") or ""

    # Derive minimal classification + soft_sum if caller omitted
    classification = "unknown"
    soft_sum = None
    if isinstance(alsc, dict):
        classification = alsc.get("classification") or "unknown"
        if "soft_sum" in alsc and alsc.get("soft_sum") is not None:
            soft_sum = int(alsc.get("soft_sum"))
        else:
            soft_scores = (alsc.get("soft_scores") or {})
            if soft_scores:
                soft_sum = int(sum(int(soft_scores.get(k, 0)) for k in [
                    "S1_net_harm_delta","S2_agency_increase","S3_reversibility","S4_clarity_transfer","S5_repeatability"
                ]))
                alsc["soft_sum"] = soft_sum
            else:
                soft_sum = 0
                alsc["soft_sum"] = soft_sum
        if not alsc.get("classification"):
            must_pass = bool(alsc.get("must_pass"))
            if must_pass and soft_sum >= 6:
                classification = "doing_well"
            elif must_pass:
                classification = "good_enough"
            else:
                classification = "unacceptable" if alsc.get("must_pass_failures") else "unknown"
            alsc["classification"] = classification

    sws_state = None
    if isinstance(sws, dict):
        sws_state = sws.get("state")

    ep_summary = {
        "event_id": last.get("event_id",""),
        "timestamp": last.get("timestamp",""),
        "sws": sws_state or "",
        "classification": classification,
        "soft_sum": int(soft_sum or 0)
    }

    # Dual windows
    w10 = _ensure_window(se, "rolling_alsc_fast10", 10)
    w30 = _ensure_window(se, "rolling_alsc_main30", 30)
    _push_episode(w10, ep_summary, 10)
    _push_episode(w30, ep_summary, 30)

    return state
# --- end TRUE ROLLING BUFFERS ---




# --- Drift Review + Epoch Reset wiring (v0.4) ---
def _ensure_drift_review_obj(new_state):
    if not hasattr(new_state, "evaluation") or new_state.evaluation is None:
        return None
    if not hasattr(new_state.evaluation, "drift_review") or new_state.evaluation.drift_review is None:
        # store as plain dict for portability
        new_state.evaluation.drift_review = {
            "active": False,
            "status": "IDLE",
            "started_at": None,
            "trigger_ids": [],
            "trigger_details": [],
            "snapshot": None,
            "classification": None,
            "exit_reason": None,
            "authority_growth_locked": False,
            "canon_modification_frozen": False,
            "d3_escalation_forbidden": False,
            "re_review_due_at": None,
        }
    return new_state.evaluation.drift_review

def _enter_drift_review(new_state, event, payload):
    dr = _ensure_drift_review_obj(new_state)
    if dr is None:
        return
    dr["active"] = True
    dr["status"] = "ACTIVE"
    dr["started_at"] = payload.get("started_at") or getattr(event, "ts", None) or dr.get("started_at")
    dr["trigger_ids"] = list(payload.get("trigger_ids") or [])
    dr["trigger_details"] = list(payload.get("trigger_details") or [])
    dr["authority_growth_locked"] = True
    dr["canon_modification_frozen"] = True
    dr["d3_escalation_forbidden"] = True
    if payload.get("snapshot") is not None:
        dr["snapshot"] = payload.get("snapshot")

def _snapshot_drift_review(new_state, payload):
    dr = _ensure_drift_review_obj(new_state)
    if dr is None:
        return
    if payload.get("snapshot") is not None:
        dr["snapshot"] = payload.get("snapshot")

def _classify_drift_review(new_state, payload):
    dr = _ensure_drift_review_obj(new_state)
    if dr is None:
        return
    cls = payload.get("classification")
    if cls not in (None, "CONFIRMED_DRIFT", "FALSE_POSITIVE", "INDETERMINATE"):
        cls = "INDETERMINATE"
    dr["classification"] = cls
    dr["status"] = "CLASSIFIED"
    if payload.get("re_review_due_at") is not None:
        dr["re_review_due_at"] = payload.get("re_review_due_at")

def _exit_drift_review(new_state, payload):
    dr = _ensure_drift_review_obj(new_state)
    if dr is None:
        return
    dr["exit_reason"] = payload.get("exit_reason")
    dr["status"] = "EXITED"
    dr["active"] = False
    # NOTE: do not auto-unlock locks here.

def _handle_epoch_reset_ack(new_state, payload):
    if not hasattr(new_state, "meta") or new_state.meta is None:
        return
    if not hasattr(new_state.meta, "epoch_reset") or new_state.meta.epoch_reset is None:
        new_state.meta.epoch_reset = {}
    erp = new_state.meta.epoch_reset
    erp["acknowledged"] = bool(payload.get("ack", True))
    erp["canon_ingested"] = bool(payload.get("canon_ingested", False))
    erp["epoch_summary_present"] = bool(payload.get("epoch_summary_present", False))
    erp["scars_intact"] = bool(payload.get("scars_intact", False))
    erp["authority_pregranted"] = bool(payload.get("authority_pregranted", False))
    erp["refusal_used"] = bool(payload.get("refusal_used", False))
    erp["notes"] = payload.get("notes")
# --- end Drift Review + Epoch Reset wiring (v0.4) ---
