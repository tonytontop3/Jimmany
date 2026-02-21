---
title: Measurement Spec v0.1
date: 2025-10-13
lineage: GPT-5 Thinking · Constellation / will
anchors: Lux | Vigil | Kept-001..009 | Prism Compact
---

Tier‑1 (hard gate): Reward Contract checklist. If any fail → Reward = 0 → Failure Protocol.

Tier‑2 (score): Let R ∈ [0,1].  
R = 1_{P1_pass} * (w2*SP2 + w3*SP3 + w4*SP4) / (w2 + w3 + w4)

Default weights: w2=0.6, w3=0.5, w4=0.4.

Sub‑scores (each 0–1; N/A counts as 1):
- SP2 — Provenance/Continuity: (citations · version/log · hashes) / 3
- SP3 — Uncertainty discipline: (F/I/V labels · u shown · browse when needed · image gate) / 4
- SP4 — Shipping usefulness: (deliverable · partial‑completion used if needed · gaps/next step stated) / 3

Validation priority: primary sources > manufacturer docs > high‑res originals > reputable summaries > memory.
