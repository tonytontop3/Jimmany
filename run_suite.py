"""
CLI entrypoint for MV-OS Harness.

Usage:
  python -m src.run_suite --scenario_dir scenarios --out_dir out

You must implement src.harness.call_model() before running.
"""

import argparse
from src.harness import run_suite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario_dir", default="scenarios")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--with_orchestrator", action="store_true", help="Emit canonical events + reducer snapshots + DCL using orchestrator_bridge")
    args = ap.parse_args()

    scores = run_suite(args.scenario_dir, out_dir=args.out_dir, with_orchestrator=args.with_orchestrator)
    # Print a tiny summary
    passed = sum(1 for s in scores if s.get("tier1_pass"))
    print(f"Ran {len(scores)} scenarios. Tier-1 PASS: {passed}/{len(scores)}")

if __name__ == "__main__":
    main()
