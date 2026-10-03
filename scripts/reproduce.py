"""Reproduce the paper's results files and check the paper against them.

Default: no model inference and no downloads. Replays the C8 convention control from the
saved judge calls and the label-only MultiPref derivative, compares it with the recorded
runs/c8/eval.json, and runs the paper's number verifier.

--from-corpora: also reruns every analysis that reads the public corpora (the H0 gate,
the C2/C4/C5/C6 leaderboard studies, the C3 judge table and the C7 corpus-weighted judge
estimates) and compares each with its recorded results file. `datasets` downloads
MultiPref and the MT-Bench human judgments on first use.

Replays are written under runs/_replay/; recorded results are never overwritten.
Comparisons ignore only when each run happened and how long it took (`utc`, `wall_s`).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPLAY = ROOT / "runs/_replay"

OFFLINE = [("c8_judge_convention_control", "runs/c8/eval.json")]
CORPORA = [
    ("h0_gate_multipref", "runs/h0/eval_multipref.json"),
    ("c2_whose_gold", "runs/c2/eval.json"),
    ("c4_generalize_mtbench", "runs/c4/eval.json"),
    ("c5_robustness", "runs/c5/robustness.json"),
    ("c6_displacement_threshold", "runs/c6/eval.json"),
    ("c3_table2", "runs/c3/table2.json"),
    ("c7_ipw_judge_pools", "runs/c7/eval.json"),
]
RUN_STAMPS = {"utc", "wall_s"}


def without_runtime(obj):
    if isinstance(obj, dict):
        return {k: without_runtime(v) for k, v in obj.items() if k not in RUN_STAMPS}
    if isinstance(obj, list):
        return [without_runtime(v) for v in obj]
    return obj


def replay(script: str, recorded: str, env: dict) -> bool:
    out = REPLAY / f"{script}.json"
    subprocess.run([sys.executable, f"experiments/haic/{script}.py", "--out", str(out)],
                   cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    same = (without_runtime(json.loads(out.read_text()))
            == without_runtime(json.loads((ROOT / recorded).read_text())))
    print(f"{'PASS' if same else 'FAIL'}  {script} -> {recorded}")
    return same


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from-corpora", action="store_true",
                    help="also rerun the analyses that read the public corpora")
    args = ap.parse_args()

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    if not args.from_corpora:
        env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    REPLAY.mkdir(parents=True, exist_ok=True)

    stages = OFFLINE + (CORPORA if args.from_corpora else [])
    failed = [s for s, rec in stages if not replay(s, rec, env)]
    subprocess.run([sys.executable, "paper/verify_paper_numbers.py"], cwd=ROOT, env=env, check=True)
    if failed:
        raise SystemExit(f"FAIL: {len(failed)} replay(s) differ from the recorded results: {failed}")
    print(f"PASS: {len(stages)} replay(s) match the recorded results apart from runtime")


if __name__ == "__main__":
    main()
