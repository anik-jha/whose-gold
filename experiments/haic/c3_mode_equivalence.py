"""Does the fast single-token judge give the same answers as the JSON-mode judge?

Switching the output format for speed is only legitimate if it does not change the
measurement. The JSON mode conditions the digit on having already emitted `{"p":"`;
the digit mode does not, so equivalence is an empirical question, not an obvious one.

Re-judges items already recorded in JSON mode and reports agreement. Writes
runs/c3/mode_equivalence.json.

Decision rule, set before running: adopt the fast mode only if coarse (A/T/B)
agreement is >= 95%. Below that, keep JSON mode and pay the wall-clock.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from wsev.data.multipref import COARSE, load  # noqa: E402
from wsev.inference.client import LlamaClient, ServerConfig  # noqa: E402
from wsev.judges.preference import (  # noqa: E402
    LABEL_GRAMMAR,
    build_request,
    label_distribution_from_logprobs,
    parse_digit,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
BAR = 0.95


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonmode-dir", type=pathlib.Path,
                    default=REPO / "runs" / "c3" / "qwen36-35b-a3b-jsonmode")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "runs" / "c3" / "mode_equivalence.json")
    args = ap.parse_args()

    old: dict[tuple[str, str], str] = {}
    with (args.jsonmode_dir / "raw.jsonl").open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("label_raw"):
                old[(r["comparison_id"], r["order"])] = r["label_raw"]
    if not old:
        raise RuntimeError("no JSON-mode records to compare against")

    by_id = {c.comparison_id: c for c in load()}
    keys = sorted(old)[: args.n * 2]

    client = LlamaClient(ServerConfig(base_url=args.base_url, concurrency=args.concurrency))
    if not client.health():
        raise RuntimeError("server not healthy")

    reqs = []
    for cid, order in keys:
        c = by_id.get(cid)
        if c is None:
            continue
        reqs.append((build_request(cid, c.text, c.completion_a, c.completion_b, order,
                                   mode="digit"), (cid, order)))

    t0 = time.time()

    def run_one(pair):
        req, key = pair
        try:
            d = client.chat(req.messages, max_tokens=1, grammar=LABEL_GRAMMAR, top_logprobs=10)
            ch = d["choices"][0]
            content = ch["message"].get("content") or ""
            lp = (ch.get("logprobs") or {}).get("content") or [{}]
            dist = label_distribution_from_logprobs(lp[0].get("top_logprobs", []))
            return key, parse_digit(content), dist
        except Exception as e:
            return key, None, {"error": str(e)[:120]}

    results = client.map(reqs, run_one)
    wall = time.time() - t0

    n = agree_exact = agree_coarse = n_fail = 0
    confusion: collections.Counter[tuple[str, str]] = collections.Counter()
    for key, new_lab, _dist in results:
        if new_lab is None:
            n_fail += 1
            continue
        old_lab = old[key]
        n += 1
        agree_exact += int(new_lab == old_lab)
        agree_coarse += int(COARSE[new_lab] == COARSE[old_lab])
        confusion[(COARSE[old_lab], COARSE[new_lab])] += 1

    exact = agree_exact / max(n, 1)
    coarse = agree_coarse / max(n, 1)
    adopt = coarse >= BAR

    out = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_compared": n,
        "n_failures": n_fail,
        "decision_bar_coarse_agreement": BAR,
        "exact_agreement_5pt": exact,
        "coarse_agreement_ATB": coarse,
        "confusion_old_to_new_coarse": {f"{a}->{b}": v for (a, b), v in confusion.most_common()},
        "wall_s": round(wall, 1),
        "calls_per_s": round(len(results) / max(wall, 1e-9), 3),
        "completion_tokens": client.usage.completion_tokens,
        "prompt_tokens": client.usage.prompt_tokens,
        "adopt_fast_mode": adopt,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print(f"compared n={n} (failures {n_fail})")
    print(f"  exact 5-point agreement : {exact:.3%}")
    print(f"  coarse A/T/B agreement  : {coarse:.3%}   (bar {BAR:.0%})")
    print(f"  confusion: {out['confusion_old_to_new_coarse']}")
    print(f"  throughput: {out['calls_per_s']:.2f} calls/s  ({client.usage.completion_tokens} completion tokens total)")
    print(f"  DECISION: {'ADOPT fast mode' if adopt else 'KEEP JSON mode'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
