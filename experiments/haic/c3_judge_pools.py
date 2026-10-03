"""C3: which annotator pool does an LLM judge track?

Context. C2 showed the choice of pool does not move the aggregate leaderboard
(tau=1.0000) while flipping the winner on 16.8% of individual items. So the
question that matters is what happens to things that consume PER-ITEM labels.
LLM-as-judge is the most widely deployed such consumer, and "our judge agrees
with humans X% of the time" is the claim it is validated by.

If a judge tracks one pool materially more than the other, that headline number
is not a property of the judge - it is a property of who was hired to annotate.

PRE-REGISTERED (runs/plan.json, milestone c3-judge-pools):
    KILL if the judge's agreement with the expert majority and with the crowd
    majority are within a paired bootstrap 95% CI of each other AND the erasure
    rates against the two pools are indistinguishable. Report the null.

Every item is judged in BOTH presentation orders; see wsev.judges.preference for
why that is load-bearing rather than a nicety.

Writes runs/c3/<tag>/raw.jsonl (resumable) and runs/c3/<tag>/eval.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from wsev.data.multipref import COARSE, load  # noqa: E402
from wsev.inference.client import LlamaClient, ServerConfig  # noqa: E402
from wsev.judges.preference import (  # noqa: E402
    LABEL_GRAMMAR,
    PREFERENCE_SCHEMA_COMPACT,
    build_request,
    combine_orders,
    parse,
    parse_digit,
    unswap,
)
from wsev.stats.disagreement import defensible_set, majority_vote  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
TIE_BREAK = ("A", "T", "B")


def paired_diff_ci(x: np.ndarray, y: np.ndarray, *, n_boot: int = 5000,
                   alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap CI on mean(x) - mean(y); same items resampled for both."""
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(x.mean() - y.mean()), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="short model tag, e.g. qwen36-35b-a3b")
    ap.add_argument("--base-url", default="http://127.0.0.1:8081")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all items")
    ap.add_argument("--sample", type=int, default=0,
                    help="stratified subsample size (0 = no subsampling). Oversamples "
                         "pool-split items, where the signal lives, and records the strata.")
    ap.add_argument("--split-frac", type=float, default=0.6,
                    help="fraction of the sample drawn from pool-split items")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--mode", default="json", choices=("json", "digit", "verbose"),
                    help="judge output format. Single-token 'digit' failed the 95% "
                         "equivalence bar at 47.2%; 'json' is the validated default and "
                         "MUST match across all models or the cross-family comparison is invalid.")
    args = ap.parse_args()

    outdir = REPO / "runs" / "c3" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "raw.jsonl"

    cfg = ServerConfig(base_url=args.base_url, concurrency=args.concurrency)
    client = LlamaClient(cfg)
    if not client.health():
        raise RuntimeError(f"server not healthy at {args.base_url}")
    served = client.model_name()

    comps = [c for c in load() if c.labels("expert") and c.labels("normal")]
    if args.limit:
        comps = comps[: args.limit]

    strata: dict[str, str] = {}
    if args.sample and args.sample < len(comps):
        # Stratify on whether the two pools' majorities already differ. Power to detect
        # pool alignment comes almost entirely from those items, so sampling them
        # proportionally would waste most of the compute on items that cannot discriminate.
        split, agree = [], []
        for c in comps:
            me = majority_vote([COARSE[x] for x in c.labels("expert")], tie_break=TIE_BREAK)
            mc = majority_vote([COARSE[x] for x in c.labels("normal")], tie_break=TIE_BREAK)
            (split if me != mc else agree).append(c)
        rng = np.random.default_rng(args.seed)
        n_split = min(len(split), int(round(args.sample * args.split_frac)))
        n_agree = min(len(agree), args.sample - n_split)
        pick = ([split[i] for i in rng.choice(len(split), n_split, replace=False)]
                + [agree[i] for i in rng.choice(len(agree), n_agree, replace=False)])
        strata = {c.comparison_id: ("pool_split" if i < n_split else "pool_agree")
                  for i, c in enumerate(pick)}
        comps = pick
        print(f"stratified sample: {n_split} pool-split + {n_agree} pool-agree = {len(comps)}")

    by_id = {c.comparison_id: c for c in comps}

    # Resume: skip (id, order) pairs already on disk.
    done: set[tuple[str, str]] = set()
    if raw_path.exists():
        with raw_path.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["comparison_id"], r["order"]))
                except json.JSONDecodeError:
                    continue
    print(f"{len(comps)} comparisons; {len(done)} judgements already on disk")

    todo = [
        build_request(c.comparison_id, c.text, c.completion_a, c.completion_b, order,
                      max_chars=args.max_chars, mode=args.mode)
        for c in comps
        for order in ("ab", "ba")
        if (c.comparison_id, order) not in done
    ]

    t0 = time.time()
    fh = raw_path.open("a")

    digit_mode = args.mode == "digit"

    def run_one(req):
        try:
            if digit_mode:
                d = client.chat(req.messages, max_tokens=1, grammar=LABEL_GRAMMAR)
                content = d["choices"][0]["message"].get("content") or ""
                label = parse_digit(content)
            else:
                d = client.chat(req.messages, max_tokens=args.max_tokens,
                                json_schema=PREFERENCE_SCHEMA_COMPACT)
                content = d["choices"][0]["message"].get("content") or ""
                label = parse(content)
            return {"comparison_id": req.comparison_id, "order": req.order,
                    "content": content, "label_raw": label}
        except Exception as e:  # a dead item must not kill a multi-hour sweep
            return {"comparison_id": req.comparison_id, "order": req.order,
                    "content": "", "label_raw": None, "error": str(e)[:200]}

    n_written = 0

    def sink(rec):
        nonlocal n_written
        fh.write(json.dumps(rec) + "\n")
        n_written += 1
        if n_written % 25 == 0:
            fh.flush()
            el = time.time() - t0
            rate = n_written / max(el, 1e-9)
            left = (len(todo) - n_written) / max(rate, 1e-9)
            print(f"  {n_written}/{len(todo)}  {rate:.2f} calls/s  eta {left/60:.1f} min", flush=True)

    if todo:
        client.map(todo, run_one, on_result=sink)
    fh.close()

    # --- analysis over everything on disk ---
    per_item: dict[str, dict[str, str | None]] = collections.defaultdict(dict)
    n_parse_fail = n_error = 0
    with raw_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error"):
                n_error += 1
            lab = r.get("label_raw")
            if lab is None:
                n_parse_fail += 1
                per_item[r["comparison_id"]][r["order"]] = None
            else:
                per_item[r["comparison_id"]][r["order"]] = unswap(lab, r["order"])

    agree_exp: list[float] = []
    agree_crd: list[float] = []
    erase_exp: list[float] = []
    erase_crd: list[float] = []
    n_inconsistent = n_inconsistent_coarse = n_scored = 0
    judge_dist: collections.Counter[str] = collections.Counter()
    pool_split_agree = {"expert": 0, "crowd": 0, "neither": 0}

    for cid, orders in per_item.items():
        c = by_id.get(cid)
        if c is None or "ab" not in orders or "ba" not in orders:
            continue
        label, consistent = combine_orders(orders["ab"], orders["ba"])
        if label is None:
            continue
        if not consistent:
            n_inconsistent += 1
        # Direction disagreement is what indicates position bias. Strict label
        # disagreement also fires when the two orders differ only in STRENGTH
        # (clearly- vs slightly-better), which is not a position artefact.
        if orders["ab"] is not None and orders["ba"] is not None and \
           COARSE[orders["ab"]] != COARSE[orders["ba"]]:
            n_inconsistent_coarse += 1
        n_scored += 1
        jc = COARSE[label]
        judge_dist[jc] += 1

        exp = [COARSE[x] for x in c.labels("expert")]
        crd = [COARSE[x] for x in c.labels("normal")]
        me = majority_vote(exp, tie_break=TIE_BREAK)
        mc = majority_vote(crd, tie_break=TIE_BREAK)

        agree_exp.append(1.0 if jc == me else 0.0)
        agree_crd.append(1.0 if jc == mc else 0.0)
        erase_exp.append(1.0 if (jc in defensible_set(exp) and jc != me) else 0.0)
        erase_crd.append(1.0 if (jc in defensible_set(crd) and jc != mc) else 0.0)

        if me != mc:  # the 16.8%-style items where the pools actually part ways
            if jc == me:
                pool_split_agree["expert"] += 1
            elif jc == mc:
                pool_split_agree["crowd"] += 1
            else:
                pool_split_agree["neither"] += 1

    ax, ay = np.array(agree_exp), np.array(agree_crd)
    ex, ey = np.array(erase_exp), np.array(erase_crd)
    d_agree, lo_a, hi_a = paired_diff_ci(ax, ay)
    d_erase, lo_e, hi_e = paired_diff_ci(ex, ey)
    killed = (lo_a <= 0.0 <= hi_a) and (lo_e <= 0.0 <= hi_e)

    split_n = sum(pool_split_agree.values())
    out = {
        "milestone": "c3-judge-pools",
        "tag": args.tag,
        "judge_output_mode": args.mode,
        "served_model": served,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(time.time() - t0, 1),
        "n_comparisons": len(comps),
        "n_scored": n_scored,
        "n_parse_failures": n_parse_fail,
        "n_request_errors": n_error,
        "position_inconsistency_rate_strict": n_inconsistent / max(n_scored, 1),
        "position_inconsistency_rate_direction": n_inconsistent_coarse / max(n_scored, 1),
        "strata": dict(collections.Counter(strata.values())) if strata else None,
        "judge_label_distribution_coarse": dict(judge_dist),
        "agreement_with_expert_majority": float(ax.mean()) if len(ax) else None,
        "agreement_with_crowd_majority": float(ay.mean()) if len(ay) else None,
        "agreement_diff_expert_minus_crowd": d_agree,
        "agreement_diff_ci95": [lo_a, hi_a],
        "erasure_vs_expert": float(ex.mean()) if len(ex) else None,
        "erasure_vs_crowd": float(ey.mean()) if len(ey) else None,
        "erasure_diff_expert_minus_crowd": d_erase,
        "erasure_diff_ci95": [lo_e, hi_e],
        "on_pool_split_items": {
            **pool_split_agree,
            "n": split_n,
            "expert_share": pool_split_agree["expert"] / max(split_n, 1),
            "crowd_share": pool_split_agree["crowd"] / max(split_n, 1),
        },
        "usage": {
            "prompt_tokens": client.usage.prompt_tokens,
            "completion_tokens": client.usage.completion_tokens,
            "n_calls": client.usage.n_calls,
            "n_retries": client.usage.n_retries,
        },
        "verdict": {"killed_by_prereg_bar": killed},
    }
    (outdir / "eval.json").write_text(json.dumps(out, indent=2))

    print(f"\n[{args.tag}] served={served}  scored={n_scored:,}  parse_fail={n_parse_fail}  err={n_error}")
    print(f"  position inconsistency : strict {out['position_inconsistency_rate_strict']:.3%}"
          f"   direction {out['position_inconsistency_rate_direction']:.3%}")
    print(f"  judge label dist       : {dict(judge_dist)}")
    print(f"  agreement expert       : {out['agreement_with_expert_majority']:.4f}")
    print(f"  agreement crowd        : {out['agreement_with_crowd_majority']:.4f}")
    print(f"  diff (exp-crd)         : {d_agree:+.4f}  95% CI [{lo_a:+.4f}, {hi_a:+.4f}]")
    print(f"  erasure exp / crd      : {out['erasure_vs_expert']:.4f} / {out['erasure_vs_crowd']:.4f}"
          f"   diff {d_erase:+.4f} CI [{lo_e:+.4f}, {hi_e:+.4f}]")
    if split_n:
        print(f"  on {split_n} pool-split items: expert {out['on_pool_split_items']['expert_share']:.1%}"
              f" / crowd {out['on_pool_split_items']['crowd_share']:.1%}"
              f" / neither {pool_split_agree['neither']/split_n:.1%}")
    print(f"  PRE-REGISTERED: {'KILLED (null)' if killed else 'SURVIVES'}")
    print(f"wrote {outdir/'eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
