"""Judge with an HF model via one-forward-pass label scoring.

Two roles:
  --validate  score items we already judged with llama.cpp JSON generation, on the
              SAME model weights, and report agreement. This is the gate: the
              prefill-logit shortcut is only usable if it reproduces the generation
              path. The bar is pre-registered at 95% coarse agreement, the same bar
              the bare-digit mode failed at 47.2%.
  (default)   run the sweep for a model llama.cpp cannot serve.

Writes runs/c3/<tag>/raw.jsonl in the same schema as the llama.cpp path, so the
downstream analysis is shared and cannot silently diverge.
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
from wsev.inference.hf_scorer import HFPreferenceScorer  # noqa: E402
from wsev.judges.preference import (  # noqa: E402
    CODE_TO_LABEL,
    build_request,
    unswap,
)
from wsev.stats.disagreement import majority_vote  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
TIE_BREAK = ("A", "T", "B")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sample", type=int, default=2500)
    ap.add_argument("--split-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--validate-against", type=pathlib.Path, default=None,
                    help="raw.jsonl from the llama.cpp generation path; enables gate mode")
    ap.add_argument("--validate-n", type=int, default=300)
    args = ap.parse_args()

    outdir = REPO / "runs" / "c3" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    comps = [c for c in load() if c.labels("expert") and c.labels("normal")]
    by_id = {c.comparison_id: c for c in comps}

    if args.validate_against:
        ref: dict[tuple[str, str], str] = {}
        with args.validate_against.open() as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("label_raw"):
                    ref[(r["comparison_id"], r["order"])] = r["label_raw"]
        keys = sorted(ref)[: args.validate_n * 2]
        targets = [(cid, order) for cid, order in keys if cid in by_id]
    else:
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
        by_id = {c.comparison_id: c for c in pick}
        targets = [(c.comparison_id, o) for c in pick for o in ("ab", "ba")]
        print(f"stratified sample: {n_split} pool-split + {n_agree} pool-agree", flush=True)

    print(f"loading {args.model_path}", flush=True)
    t0 = time.time()
    scorer = HFPreferenceScorer(args.model_path)
    print(f"loaded in {time.time()-t0:.0f}s; code token ids = {scorer.code_ids}", flush=True)

    raw_path = outdir / ("validate.jsonl" if args.validate_against else "raw.jsonl")
    done = set()
    if raw_path.exists():
        with raw_path.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["comparison_id"], r["order"]))
                except json.JSONDecodeError:
                    continue
    targets = [t for t in targets if t not in done]
    print(f"{len(targets)} scorings to do ({len(done)} already on disk)", flush=True)

    t0 = time.time()
    n = 0
    with raw_path.open("a") as fh:
        for i in range(0, len(targets), args.batch_size):
            chunk = targets[i: i + args.batch_size]
            batch = []
            for cid, order in chunk:
                c = by_id[cid]
                req = build_request(cid, c.text, c.completion_a, c.completion_b,
                                    order, max_chars=args.max_chars, mode="json")
                batch.append((cid, order, list(req.messages)))
            for s in scorer.score_batch(batch, max_len=args.max_len):
                fh.write(json.dumps({
                    "comparison_id": s.comparison_id, "order": s.order,
                    "content": json.dumps({"p": s.code}),
                    "label_raw": CODE_TO_LABEL[s.code],
                    "probs": s.probs,
                }) + "\n")
                n += 1
            if (i // args.batch_size) % 10 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"  {n}/{len(targets)}  {n/max(el,1e-9):.2f} it/s  "
                      f"eta {(len(targets)-n)/max(n/max(el,1e-9),1e-9)/60:.1f} min", flush=True)

    if args.validate_against:
        agree_exact = agree_coarse = tot = 0
        conf: collections.Counter[str] = collections.Counter()
        with raw_path.open() as fh:
            for line in fh:
                r = json.loads(line)
                k = (r["comparison_id"], r["order"])
                if k not in ref:
                    continue
                a = unswap(ref[k], r["order"])
                b = unswap(r["label_raw"], r["order"])
                tot += 1
                agree_exact += a == b
                agree_coarse += COARSE[a] == COARSE[b]
                conf[f"{COARSE[a]}->{COARSE[b]}"] += 1
        bar = 0.95
        ok = tot > 0 and agree_coarse / tot >= bar
        out = {
            "check": "prefill-logit vs llama.cpp JSON generation, same weights",
            "n": tot,
            "exact_agreement_5pt": agree_exact / max(tot, 1),
            "coarse_agreement_ATB": agree_coarse / max(tot, 1),
            "bar": bar,
            "confusion": dict(conf.most_common()),
            "adopt_prefill_logit": ok,
        }
        (outdir / "prefill_equivalence.json").write_text(json.dumps(out, indent=2))
        print(f"\nn={tot}  exact {agree_exact/max(tot,1):.1%}  "
              f"coarse {agree_coarse/max(tot,1):.1%}  (bar {bar:.0%})")
        print(f"confusion: {dict(conf.most_common(8))}")
        print(f"DECISION: {'ADOPT prefill-logit' if ok else 'REJECT - do not report cross-vendor numbers'}")
    else:
        print(f"\nwrote {raw_path}  ({n} scorings, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
