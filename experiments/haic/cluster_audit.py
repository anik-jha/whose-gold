"""Audit: does resampling prompt CLUSTERS instead of comparison rows move C6's numbers?

MultiPref reuses each prompt for two comparisons against different model pairs: 10,461 rows
over 5,323 prompts, 5,138 of which appear exactly twice, so 98.2% of rows sit in a repeated
prompt. c6_displacement_threshold.py resamples rows independently, which treats those two
rows as independent observations. If annotation noise is correlated within a prompt, that
understates the resampling spread. This script reruns the same three quantities with the
prompt as the resampling unit and changes nothing else.
"""
from __future__ import annotations
import argparse, collections, json, pathlib, sys
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from wsev.data.multipref import load
from wsev.stats.ranking import Battle, kendall_tau, ranking, win_rates

MB = 50

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "runs" / "c6" / "cluster_audit.json")
    a = ap.parse_args()

    battles, exp_lab, cro_lab, unanimous, pids = [], [], [], [], []
    for c in load():
        e, n = tuple(c.labels("expert")), tuple(c.labels("normal"))
        if not e or not n:
            continue
        battles.append(Battle(c.model_a, c.model_b, tuple(c.labels())))
        exp_lab.append(e); cro_lab.append(n); pids.append(c.prompt_id)
        ce, cn = c.coarse_labels("expert"), c.coarse_labels("normal")
        unanimous.append(len(set(ce)) == 1 and len(set(cn)) == 1)
    n = len(battles)
    unanimous = np.asarray(unanimous)
    be = [Battle(b.model_a, b.model_b, l) for b, l in zip(battles, exp_lab)]
    bc = [Battle(b.model_a, b.model_b, l) for b, l in zip(battles, cro_lab)]

    # cluster index: rows grouped by prompt
    groups = collections.defaultdict(list)
    for i, p in enumerate(pids):
        groups[p].append(i)
    clusters = [np.asarray(v) for v in groups.values()]
    un_clusters = [c[unanimous[c]] for c in clusters]
    un_clusters = [c for c in un_clusters if len(c)]
    print(f"{n} rows, {len(clusters)} prompt clusters; "
          f"unanimous subset {int(unanimous.sum())} rows in {len(un_clusters)} clusters")

    wr_e = win_rates(be, hard=True, min_battles=MB)
    wr_c = win_rates(bc, hard=True, min_battles=MB)
    models = sorted(set(wr_e) & set(wr_c), key=lambda m: -wr_e[m])
    ordered = [wr_e[m] for m in models]
    gaps = [ordered[i] - ordered[i + 1] for i in range(len(ordered) - 1)]
    J = int(np.argmin(gaps))
    print(f"closest adjacent pair: {models[J]} vs {models[J+1]} at {gaps[J]*100:.2f}pp\n")

    def run(cl, label):
        rng = np.random.default_rng(a.seed)
        K = len(cl)
        taus, disp, pair = [], [], []
        for _ in range(a.n_boot):
            pick = rng.integers(0, K, K)
            idx = np.concatenate([cl[p] for p in pick])
            sx = [be[i] for i in idx]; sy = [bc[i] for i in idx]
            we, wc = win_rates(sx, hard=True, min_battles=MB), win_rates(sy, hard=True, min_battles=MB)
            rx, ry = ranking(we), ranking(wc)
            common = set(rx) & set(ry)
            if len(common) < 3:
                continue
            rx = [m for m in rx if m in common]; ry = [m for m in ry if m in common]
            ix = {m: i for i, m in enumerate(rx)}; iy = {m: i for i, m in enumerate(ry)}
            taus.append(kendall_tau(rx, ry))
            disp.append(sum(1 for m in common if ix[m] != iy[m]))
            if set(models) <= common:
                hi, lo = models[J], models[J + 1]
                pair.append((ix[hi] < ix[lo]) != (iy[hi] < iy[lo]))
        d = np.asarray(disp); t = np.asarray(taus)
        r = {"n_boot": len(d), "p_any_displacement": round(float((d > 0).mean()), 4),
             "tau_ci95": [round(float(q), 4) for q in np.quantile(t, [0.025, 0.975])],
             "p_pair_disagree": round(float(np.mean(pair)), 4) if pair else None}
        print(f"{label:22} displacement {100*r['p_any_displacement']:.1f}%   "
              f"tau CI [{r['tau_ci95'][0]:.3f}, {r['tau_ci95'][1]:.3f}]"
              + (f"   closest-pair {100*r['p_pair_disagree']:.1f}%" if pair else ""))
        return r

    out = {"unit": "prompt cluster", "n_boot": a.n_boot, "seed": a.seed,
           "n_rows": n, "n_clusters": len(clusters),
           "closest_pair": [models[J], models[J + 1]], "closest_gap": round(gaps[J], 5),
           "full_corpus": run(clusters, "full corpus"),
           "unanimous_subset": run(un_clusters, "unanimous subset")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out.relative_to(REPO)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
