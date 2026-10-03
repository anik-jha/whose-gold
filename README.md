# Whose Gold?

Code, results and per-call judge outputs for **Whose Gold? Annotator-Pool Disagreement Is
Large at the Item Level and Invisible in a Six-Model Leaderboard** (Anik Jha), at the
NeurIPS 2026 workshop *Human-AI Coevolution: Measuring Human-Agent Teams in the Agentic
Era*. The camera-ready paper is [`paper/whose_gold.pdf`](paper/whose_gold.pdf), built from
`paper/main.tex`.

## What is here

| Path | Contents |
|---|---|
| `paper/` | LaTeX source, figures and the scripts that draw them, the number verifier, the camera-ready PDF |
| `src/wsev/` | Corpus loading, the LLM judge client, preference-label handling, and the disagreement, ranking and judge-control statistics |
| `experiments/haic/` | One script per analysis, each writing a JSON results file under `runs/` |
| `runs/` | The results files the paper's numbers are read from, every judge call (`runs/c3/*/raw.jsonl`), and a label-only MultiPref derivative (`runs/c8/labels.jsonl`) |
| `tests/` | Unit tests on synthetic fixtures; no downloads |
| `provenance/` | Excerpts of the decision-rule ledger and the first committed C2 script |
| `scripts/reproduce.py` | Replays the analyses and compares them with the recorded results |

The analyses, by results file:

| Script | Results file | In the paper |
|---|---|---|
| `h0_gate_multipref.py` | `runs/h0/eval_multipref.json` | Corpus gate: contested-subset and unanimity bars on MultiPref |
| `c2_whose_gold.py` | `runs/c2/eval.json` | MultiPref pool divergence and leaderboards: Table 2, Figure 1 |
| `c4_generalize_mtbench.py` | `runs/c4/eval.json` | MT-Bench replication: Table 2, Figure 1 |
| `c5_robustness.py` | `runs/c5/robustness.json` | Tie-break conventions: Table 3 |
| `c6_displacement_threshold.py` | `runs/c6/eval.json` | Displacement bootstrap: Section 4, Figure 2, Appendix A |
| `c3_judge_pools.py`, `c3_table2.py` | `runs/c3/` | LLM judges: Section 5, Table 4 |
| `c3_mode_equivalence.py`, `c3_hf_judge.py` | `runs/c3/mode_equivalence.json`, `runs/c3/gate-qwen27/` | Output-format and scoring-shortcut checks: Section 5 |
| `c7_ipw_judge_pools.py` | `runs/c7/eval.json` | Judge estimates reweighted to the corpus: Table 4, Figure 3 |
| `c8_judge_convention_control.py` | `runs/c8/eval.json` | Convention control: Table 5, Appendix B |
| `cluster_audit.py` | (prints) | Prompt-cluster versus row resampling for C6 |

## Reproducing

Python 3.13, with `poppler-utils` and either `tectonic` or `latexmk` to build the paper.

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest                              # unit tests on synthetic fixtures
python scripts/reproduce.py                   # offline: replay C8, run the paper's number verifier
python scripts/reproduce.py --from-corpora    # also rerun every analysis that reads the public corpora
bash paper/build.sh                           # verify, compile, and check the page limit
```

`scripts/reproduce.py` writes replays under `runs/_replay/` and compares each with its
recorded results file, ignoring only each run's timestamp and wall-clock time. `--from-corpora` downloads MultiPref
and the MT-Bench human judgments through `datasets` on first use. The recorded results were
produced on aarch64 Linux with the versions in `requirements.txt`; another platform or
library version can move bootstrap results in the last digits, so inspect a mismatch rather
than widening a tolerance.

`paper/verify_paper_numbers.py` checks 106 of the numbers in `paper/main.tex`, and every row
of the judge table in `runs/c3/table2.json`, against the results files that produced them. `paper/make_figure.py`, `make_figure2.py` and
`make_figure3.py` redraw the three figures from `runs/`.

Regenerating the judge calls themselves needs one of the three GGUF checkpoints named in
Appendix B of the paper, served by `llama.cpp` with the settings given there:

```bash
python experiments/haic/c3_judge_pools.py --tag my-rerun --base-url http://127.0.0.1:8081
```

It writes to `runs/c3/<tag>/`, so use a new tag to keep the recorded calls; `c3_table2.py`
then adds that tag to the judge table. `c3_hf_judge.py` also needs `torch` and
`transformers`.

## Pre-registration

Decision rules were kept in a versioned plan ledger in a private working repository.
`provenance/` holds exact excerpts of the relevant ledger commits and the first committed C2
script, so the change of stopping rule described in the paper's Setup can be checked. Git
timestamps are author-controlled records, not third-party registration.

## Data and licenses

No third-party prompt or response text is included. Dataset attribution, licenses and the
transformations applied are in [`ASSETS.md`](ASSETS.md). The code is MIT-licensed
([`LICENSE`](LICENSE)).

## Citation

```bibtex
@inproceedings{jha2026whosegold,
  title     = {Whose Gold? Annotator-Pool Disagreement Is Large at the Item Level and Invisible in a Six-Model Leaderboard},
  author    = {Jha, Anik},
  booktitle = {NeurIPS 2026 Workshop on Human-AI Coevolution: Measuring Human-Agent Teams in the Agentic Era},
  year      = {2026}
}
```
