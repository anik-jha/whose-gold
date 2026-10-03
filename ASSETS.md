# Assets and licenses

Licenses checked against each dataset's Hugging Face metadata on 2026-10-03.

- **MultiPref.** Miranda et al., *Hybrid Preferences: Learning to Route Instances for Human
  vs. AI Feedback*, https://arxiv.org/abs/2410.19133. Dataset
  https://huggingface.co/datasets/allenai/multipref, revision
  `12910233a0238a997ebe425656e9dfed7b0ff031` (the revision every reported number was
  computed from), licensed under the Open Data Commons Attribution License (ODC-BY) v1.0,
  https://opendatacommons.org/licenses/by/1-0/.
  `runs/c8/labels.jsonl` is a derivative: for each comparison, its comparison and prompt
  IDs and both pools' annotator labels coarsened to A/Tie/B. It contains no prompt or
  response text. ODC-BY governs database rights and does not license every right in the
  underlying content; keep this attribution and the license notice when redistributing
  the derivative.
- **MT-Bench human judgments.** Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena*, https://arxiv.org/abs/2306.05685. Dataset
  https://huggingface.co/datasets/lmsys/mt_bench_human_judgments, revision
  `f7d2896d2cc5d80f8b55c2bbc722613555233c25`, licensed under CC BY 4.0,
  https://creativecommons.org/licenses/by/4.0/. Only derived statistics are included
  (`runs/c4/eval.json`).
- **Judge outputs.** `runs/c3/*/raw.jsonl` and `runs/c3/gate-qwen27/validate.jsonl` hold,
  per call, the comparison ID, presentation order, the judge's raw output and the parsed
  label. They were generated locally with the checkpoints named in Appendix B of the
  paper. No model weights are included; the checkpoints' own terms apply to them.
- **Style file.** `paper/neurips_2026.sty` is the NeurIPS 2026 style file, unchanged.
- **Code.** Anik Jha, 2026, MIT license (`LICENSE`). The dataset licenses above are not a
  license for the code, and the code license does not relicense the dataset derivatives.
