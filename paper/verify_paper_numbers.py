"""Check the minimal camera-ready's quantitative claims against the results files.

Adapted from the accepted paper's verifier: every check it ran still runs here against
the main.tex beside this file, and the convention-control numbers added for the
camera-ready are checked against runs/c8/eval.json and runs/h0/eval_multipref.json.
Run before every build.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
TEX = (HERE / "main.tex").read_text()
FLAT = re.sub(r"\s+", " ", TEX)  # LaTeX rewraps prose; phrase checks must survive that


def j(p):
    return json.loads((REPO / p).read_text())


fails = []

c2 = j("runs/c2/eval.json")
c4 = j("runs/c4/eval.json")
c5 = j("runs/c5/robustness.json")
c6 = j("runs/c6/eval.json")
c7 = j("runs/c7/eval.json")
t2 = j("runs/c3/table2.json")
me = j("runs/c3/mode_equivalence.json")
pf = j("runs/c3/gate-qwen27/prefill_equivalence.json")
ps = c5["judge_pool_split_three_way"]

un = c5["tiebreak_sensitivity"]["both_pools_unanimous_no_tiebreak"]
cf = c4["convention_free_subset"]
pert = c6["perturbation"]
obs = c6["observed_leaderboard"]
thr = c6["threshold"]
gr = c6["gaussian_route"]
boot = c6["bootstrap"]
ksize = {r["n_models"]: r for r in thr["by_leaderboard_size"]}

CHECKS = [
    ("MultiPref unanimous n", "2{,}885", un["n_items"], 0),
    ("MultiPref majority differs", "23.6", un["majority_differs"] * 100, 0.05),
    ("MultiPref winner reverses", "9.2", un["winner_reverses"] * 100, 0.05),
    ("MT-Bench convention-free n", "246", cf["n_items"], 0),
    ("MT-Bench majority differs", "30.5", cf["majority_differs_rate"] * 100, 0.05),
    ("MT-Bench winner reverses", "8.5", cf["winner_reversal_rate_excl_ties"] * 100, 0.05),
    ("MultiPref tau", "1.0000", c2["kendall_tau_expert_vs_crowd"], 1e-9),
    ("MT-Bench doubly-rated", "292", c4["n_items_both_pools"], 0),
    ("B-skew", "1.78", c5["B_to_A_vote_ratio"], 0.005),
    # C6, the displacement threshold
    ("shift SD", "1.9", pert["sd_of_shift"] * 100, 0.05),
    ("pair-difference SD", "2.6", pert["sd_of_pairwise_shift_difference"] * 100, 0.05),
    ("smallest adjacent gap", "0.8", obs["smallest_adjacent_gap"] * 100, 0.05),
    ("p swap at smallest gap", "38", obs["p_swap_at_smallest_gap"] * 100, 0.6),
    ("bootstrap displacement", "28", boot["p_any_displacement"] * 100, 0.3),
    ("simulated K=6", "0.293", ksize[6]["p_any_displacement"], 5e-4),
    ("bootstrap K=6 crosscheck", "0.279", boot["p_any_displacement"], 6e-4),
    ("K=10 Monte Carlo", "0.92", ksize[10]["p_any_displacement"], 0.005),
    # The claim that the old independent-swap closed form understates K=10 by ~5pp.
    ("K=10 closed-form gap", "5",
     (ksize[10]["p_any_displacement"] - ksize[10]["p_any_displacement_indep_product"]) * 100,
     0.6),
    ("gap for 5% swap", "4.3", abs(thr["gap_for_p_swap_0.05"]) * 100, 0.05),
    # --- added 2026-08-22: the Gaussian swap route is reported as failing, so every number
    # --- in that argument needs a gate of its own, including the ones it is failing against.
    ("measured swap at the 0.8pp pair", "22.6",
     c6["bootstrap"]["measured_adjacent_pairs"][3]["p_disagree_measured"] * 100, 0.05),
    ("largest other measured pair", "5.2",
     max(r["p_disagree_measured"] for i, r in
         enumerate(c6["bootstrap"]["measured_adjacent_pairs"]) if i != 3) * 100, 0.06),
    ("gaussian swap at the 0.8pp pair", "38",
     c6["bootstrap"]["measured_adjacent_pairs"][3]["p_swap_gaussian"] * 100, 0.6),
    ("sd_pair assumed independent", "2.6", gr["sd_pair_assumed_independent"] * 100, 0.05),
    ("sd_pair measured bootstrap", "1.3", gr["sd_pair_measured_bootstrap"] * 100, 0.05),
    ("gaussian union with measured sd", "32", gr["union_with_measured_sd"] * 100, 0.6),
    ("unanimous-subset displacement", "34.7",
     c6["unanimous_subset"]["p_any_displacement"] * 100, 0.06),
    ("simulation on the observed gaps", "0.70",
     c6["calibration_of_the_illustrative_model"]["sim_on_observed_gaps"], 0.005),
    # C3, the judge arm
    ("format self-agreement", "47.2", me["coarse_agreement_ATB"] * 100, 0.05),
    ("format n", "n=500", me["n_compared"], 0),
    ("A->B moves", "146", me["confusion_old_to_new_coarse"]["A->B"], 0),
    ("B->A moves", "3", me["confusion_old_to_new_coarse"]["B->A"], 0),
    ("prefill recovery", "94.6", pf["coarse_agreement_ATB"] * 100, 0.05),
    # C7, the corpus reweighting of the stratified judge sample
    ("corpus pool-split n", "5210", c7["population"]["pool_split"], 0),
    ("corpus pool-agree n", "5251", c7["population"]["pool_agree"], 0),
    ("corpus pool-split frac", "49.8", c7["population"]["pool_split_frac"] * 100, 0.05),
    ("self-consistent kept, worst", "55",
     t2["qwen36-35b-a3b"]["n_clean"] / t2["qwen36-35b-a3b"]["n_items_scored"] * 100, 0.6),
    ("self-consistent kept, best", "76",
     t2["gemma4-26b-a4b"]["n_clean"] / t2["gemma4-26b-a4b"]["n_items_scored"] * 100, 0.6),
]

# Every judge row of Table 4 carries three estimators; all three must trace to a file.
for _tag in ("qwen36-35b-a3b", "qwen36-27b-dense", "gemma4-26b-a4b"):
    _c, _s = c7["judges"][_tag]["corpus_ipw"], c7["judges"][_tag]["sample"]
    CHECKS.append((f"corpus agree {_tag}", f"{_c['agree_diff']:.3f}", _c["agree_diff"], 5e-4))
    CHECKS.append((f"corpus agree CI lo {_tag}", f"{_c['agree_ci95'][0]:.3f}",
                   _c["agree_ci95"][0], 5e-4))
    CHECKS.append((f"corpus agree CI hi {_tag}", f"{_c['agree_ci95'][1]:+.3f}".replace("+", "+"),
                   _c["agree_ci95"][1], 5e-4))
    CHECKS.append((f"sample agree {_tag}", f"{_s['agree_diff']:.3f}", _s["agree_diff"], 5e-4))
    CHECKS.append((f"corpus erasure {_tag}", f"{_c['erasure_diff'] * 100:.1f}",
                   _c["erasure_diff"] * 100, 0.05))

# The tie-break table: every row, not just the unanimous one. Four of the five rows were
# unchecked, which is how a paper that is *about* convention-dependence could quote one
# convention's quality-gap reading as if it were the finding.
_TB_ROWS = [("default_alphabetical_what_c2_ran", "44.9", "29.8"),
            ("force_tie_as_paper_states", "41.2", "2.5"),
            ("B_first_mirror_check", "36.6", "9.4"),
            ("A_over_tie_hypothetical", "49.8", "16.8")]
for _rule, _md, _wr in _TB_ROWS:
    _row = c5["tiebreak_sensitivity"][_rule]
    CHECKS.append((f"tiebreak {_rule} majority", _md, _row["majority_differs"] * 100, 0.05))
    CHECKS.append((f"tiebreak {_rule} reverses", _wr, _row["winner_reverses"] * 100, 0.05))
for _rule, _tau in (("default_alphabetical_what_c2_ran", "1.0000"),
                    ("A_over_tie_hypothetical", "0.4667")):
    CHECKS.append((f"tiebreak tau {_rule}", _tau,
                   c5["leaderboard_tau_by_rule"][_rule]["tau"], 1e-4))

# Quality-gap tertiles, one entry per convention. The paper quotes the convention-free
# subset in prose and the r column of the tie-break table for all five; both come from
# here. This block exists because the generator originally emitted only the
# A_over_tie_hypothetical reading, which is the single convention under which the
# relationship looks flat, and the paper called it flat.
_QG = c5["quality_gap_buckets"]
for _rule, _r in (("unanimous_subset_primary", "-0.154"),
                  ("default_alphabetical_what_c2_ran", "-0.081"),
                  ("force_tie_as_paper_states", "+0.097"),
                  ("B_first_mirror_check", "-0.220"),
                  ("A_over_tie_hypothetical", "+0.030")):
    CHECKS.append((f"quality-gap r {_rule}", _r,
                   _QG[_rule]["pearson_r_gap_vs_divergence"], 0.0006))
for _lab, _v in (("narrow", "30.8"), ("middle", "24.3"), ("wide", "15.3")):
    CHECKS.append((f"quality-gap {_lab} tertile (primary)", _v,
                   _QG["unanimous_subset_primary"]["tertiles"][_lab]["divergence"] * 100, 0.05))

# Position sensitivity and the adjacent-gap list, both quoted in prose and both previously
# unchecked. The gap list had drifted: the paper said 2.8pp where c6 says 2.746.
for _tag, _v in (("qwen36-35b-a3b", "44.6"), ("qwen36-27b-dense", "24.5"),
                 ("gemma4-26b-a4b", "23.7")):
    CHECKS.append((f"position sensitivity {_tag}", _v,
                   j(f"runs/c3/{_tag}/eval.json")["position_inconsistency_rate_direction"] * 100,
                   0.05))
for _i, _v in enumerate(["11.1", "2.9", "2.7", "0.8", "2.5"]):
    CHECKS.append((f"adjacent gap {_i}", _v, obs["adjacent_gaps"][_i] * 100, 0.05))

# The underpowered pilot the paper reports by name. Its eval.json was overwritten by the
# matched-power rerun, so the numbers live in a transcription of the run log; without this
# check they are the one set of quoted values with no machine-readable source at all.
_pilot = j("runs/c3/qwen36-27b-dense/pilot_n1250.json")
CHECKS += [
    ("pilot diff", "-0.8", _pilot["agreement_diff_expert_minus_crowd"] * 100, 0.05),
    ("pilot CI lo", "-4.4", _pilot["agreement_diff_ci95"][0] * 100, 0.05),
    ("pilot CI hi", "+2.8", _pilot["agreement_diff_ci95"][1] * 100, 0.05),
    ("pilot n", "half this", _pilot["n_comparisons"], 1e9),
    ("full-power comparator", "-4.2",
     j("runs/c3/qwen36-27b-dense/eval.json")["agreement_diff_expert_minus_crowd"] * 100, 0.05),
]

# F8: the cross-vendor pool-split three-way breakdown quoted in prose, one row per judge.
# (This is the class of bug that shipped once already: a judge added to runs/c3/ after
# c5_robustness.py's tag loop was last extended, so the paper quoted a hand-estimated
# number instead of one from a results file. Every tag in table2.json must appear here.)
for tag in t2:
    if tag not in ps:
        fails.append(f"MISSING pool-split breakdown for {tag}: not in "
                      f"runs/c5/robustness.json judge_pool_split_three_way "
                      f"(c5_robustness.py's tag loop needs updating)")
        continue
    row = ps[tag]
    CHECKS.append((f"{tag} crowd share", f"{row['crowd_share_of_all']*100:.1f}",
                   row["crowd_share_of_all"] * 100, 0.05))
    CHECKS.append((f"{tag} expert share", f"{row['expert_share_of_all']*100:.1f}",
                   row["expert_share_of_all"] * 100, 0.05))
    CHECKS.append((f"{tag} neither share", f"{row['neither_share_of_all']*100:.1f}",
                   row["neither_share_of_all"] * 100, 0.05))
    CHECKS.append((f"{tag} crowd share of decided", f"{row['crowd_share_of_decided']*100:.1f}",
                   row["crowd_share_of_decided"] * 100, 0.05))

# The primary-estimator (all-items) diffs quoted in the Table 4 caption, one per judge.
# The needle is DERIVED from the results file rather than typed here: it was typed once,
# as -0.043 against a true -0.04247, and the tolerance was set wide enough to accept it.
for _tag in ("qwen36-35b-a3b", "qwen36-27b-dense", "gemma4-26b-a4b"):
    _d = j(f"runs/c3/{_tag}/eval.json")["agreement_diff_expert_minus_crowd"]
    CHECKS.append((f"primary estimator {_tag}", f"{_d:.3f}", _d, 5e-4))

# --- Camera-ready additions -------------------------------------------------------------
# The within-expert split, now stated on its coarse scale, and the per-analysis bootstrap
# sizes that replaced the accepted paper's blanket B=4000.
h0 = j("runs/h0/eval_multipref.json")
c8 = j("runs/c8/eval.json")
JUDGES = ("qwen36-35b-a3b", "qwen36-27b-dense", "gemma4-26b-a4b")
NAMES = {"qwen36-35b-a3b": "Qwen3.6-35B-A3B", "qwen36-27b-dense": "Qwen3.6-27B",
         "gemma4-26b-a4b": "Gemma-4-26B-A4B"}
st = {t: c8["judges"][t]["statistics"] for t in JUDGES}
CHECKS += [
    ("within-expert coarse split", "50.1", h0["expert_pair_split_fraction"] * 100, 0.05),
    ("leaderboard tau bootstrap B", "$B=2000$ (leaderboard", c2["n_bootstrap"], 0),
    ("displacement bootstrap B", "$B=4000$ (displacement", boot["n_boot"], 0),
    ("control replicates", "$5{,}000$ replicates", c8["n_boot"], 0),
    ("control seed", "seed $20261002$", c8["seed"], 0),
    ("Holm six-test p, dense Qwen agreement", "adjusted $p = 0.016$",
     st["qwen36-27b-dense"]["A_T_B"]["p_holm_six"], 5e-4),
    ("Holm three-test p, dense Qwen per annotator", "Holm-adjusted $p = 0.010$",
     st["qwen36-27b-dense"]["per_annotator"]["p_holm_three"], 5e-4),
    ("dense Qwen per-annotator lean", "($-2.5$pp, Holm-adjusted",
     st["qwen36-27b-dense"]["per_annotator"]["estimate"] * 100, 0.05),
]
# The quality-gap correlation range quoted in prose must span every convention's r. It
# shipped once as "-0.220 to +0.030" while the table beside it showed +0.097.
_rs = [_QG[r]["pearson_r_gap_vs_divergence"] for r in
       ("unanimous_subset_primary", "default_alphabetical_what_c2_ran",
        "force_tie_as_paper_states", "B_first_mirror_check", "A_over_tie_hypothetical")]
_range = f"runs from ${min(_rs):.3f}$ to ${max(_rs):+.3f}$"
if _range not in re.sub(r"\s+", " ", TEX):
    fails.append(f"MISMATCH quality-gap range: expected {_range!r}")

# The self-consistent column's bootstrap settings are not in its results file; they are the
# recorded-generation defaults of c3_table2.py, which reproduce runs/c3/table2.json exactly.
_src = (REPO / "experiments/haic/c3_table2.py").read_text()
if not (re.search(r'"--n-boot", type=int, default=10000', _src)
        and re.search(r'"--seed", type=int, default=17', _src)):
    fails.append("MISMATCH self-consistent bootstrap settings: c3_table2.py defaults changed")
if "$B=10{,}000$ (self-consistent judge estimates, seed $17$)" not in TEX:
    fails.append("MISSING self-consistent bootstrap settings in Appendix B")

CHECKS = [(lab, re.sub(r"\s+", " ", needle), val, tol) for lab, needle, val, tol in CHECKS]

for _t in JUDGES:
    if c8["judges"][_t]["rows_in_repeated_prompts"] != 604:
        fails.append(f"MISMATCH repeated-prompt rows for {_t}: paper says 604")
_n_split = {c8["judges"][t]["n_unanimous_split"] for t in JUDGES}
if _n_split != {214}:
    fails.append(f"MISMATCH unanimous-and-disagree count: paper says 214, file says {_n_split}")
_m = {t: c8["judges"][t]["unanimous_split_matches"] for t in JUDGES}
_phrase = ("pool ${}$, ${}$ and ${}$ times and with the normal pool ${}$, ${}$ and ${}$ times"
           .format(*[_m[t]["expert"] for t in JUDGES], *[_m[t]["normal"] for t in JUDGES]))
if _phrase not in FLAT:
    fails.append(f"MISSING unanimous-split counts: expected {_phrase!r}")


def _signed(v):
    return f"{v:+.1f}".replace("+-", "-")


# The order-reversal range in the recommendations must span all three judges. It shipped
# as 24.5--44.6% from the two Qwen judges after Gemma (23.7%) was added.
_pos = [j(f"runs/c3/{t}/eval.json")["position_inconsistency_rate_direction"] * 100 for t in JUDGES]
_pos_range = f"At ${min(_pos):.1f}$--${max(_pos):.1f}\\%$ direction reversal"
if _pos_range not in FLAT:
    fails.append(f"MISMATCH order-reversal range: expected {_pos_range!r}")

# The two format-check disagreement rates, derived from the agreement rates they complement.
_fmt = (f"on ${100 * (1 - pf['coarse_agreement_ATB']):.1f}\\%$ of items, against "
        f"${100 * (1 - me['coarse_agreement_ATB']):.1f}\\%$ for the JSON-versus-digit condition")
if _fmt not in FLAT:
    fails.append(f"MISMATCH format-check disagreement rates: expected {_fmt!r}")

# Per-annotator intervals reported in Appendix B, in table order.
for _t in JUDGES:
    _s = st[_t]["per_annotator"]
    _cell = (f"${_signed(100 * _s['estimate'])}$ "
             f"$[{100 * _s['ci95'][0]:+.2f}, {100 * _s['ci95'][1]:+.2f}]$")
    if _cell not in FLAT:
        fails.append(f"MISMATCH per-annotator interval for {_t}: expected {_cell!r}")


# Every row of the convention table is rebuilt from runs/c8/eval.json and must appear
# verbatim, so a hand-edited cell or a stale star fails.
_COLS = ("A_T_B", "A_B_T", "B_T_A", "force_tie", "per_annotator", "excess_stratified_marginal")
for _t in JUDGES:
    cells = []
    for c in _COLS:
        s = st[_t][c]
        lo, hi = s["ci95"]
        star = "^{*}" if lo > 0 or hi < 0 else ""
        cells.append(f"${_signed(100 * s['estimate'])}{star}$")
    row = f"{NAMES[_t]} & " + " & ".join(cells) + " \\\\"
    if row not in FLAT:
        fails.append(f"MISMATCH convention-table row for {_t}: expected {row!r}")

# Summary table: original-rule leans with the accepted row-bootstrap stars (C7), and the
# per-annotator leans with prompt-resampled stars (C8).
_orig = ", ".join(
    f"${_signed(100 * c7['judges'][t]['corpus_ipw']['agree_diff'])}"
    f"{'^{*}' if c7['judges'][t]['corpus_ipw']['agree_excludes_zero'] else ''}$" for t in JUDGES)
_pa = ", ".join(
    f"${_signed(100 * st[t]['per_annotator']['estimate'])}"
    f"{'^{*}' if (st[t]['per_annotator']['ci95'][0] > 0 or st[t]['per_annotator']['ci95'][1] < 0) else ''}$"
    for t in JUDGES)
for _label, _row in (("original rule", _orig), ("per annotator", _pa)):
    if f"{_row}pp" not in FLAT:
        fails.append(f"MISMATCH summary-table judge row ({_label}): expected {_row}pp")

# Prose claims about the controls, checked as logic rather than as digits.
_holm6 = {(t, m): st[t][m]["p_holm_six"] for t in JUDGES for m in ("A_T_B", "erasure")}
if [k for k, p in _holm6.items() if p < 0.05] != [("qwen36-27b-dense", "A_T_B")]:
    fails.append("CLAIM: 'Holm leaves only Qwen3.6-27B's agreement difference' no longer holds")
_pa_sig = [t for t in JUDGES if st[t]["per_annotator"]["p_holm_three"] < 0.05]
_pa_ci = [t for t in JUDGES if st[t]["per_annotator"]["ci95"][1] < 0 or st[t]["per_annotator"]["ci95"][0] > 0]
if _pa_sig != ["qwen36-27b-dense"] or _pa_ci != ["qwen36-27b-dense"]:
    fails.append("CLAIM: per-annotator lean distinguishable only for Qwen3.6-27B no longer holds")
if not st["qwen36-27b-dense"]["excess_stratified_marginal"]["ci95"][1] < 0:
    fails.append("CLAIM: dense Qwen's lean exceeding its marginal expectation no longer holds")
_g = st["gemma4-26b-a4b"]
_share = _g["stratified_marginal"]["estimate"] / _g["A_T_B"]["estimate"]
if not 0.4 <= _share <= 0.6:
    fails.append(f"CLAIM: 'about half of Gemma-4's lean' is marginal; file gives {_share:.2f}")
for _rule in ("A_B_T", "B_T_A"):
    if f"{_g[_rule]['estimate'] * 100:+.1f}" != "+0.5":
        fails.append(f"MISMATCH Gemma-4 sign under {_rule}: paper says +0.5pp")
for _t in JUDGES:
    _c = c7["judges"][_t]["corpus_ipw"]
    for _m, _key in (("A_T_B", "agree"), ("erasure", "erasure")):
        lo, hi = st[_t][_m]["ci95"]
        if (lo > 0 or hi < 0) != _c[f"{_key}_excludes_zero"]:
            fails.append(f"CLAIM: prompt resampling changes the {_key} conclusion for {_t}")
_shrink = {round(1 - c7["judges"][t]["corpus_ipw"]["agree_diff"] / c7["judges"][t]["sample"]["agree_diff"], 2)
           for t in JUDGES}
if _shrink != {0.17}:
    fails.append(f"MISMATCH reweighting shrinkage: paper says 17%, file gives {_shrink}")

for label, needle, value, tol in CHECKS:
    if needle not in FLAT:
        fails.append(f"MISSING  {label}: '{needle}' absent from main.tex")
        continue
    m = re.search(r"-?\d+\.?\d*", needle.replace("{,}", ""))
    if m and abs(float(m.group(0)) - float(value)) > tol:
        fails.append(f"MISMATCH {label}: paper says {m.group(0)}, results file says {value}")

# Judge arm: every model in table2.json must appear, and each diff must match.
for tag, row in t2.items():
    pct = abs(row["diff"]) * 100
    if f"{pct:.1f}" not in FLAT:
        fails.append(f"MISSING judge diff for {tag}: {pct:.1f}pp not in main.tex")
    if not row.get("replicates"):
        fails.append(f"{tag} no longer clears the pre-registered bar")

# Hard rule: every bibliography entry must be a clickable external link (\href{ or
# \url{), not plain text like "arXiv:XXXX.XXXXX". The in-text citation already jumps to
# the entry via natbib+hyperref's internal cross-reference; that is not enough on its
# own, because the entry itself then has to link out to the actual paper. This
# regressed once silently (6 of 8 entries were dead text with no visible symptom until
# a reader actually clicked one), so it is checked mechanically now, every build.
biblio_match = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", TEX, re.DOTALL)
if not biblio_match:
    fails.append("BIBLIOGRAPHY MISSING: no thebibliography environment found in main.tex")
else:
    for entry in re.split(r"\\bibitem", biblio_match.group(0))[1:]:
        key = (re.search(r"\{([\w:]+)\}", entry) or [None, "<unknown key>"])[1]
        if "\\href{" not in entry and "\\url{" not in entry:
            fails.append(f"DEAD CITATION: bibliography entry '{key}' has no \\href{{}} or "
                         f"\\url{{}} -- renders as non-clickable plain text")

# Both corpora reuse an item across model pairs, so the row bootstrap treats dependent
# rows as independent. This defect was fixed in c6 and then found again in c2, c4 and c5,
# which is why every analysis that resamples now records its unit and every one is checked
# here rather than only the one that was caught first.
for _src, _obj, _want in (("c6 displacement", c6["bootstrap"], "prompt"),
                          ("c2 leaderboard tau", c2, "prompt"),
                          ("c4 MT-Bench tau", c4, "question"),
                          ("c5 tie-break tau", c5, "prompt")):
    _u = _obj.get("resampling_unit", "")
    if _u.split()[:1] != [_want]:
        fails.append(f"RESAMPLING UNIT: {_src} records {_u!r}, expected a {_want} cluster; "
                     f"a row bootstrap here treats dependent rows as independent")

# --- House-style and consistency gate ---------------------------------------------------
# The sibling project gates prose style and this one did not, so 21 em dashes accumulated,
# two of them added during a review pass that was supposed to be removing them. No other
# check here can see punctuation or phrasing: every number stayed correct the whole time.
_body = TEX.split("\\section*{NeurIPS Paper Checklist}")[0]
# Verbatim spans reproduce external text literally, so "---" inside one is three hyphens
# and not an em dash. The judge prompt in the appendix contains exactly that, and it must
# stay literal or the paper misreproduces the prompt it is reproducing. Strip \verb spans
# before the style scan; prose outside them is still checked.
_body = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", _body, flags=re.DOTALL)
_body = re.sub(r"\\verb(.)(.*?)\1", "", _body)

if "---" in _body:
    fails.append(f"HOUSE STYLE: {_body.count('---')} em dash(es) in the main text. The prose "
                 f"is em-dash-free by choice; use a colon, semicolon, parentheses or a "
                 f"sentence break.")
for _bad, _why in (("\u2014", "a literal Unicode em dash (U+2014)"),
                   ("\u2013", "a literal Unicode en dash (U+2013); use --"),
                   ("\u2019", "a literal Unicode right single quote; use '"),
                   ("\u201c", "a literal Unicode left double quote; use ``"),
                   ("\u2026", "a literal Unicode ellipsis; use \\ldots")):
    if _bad in _body:
        fails.append(f"HOUSE STYLE: {_why} in the main text")

# Phrases that read as machine-written filler rather than as argument. Each one is a hedge
# or a connective doing no work; the paper's register is direct and every claim is owned.
_TELLS = ["it is worth noting", "it is important to note", "it should be noted",
          "delve into", "delves into", "a testament to", "plays a crucial role",
          "plays a vital role", "sheds light on", "paves the way", "underscores the",
          "highlights the importance", "in today's", "in the realm of", "a tapestry",
          "the landscape of", "navigate the", "it is crucial to", "it is essential to",
          "furthermore, it", "moreover, it", "in conclusion,", "last but not least",
          "seamlessly", "rich tapestry", "ever-evolving", "at the end of the day",
          # a 28-35% event is not a coin flip; this overstated twice before it was caught
          "coin-flip", "coin flip", "toss-up"]
_low = re.sub(r"\s+", " ", _body).lower()
for _tell in _TELLS:
    if _tell in _low:
        fails.append(f"HOUSE STYLE: filler phrase {_tell!r} in the main text")

# Section 4 twice grew a claim about which way the K>6 extrapolation errs, while the
# paragraph beside it says the two biases oppose and the net direction is unknown. The
# numbers are correct either way, so only an explicit assertion catches the contradiction.
if "do not claim to know the net direction" in re.sub(r"\s+", " ", _body):
    for _dir in ("overestimate", "upper-leaning", "upper leaning", "an over-estimate"):
        if _dir in _low:
            fails.append(f"CONTRADICTION: the paper says the net direction of the K>6 rows "
                         f"is unknown, and also calls them {_dir!r}")

for stale in ["both models we test", "a cross-vendor test is still owed",
              "Aggregate rankings are robust to who annotates",
              "It is flat.", "not concentrated on hard cases"]:
    if stale in FLAT:
        fails.append(f"STALE text still present: '{stale}'")

# abstract.txt is the plain-text abstract pasted into the OpenReview form. It is derived from
# main.tex (--write-abstract) and gated here, because a pasted abstract that disagrees with
# the paper went unnoticed once before.
def detex_abstract(tex: str) -> str:
    a = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL).group(1)
    a = re.sub(r"(?<!\\)%.*", "", a)
    a = a.replace(r"\%", "\x00")
    for pat in (r"\\textsc\{([^{}]*)\}", r"\\emph\{([^{}]*)\}", r"\\textbf\{([^{}]*)\}"):
        for _ in range(3):
            a = re.sub(pat, r"\1", a)
    a = re.sub(r"\\cite[tp]?\*?(\[[^]]*\])*\{[^}]*\}", "", a)
    a = a.replace(r"\tau", "tau")
    a = re.sub(r"\$([^$]*)\$", r"\1", a)
    a = a.replace("{,}", ",").replace(r"\,", "").replace("---", " - ").replace("--", "-")
    a = re.sub(r"\\[a-zA-Z]+\s*", "", a)
    a = a.replace("{", "").replace("}", "").replace("\x00", "%")
    paras = [re.sub(r"\s+", " ", q).strip() for q in a.split("\n\n")]
    return "\n\n".join(q for q in paras if q)

abstract_txt = HERE / "abstract.txt"
if "--write-abstract" in sys.argv:
    abstract_txt.write_text(detex_abstract(TEX) + "\n")
if not abstract_txt.exists() or abstract_txt.read_text().strip() != detex_abstract(TEX):
    fails.append("ABSTRACT STALE: abstract.txt does not match main.tex; rerun with --write-abstract")

print(f"{len(CHECKS)} numeric claims + {len(t2)} judge rows checked against 9 results files")
if fails:
    print("\n".join(fails))
    print(f"\nFAIL: {len(fails)} problem(s)")
    sys.exit(1)
print("PASS: every checked number matches its results file")
sys.exit(0)
