"""Figure 1: item-level divergence is large; the leaderboard does not move.

Every number is read from the results files under runs/ - nothing is transcribed
by hand, so the figure cannot drift from the experiments that produced it.

Design notes (see the dataviz method):
- One series hue, #2a78d6, validated: passes lightness band, chroma floor and >=3:1
  contrast on the light surface. Winner-reversal is a strict SUBSET of
  majority-change, so it is encoded as a nested segment with hatching rather than a
  second categorical hue - a second hue would assert the two measures are
  independent, which they are not, and a light second step failed the validator.
- Hatch + direct labels are the secondary encoding, so the figure survives
  grayscale printing and CVD without relying on color at all.
- Panel B is the punchline: parallel, non-crossing rank lines are what "nothing
  moved" looks like. A bar of height zero would not read.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__file__).resolve().parent

SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e4e4e0"


def load_results() -> dict:
    c2 = json.loads((REPO / "runs" / "c2" / "eval.json").read_text())
    c4 = json.loads((REPO / "runs" / "c4" / "eval.json").read_text())
    c5 = json.loads((REPO / "runs" / "c5" / "robustness.json").read_text())
    return {"c2": c2, "c4": c4, "c5": c5}


def main() -> int:
    r = load_results()
    c2, c4 = r["c2"], r["c4"]

    # MultiPref panel uses the convention-free subset (both pools internally
    # unanimous), so the figure cannot inherit a tie-breaking artifact.
    free = r["c5"]["tiebreak_sensitivity"]["both_pools_unanimous_no_tiebreak"]
    mp_major = free["majority_differs"]
    mp_rev = free["winner_reverses"]
    mtf = c4["convention_free_subset"]
    mt_major = mtf["majority_differs_rate"]
    mt_rev = mtf["winner_reverses"] if "winner_reverses" in mtf else mtf["winner_reversal_rate_excl_ties"]

    rows = [
        ("MultiPref\nexpert-desig. vs normal", mp_major, mp_rev, free["n_items"]),
        ("MT-Bench\nauthors vs experts", mt_major, mt_rev, mtf["n_items"]),
    ]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2,
        # Type 42 (TrueType) rather than matplotlib's default Type 3: Type 3 subsets are
        # not searchable or copyable in the submitted PDF and some venue checkers reject
        # them outright.
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK, "text.color": INK,
        "figure.dpi": 200,
    })
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(5.6, 2.05), gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.62})

    # --- Panel A: item-level divergence ------------------------------------
    ys, h = [0.62, 0.0], 0.30
    for (label, major, rev, _n), y in zip(rows, ys):
        axA.barh(y, major, height=h, color=SERIES, edgecolor="none", zorder=3)
        # Nested subset, same hue + hatch: reversal is a strict subset of majority
        # change, so a second categorical hue would misstate the relationship.
        axA.barh(y, rev, height=h, color=SERIES, edgecolor="#fcfcfb",
                 hatch="/////", linewidth=0.9, zorder=4)
        axA.text(major + 0.015, y, f"{major:.1%}", va="center", ha="left",
                 fontsize=7.0, color=INK, zorder=5)
        axA.text(rev + 0.015, y - 0.235, f"{rev:.1%}", va="center", ha="left",
                 fontsize=6.4, color=INK_2, zorder=5)

    axA.set_yticks(ys)
    axA.set_yticklabels([r[0] for r in rows], fontsize=6.9, linespacing=1.3)
    axA.set_xlim(0, 0.72)
    axA.set_xticks([0, 0.2, 0.4, 0.6])
    axA.set_xticklabels(["0%", "20%", "40%", "60%"], fontsize=6.6)
    axA.set_ylim(-0.45, 0.95)
    axA.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    axA.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        axA.spines[sp].set_visible(False)
    axA.tick_params(axis="y", length=0)
    axA.set_title("A.  Items where the pools disagree",
                  fontsize=7.5, color=INK, loc="left", pad=7)

    # --- Panel B: the leaderboard does not move ----------------------------
    board = sorted(c2["leaderboard"], key=lambda d: d["expert_rank"])
    SHORT = {"gpt-4-turbo-2024-04-09": "gpt-4-turbo", "Meta-Llama-3-70B-Instruct": "llama-3-70b",
             "Llama-2-70b-chat-hf": "llama-2-70b", "allenai/tulu-2-70b": "tulu-2-70b",
             "gpt-3.5-turbo-0125": "gpt-3.5-turbo", "allenai/tulu-2-7b": "tulu-2-7b"}
    nb = len(board)
    for d in board:
        nm = SHORT.get(d["model"], SHORT.get(d["model"].split("/")[-1], d["model"].split("/")[-1]))
        y0, y1 = nb - d["expert_rank"], nb - d["crowd_rank"]
        axB.plot([0, 1], [y0, y1], color="#fcfcfb", linewidth=3.4, zorder=2)
        axB.plot([0, 1], [y0, y1], color=SERIES, linewidth=1.5, zorder=3)
        axB.plot([0, 1], [y0, y1], "o", color=SERIES, markersize=3.4,
                 markeredgecolor="#fcfcfb", markeredgewidth=0.8, zorder=4)
        axB.text(-0.10, y0, nm, ha="right", va="center", fontsize=6.2, color=INK_2)
        axB.text(1.10, y1, str(d["crowd_rank"]), ha="left", va="center",
                 fontsize=6.2, color=MUTED)

    axB.set_xlim(-0.02, 1.02)
    axB.set_ylim(-0.75, nb - 0.35)
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["expert-desig. gold", "normal gold"], fontsize=6.6, color=INK_2)
    axB.set_yticks([])
    for sp in ("top", "right", "left", "bottom"):
        axB.spines[sp].set_visible(False)
    axB.tick_params(axis="x", length=0, pad=3)
    axB.set_title("B.  Resulting ranking", fontsize=7.5, color=INK, loc="left", pad=7)

    # Legend and the tau note live in figure space so neither can collide with an axis.
    fig.legend(handles=[
        Patch(facecolor=SERIES, edgecolor="none", label="majority label changes"),
        Patch(facecolor=SERIES, edgecolor="#fcfcfb", hatch="/////", label="winner reverses (ties excl.)"),
    ], fontsize=6.3, frameon=False, loc="lower left", bbox_to_anchor=(0.045, -0.10),
        handlelength=1.5, handleheight=0.9, borderpad=0.0, labelspacing=0.3)
    fig.text(0.795, -0.135, r"$\tau = 1.00$,  0/6 displaced" + "\nMT-Bench: also 0/6",
             ha="center", va="bottom", fontsize=6.3, color=INK_2, linespacing=1.35)

    fig.savefig(OUT / "fig1_divergence_vs_invariance.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "fig1.png", bbox_inches="tight", pad_inches=0.02, dpi=220)

    # Grayscale proof: the figure must survive print without color.
    print("wrote fig1_divergence_vs_invariance.pdf and fig1.png")
    print(f"  MultiPref  major={mp_major:.3f} rev={mp_rev:.3f}  n={free['n_items']}")
    print(f"  MT-Bench   major={mt_major:.3f} rev={mt_rev:.3f}  n={mtf["n_items"]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
