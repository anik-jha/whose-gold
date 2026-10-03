"""Figure 2: the whole leaderboard risk sits at one adjacent pair, and the Gaussian
route misses it in the same direction everywhere.

Every number is read from runs/c6/eval.json - nothing is transcribed by hand, so the
figure cannot drift from the experiment that produced it.

Design notes (see the dataviz method):
- Form is EMPHASIS, not categorical: the measured rate is the subject and the Gaussian
  prediction is context, so one validated hue (#2a78d6, passes lightness band, chroma
  floor and >=3:1 contrast on the light surface as a 1-slot palette) carries the
  measurement and the de-emphasis gray carries the foil. A second categorical hue would
  assert two equal series; these are a measurement and a discredited approximation.
- Identity never rests on color: each row is named on the axis, the two marks differ in
  shape (filled circle vs open square), and both are direct-labelled. The figure reads
  in grayscale and under any CVD.
- Rows are ordered by gap so the reader sees swap probability collapse as the gap opens,
  which is the section's claim, with the 0.8pp row at the top where the eye lands.
"""
from __future__ import annotations
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
# NeurIPS prohibits Type 3 fonts. matplotlib emits them by default, so both figure
# scripts must set this; make_figure.py already did and these two did not, which put
# five Type 3 faces into the submission PDF.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(__file__).resolve().parent
SERIES, GRAY = "#2a78d6", "#8a8a85"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e4e0"

SHORT = {"gpt-4-turbo-2024-04-09": "gpt-4-turbo",
         "meta-llama/Meta-Llama-3-70B-Instruct": "llama-3-70b",
         "meta-llama/Llama-2-70b-chat-hf": "llama-2-70b",
         "allenai/tulu-2-70b": "tulu-2-70b",
         "gpt-3.5-turbo-0125": "gpt-3.5-turbo",
         "allenai/tulu-2-7b": "tulu-2-7b"}


def main() -> int:
    c6 = json.loads((REPO / "runs" / "c6" / "eval.json").read_text())
    # widest gap at the bottom, closest at the top: the eye lands on the row that carries
    # the result, and it leaves the lower right empty for the legend.
    pairs = sorted(c6["bootstrap"]["measured_adjacent_pairs"], key=lambda r: -r["gap"])
    labels = [f"{SHORT[r['above']]} / {SHORT[r['below']]}" for r in pairs]
    gaps = [r["gap"] * 100 for r in pairs]
    meas = [r["p_disagree_measured"] * 100 for r in pairs]
    gauss = [r["p_swap_gaussian"] * 100 for r in pairs]
    y = list(range(len(pairs)))

    fig, ax = plt.subplots(figsize=(6.6, 2.05))
    for i, (m, g) in enumerate(zip(meas, gauss)):
        ax.plot([m, g], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
    ax.scatter(gauss, y, s=52, facecolors="none", edgecolors=GRAY, lw=1.6,
               marker="s", zorder=3, label="Gaussian route (discredited)")
    ax.scatter(meas, y, s=60, color=SERIES, zorder=4, label="measured by bootstrap")

    for i, (m, g) in enumerate(zip(meas, gauss)):
        ax.annotate(f"{m:.1f}%", (m, i), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.5, color=INK)
        # When the two marks coincide (the 11.1pp row, both ~0) one label above the other
        # would overprint, so the foil's label drops below its mark.
        dy = -12 if abs(m - g) < 1.5 else 7
        ax.annotate(f"{g:.0f}%", (g, i), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=7.5, color=INK_2)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}\n{gp:.1f}pp apart" for lab, gp in zip(labels, gaps)],
                       fontsize=7.5, color=INK)
    ax.set_xlabel("pair ordered differently by the two golds: bootstrap frequency / Gaussian prediction",
                  fontsize=8, color=INK_2)
    ax.set_xlim(-2, 46)
    ax.set_ylim(-0.6, len(pairs) - 0.15)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.tick_params(axis="x", labelsize=7.5, colors=INK_2, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=SERIES, markersize=7,
                              label="measured by bootstrap"),
                       Line2D([], [], marker="s", ls="", mfc="none", mec=GRAY, mew=1.6,
                              markersize=7, label="Gaussian route (discredited)")],
              loc="lower right", frameon=False, bbox_to_anchor=(1.0, -0.02), fontsize=7.5, labelcolor=INK_2,
              handletextpad=0.4, borderpad=0.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig2_swap_by_gap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2.png", dpi=200, bbox_inches="tight")
    print("wrote fig2_swap_by_gap.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
