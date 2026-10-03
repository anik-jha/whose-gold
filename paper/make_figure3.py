"""Figure 3: every judge leans toward the normal pool, and how far depends on the
estimator you designate primary.

Numbers are read from runs/c7/eval.json and runs/c3/table2.json - nothing is
transcribed by hand, so the figure cannot drift from the experiments.

Design notes (see the dataviz method):
- A forest plot is the form for "identity plus interval". Nine point estimates with
  95% CIs against a reference line is exactly what a table renders badly.
- Form is EMPHASIS. The corpus-reweighted estimator is the one the paper designates
  primary, so it alone carries the validated hue (#2a78d6: passes lightness band,
  chroma floor and >=3:1 contrast as a 1-slot palette). The two secondary estimators
  are the de-emphasis gray. A second and third categorical hue were rejected: the
  lighter blue step fails the chroma floor and the contrast check, and three hues here
  would assert three equal readings when the paper's point is that one is primary.
- Identity is never color-alone: every row is named on the axis, the three estimators
  use different markers, and each point estimate is direct-labelled. Grayscale and CVD
  safe by construction.
- The zero line does the significance encoding. An interval that crosses it is visibly
  unresolved; no asterisk is needed and none is used.
"""
from __future__ import annotations
import json, pathlib
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

JUDGES = [("qwen36-35b-a3b", "Qwen3.6-35B-A3B"),
          ("qwen36-27b-dense", "Qwen3.6-27B"),
          ("gemma4-26b-a4b", "Gemma-4-26B-A4B")]
# (label, marker, colour, filled) - corpus first because it is the primary reading
ROWS = [("corpus (primary)", "o", SERIES, True),
        ("sample as drawn", "s", GRAY, False),
        ("self-consistent", "^", GRAY, False)]


def main() -> int:
    c7 = json.loads((REPO / "runs" / "c7" / "eval.json").read_text())["judges"]
    t2 = json.loads((REPO / "runs" / "c3" / "table2.json").read_text())

    def estimates(key):
        j = c7[key]
        return [(j["corpus_ipw"]["agree_diff"], j["corpus_ipw"]["agree_ci95"]),
                (j["sample"]["agree_diff"], j["sample"]["agree_ci95"]),
                (t2[key]["diff"], t2[key]["ci95"])]

    fig, ax = plt.subplots(figsize=(6.6, 2.55))
    ax.axvline(0, color=INK_2, lw=1.0, zorder=2)
    # Classic forest layout: each judge gets a header ROW of its own in the label column,
    # then its three estimator rows. Annotating the name inside the plot overprinted the
    # data, which is what a header row exists to avoid.
    yticks, ylabels, bold_rows, y = [], [], [], 0.0
    for ji, (key, jname) in enumerate(JUDGES):
        yticks.append(y); ylabels.append(jname); bold_rows.append(len(ylabels) - 1)
        y -= 0.72
        for (est, ci), (rlab, marker, col, filled) in zip(estimates(key), ROWS):
            lo, hi = ci
            ax.plot([lo * 100, hi * 100], [y, y], color=col, lw=2, zorder=3,
                    solid_capstyle="butt", alpha=1.0 if filled else 0.85)
            for cap in (lo, hi):
                ax.plot([cap * 100, cap * 100], [y - 0.13, y + 0.13], color=col, lw=1.4, zorder=3)
            ax.scatter([est * 100], [y], s=58, marker=marker, zorder=4,
                       color=col if filled else "none",
                       edgecolors=col, linewidths=0 if filled else 1.6)
            # Value labels live in a right-hand column, the forest-plot convention, and
            # not above the marker. A row is ~10pt tall here, so the 8pt vertical offset an
            # earlier version used put each label on the NEXT ROW UP: the primary blue row
            # read "-2.6" when its own estimate is -2.2, on a headline judge number.
            ax.annotate(f"{est*100:+.1f}", (1.015, y), xycoords=("axes fraction", "data"),
                        ha="left", va="center", fontsize=7,
                        color=INK if filled else INK_2)
            yticks.append(y); ylabels.append(rlab)
            y -= 1.0
        if ji < len(JUDGES) - 1:
            ax.axhline(y + 0.5, color=GRID, lw=0.8, zorder=1)
        y -= 0.45

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.5, color=INK_2)
    for i, lab in enumerate(ax.get_yticklabels()):
        if i in bold_rows:
            lab.set_fontweight("bold"); lab.set_color(INK); lab.set_fontsize(8)
    ax.set_ylim(y + 0.6, 0.9)
    ax.set_xlabel("agreement with expert-designated pool minus agreement with normal pool "
                  "(pp)", fontsize=8, color=INK_2)
    ax.tick_params(axis="x", labelsize=7.5, colors=INK_2, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    # Every estimate is negative, so the right of the zero line is empty; labelling an
    # empty region is noise. One cue, clear of the line.
    ax.annotate("$\\leftarrow$ leans toward the normal pool", (-0.25, 1.03),
                xycoords=("data", "axes fraction"), ha="right", va="bottom",
                fontsize=7.5, color=INK_2)
    ax.legend(handles=[Line2D([], [], marker=m, ls="", markersize=7, color=c,
                              mfc=c if f else "none", mec=c, mew=0 if f else 1.6, label=l)
                       for l, m, c, f in ROWS],
              loc="lower left", frameon=False, fontsize=7.5, labelcolor=INK_2,
              handletextpad=0.4, borderpad=0.2, ncol=3, columnspacing=1.2,
              bbox_to_anchor=(0.0, -0.30))
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig3_judge_forest.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3.png", dpi=200, bbox_inches="tight")
    print("wrote fig3_judge_forest.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
