"""cortec.plots: figures in the paper's style, from a RunRecord.

The rc parameters are the paper's (paper/make_figures.py): one type scale, constrained layout,
no top or right spine, a light grid behind the marks, DejaVu Sans, and the validated palette
assigned in fixed order. `figure_for(record, path)` draws the figure a record's stage calls for
and returns the path, or None when the stage has no figure. matplotlib is optional: the import
happens inside the functions, and `RunRecord.save()` records a missing matplotlib instead of failing.
"""
from __future__ import annotations

from .report import GRID, INK, INK2, MUTED, PALETTE, SERIES, RunRecord

BASE = 9.0


def paper_rcparams(base: float = BASE) -> dict:
    """The paper's rc parameters, for callers drawing their own figures."""
    return {
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.05, "figure.constrained_layout.w_pad": 0.05,
        "figure.constrained_layout.hspace": 0.04, "figure.constrained_layout.wspace": 0.04,
        "font.family": "DejaVu Sans", "font.size": base,
        "axes.titlesize": base, "axes.titleweight": "bold", "axes.titlepad": 6,
        "axes.labelsize": base - 0.5, "axes.labelcolor": INK, "axes.labelpad": 4,
        "xtick.labelsize": base - 1.5, "ytick.labelsize": base - 1.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.labelweight": "normal",
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.55, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": base - 1.5,
        "legend.handletextpad": 0.5, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.4,
        "figure.facecolor": "white", "axes.facecolor": "white", "lines.solid_capstyle": "round",
    }


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:      # pragma: no cover
        raise ImportError("cortec.plots needs matplotlib: pip install matplotlib") from e
    return plt


def _series_colour(i: int) -> str:
    return PALETTE[SERIES[i % len(SERIES)]]


def _title(fig, record: RunRecord) -> None:
    fig.suptitle(f"{record.stage}: {record.title} ({record.schema})", fontsize=BASE + 1.5,
                 fontweight="bold", color=INK)


def _bar_labels(ax, bars, decimals=3):
    for b in bars:
        h = b.get_height()
        if h == h:      # not NaN
            ax.annotate(f"{h:.{decimals}f}", (b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=BASE - 2.5,
                        color=INK2)


def figure_for(record: RunRecord, path: str) -> str | None:
    """Draw the figure for `record`'s stage into `path` (PNG). None when there is no figure."""
    drawer = {"Evaluation": _fig_evaluation, "Stage C": _fig_bound, "Stage A": _fig_release,
              "Correction": _fig_correction}.get(record.stage)
    if drawer is None:
        return None
    plt = _mpl()
    with plt.rc_context(paper_rcparams()):
        fig = drawer(plt, record)
        if fig is None:
            return None
        fig.savefig(path)
        plt.close(fig)
    return path


# ── Evaluation: fidelity and utility beside the references ───────────────────────────
def _fig_evaluation(plt, record: RunRecord):
    t = record.table("fidelity and utility")
    if t is None:
        return None
    ci = {c: j for j, c in enumerate(t.columns)}
    synth = [r for i, r in enumerate(t.rows) if i not in t.reference_rows]
    refs = [t.rows[i] for i in t.reference_rows]
    ceiling = refs[0] if refs else None
    floor = refs[1] if len(refs) > 1 else None
    students = [c for c in ("TSTR-LR", "TSTR-RF", "TSTR-GBM") if c in ci]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.9), gridspec_kw={"width_ratios": [1, 1.6]})
    # left: 1-way TV
    xs = list(range(len(synth)))
    bars = ax1.bar(xs, [r[ci["1-way TV"]] for r in synth], width=0.62,
                   color=[_series_colour(i) for i in xs], zorder=3)
    _bar_labels(ax1, bars)
    # the two reference lines can coincide (permuting the target changes one marginal only), so
    # the ceiling is labelled above its line and the floor below, at opposite ends of the axis
    for ref, ls, lab, va, dy, x in ((ceiling, (0, (4, 3)), "real sample", "bottom", 2, 1.0),
                                    (floor, (0, (1, 2.5)), "permuted floor", "top", -2, 0.0)):
        if ref is not None:
            ax1.axhline(ref[ci["1-way TV"]], color=INK2, lw=0.9, ls=ls, zorder=2)
            ax1.annotate(lab, (x, ref[ci["1-way TV"]]), xycoords=("axes fraction", "data"),
                         xytext=(-2 if x else 2, dy), textcoords="offset points",
                         ha="right" if x else "left", va=va, fontsize=BASE - 2.5, color=INK2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(r[0]) for r in synth], rotation=0)
    ax1.set_xlim(-0.75, len(synth) - 0.25)
    ax1.set_ylabel("1-way TV against the holdout (lower is better)")
    ax1.set_title("Fidelity")
    ax1.xaxis.grid(False)
    # right: TSTR AUC per student, one bar per synthetic table, references as marks per student
    k = max(len(synth), 1)
    w = 0.7 / k
    for i, r in enumerate(synth):
        pos = [j + (i - (k - 1) / 2) * w for j in range(len(students))]
        bars = ax2.bar(pos, [r[ci[s]] for s in students], width=w * 0.92, color=_series_colour(i),
                       zorder=3, label=str(r[0]))
        _bar_labels(ax2, bars)
    for ref, ls, lab in ((ceiling, (0, (4, 3)), "real sample"), (floor, (0, (1, 2.5)), "permuted floor")):
        if ref is not None:
            for j, s in enumerate(students):
                ax2.hlines(ref[ci[s]], j - 0.42, j + 0.42, color=INK2, lw=1.1, ls=ls, zorder=4,
                           label=lab if j == 0 else None)
    ax2.set_xticks(range(len(students)))
    ax2.set_xticklabels([s.replace("TSTR-", "") for s in students])
    ax2.set_ylabel("TSTR AUC on the holdout (higher is better)")
    ax2.set_title("Utility")
    ax2.xaxis.grid(False)
    lo = min([r[ci[s]] for r in synth + refs for s in students if r[ci[s]] == r[ci[s]]] or [0.4])
    ax2.set_ylim(max(0.0, lo - 0.08), 1.0)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(4, k + 2))
    _title(fig, record)
    return fig


# ── Stage C: the bound per condition ─────────────────────────────────────────────────
def _fig_bound(plt, record: RunRecord):
    t = record.table("bound per condition")
    if t is None:
        return None
    ci = {c: j for j, c in enumerate(t.columns)}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = [str(r[0]) for r in t.rows]
    x = range(len(t.rows))
    cols = [PALETTE["blue"] if i not in t.reference_rows else MUTED for i in x]
    b1 = ax.bar([i - 0.19 for i in x], [r[ci["worst bound"]] for r in t.rows], width=0.36,
                color=cols, zorder=3, label="worst-case bound")
    b2 = ax.bar([i + 0.19 for i in x], [r[ci["mean bound"]] for r in t.rows], width=0.36,
                color=cols, alpha=0.45, zorder=3, label="mean bound")
    _bar_labels(ax, b1); _bar_labels(ax, b2)
    tol = record.value("tolerance")
    if tol is not None:
        ax.axhline(tol, color=PALETTE["magenta"], lw=1.3, ls=(0, (5, 3)), zorder=2)
        ax.annotate(f"tolerance {tol}", (1.0, tol), xycoords=("axes fraction", "data"),
                    xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom",
                    fontsize=BASE - 2.5, color=PALETTE["magenta"])
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("one-sided bound on the conditional gap")
    ax.xaxis.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _title(fig, record)
    return fig


# ── Stage A: cohorts and the finest conditional level ────────────────────────────────
def _fig_release(plt, record: RunRecord):
    cohorts = record.table("cohorts")
    cond = next((t for t in record.tables if t.name.startswith("conditional table")), None)
    if cohorts is None and cond is None:
        return None
    n = int(cohorts is not None) + int(cond is not None)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 1.2, 3.6), squeeze=False)
    axes = list(axes[0])
    if cohorts is not None:
        ax = axes.pop(0)
        ci = {c: j for j, c in enumerate(cohorts.columns)}
        x = range(len(cohorts.rows))
        bars = ax.bar(list(x), [r[ci["positive rate"]] or 0.0 for r in cohorts.rows], width=0.62,
                      color=PALETTE["blue"], zorder=3)
        _bar_labels(ax, bars)
        ax.set_xticks(list(x)); ax.set_xticklabels([str(r[0]) for r in cohorts.rows], rotation=30, ha="right")
        ax.set_xlim(-0.75, len(cohorts.rows) - 0.25)
        ax.set_ylabel("released positive rate"); ax.set_title("Cohorts"); ax.xaxis.grid(False)
    if cond is not None:
        ax = axes.pop(0)
        ci = {c: j for j, c in enumerate(cond.columns)}
        x = range(len(cond.rows))
        bars = ax.bar(list(x), [r[ci["rate"]] for r in cond.rows], width=0.62, color=PALETTE["blue"], zorder=3)
        _bar_labels(ax, bars)
        ax.set_xticks(list(x)); ax.set_xticklabels([str(r[0]) for r in cond.rows], rotation=35, ha="right")
        ax.set_xlim(-0.75, len(cond.rows) - 0.25)
        ax.set_ylabel("released P(positive | cell)"); ax.set_title(cond.name); ax.xaxis.grid(False)
    _title(fig, record)
    return fig


# ── cortec-hybrid: before and after ──────────────────────────────────────────────────
def _fig_correction(plt, record: RunRecord):
    before, after = record.value("conditional error before"), record.value("conditional error after")
    if before is None or after is None:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    bars = ax.bar([0, 1], [before, after], width=0.6, color=[MUTED, PALETTE["blue"]], zorder=3)
    _bar_labels(ax, bars)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["before", "after"])
    ax.set_ylabel("conditional error against the released table")
    shift = record.value("feature marginal shift")
    ax.set_title(f"feature marginal shift {shift:.3f}" if shift is not None else "correction")
    ax.xaxis.grid(False)
    _title(fig, record)
    return fig
