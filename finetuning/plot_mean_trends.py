"""Trend plots for the negative-ratio sweep and the dataset-scaling study (report 21).

Both use the seed-averaged Dev-selected all-relations BioRED F1 (the report-21 headline) on the
y-axis, the mixture axis on the x-axis, and one line per dataset, so the general trend across
mixtures is visible. Means and std come from the same multi-seed runs behind ``21_MEAN.md``
(via ``collect_mean.gather``); error bars are +/- one std across the available seeds.

  * Negative-ratio sweep: x = training NoRelation ratio (1:1 / 1:2 / 1:3 / 1:all); the 1:1 anchor
    is each dataset's base row. Lines: Real BioRED (394) and Synthetic 8B (394) plus the four QC
    subsets (QC dedup, QC dedup NoRel, Real subset, Real subset NoRel).
  * Dataset scaling: x = training paper count (100 / 200 / 300 / 394); n394 is the base row.
    Lines: Real BioRED and Synthetic 8B.
  * Data-replacement swap on QC subsets: x = swap fraction (0 / 25 / 50 / 75 / 100 %); 0% is the
    real-subset base row and 100% the real-Dev QC synthetic endpoint. Lines: the QC dedup (212)
    and QC dedup NoRel (112) gradients.
  * All swap gradients on a common % axis: the full-394 gradient (Real BioRED -> Synthetic 8B,
    report 13; swap-100/200/300 = ~25/51/76 %) plus the two QC-subset gradients, so gradients of
    different absolute size are directly comparable.

Usage::

    nix develop . -c python finetuning/plot_mean_trends.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from finetuning.collect_mean import COLUMNS, _val, gather  # noqa: E402

PREFIX = "report_"

NEGRATIO_OUT = REPO_ROOT / "finetuning" / f"{PREFIX}negratio_trend_mean.png"
SCALING_OUT = REPO_ROOT / "finetuning" / f"{PREFIX}dataset_scaling_trend_mean.png"
QCSWAP_OUT = REPO_ROOT / "finetuning" / f"{PREFIX}qcswap_gradient_mean.png"
SWAP_ALL_OUT = REPO_ROOT / "finetuning" / f"{PREFIX}swap_gradients_pct_mean.png"

# Real-text datasets are drawn solid, synthetic dashed, so the "negatives help real, not
# synthetic" split reads off the line style alone.
RATIOS: list[tuple[str, str]] = [("1", "1:1"), ("2", "1:2"), ("3", "1:3"), ("all", "1:all")]
PAPER_COUNTS: list[int] = [100, 200, 300, 394]


def mean_std(data: dict, key: tuple) -> tuple[float | None, float]:
    """Seed-averaged all-rels BioRED F1 mean and std for one experiment key."""
    runs = data.get(key, {})
    vals = [v for s, _ in COLUMNS if (v := _val(runs, s, "all_pairs", "biored_f1")) is not None]
    if not vals:
        return None, 0.0
    return statistics.fmean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def series(data: dict, keys: list[tuple], xs: list) -> tuple[list, list[float], list[float]]:
    """Filter to points that have a mean; return aligned (x, mean, std) lists."""
    x_out: list = []
    means: list[float] = []
    stds: list[float] = []
    for x, key in zip(xs, keys):
        mean, std = mean_std(data, key)
        if mean is not None:
            x_out.append(x)
            means.append(mean)
            stds.append(std)
    return x_out, means, stds


def plot_negratio(data: dict) -> None:
    """Negative-ratio sweep: F1 vs training NoRelation ratio, one line per dataset."""
    # (label, key-builder over ratio code, marker+style, colour).
    lines = [
        ("Real BioRED (394)", lambda r: ("real", 394, r), "o-", "#d62728"),
        ("Real subset (212)", lambda r: ("qc", "real_subset", r), "s-", "#ff7f0e"),
        ("Real subset NoRel (112)", lambda r: ("qc", "real_subset_norel", r), "^-", "#8c564b"),
        ("Synthetic 8B (394)", lambda r: ("synth8b", 394, r), "o--", "#1f77b4"),
        ("QC dedup (212)", lambda r: ("qc", "dedup", r), "s--", "#2ca02c"),
        ("QC dedup NoRel (112)", lambda r: ("qc", "dedup_norel", r), "^--", "#9467bd"),
    ]
    x_idx = list(range(len(RATIOS)))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, key_of, fmt, colour in lines:
        keys = [key_of(code) for code, _ in RATIOS]
        xs, means, stds = series(data, keys, x_idx)
        ax.errorbar(xs, means, yerr=stds, fmt=fmt, color=colour, label=label, capsize=3, markersize=6, linewidth=1.6)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([lab for _, lab in RATIOS])
    ax.set_xlabel("Relation Mixture (Relations : NoRelation)")
    ax.set_ylabel("BioRED F1 score on real BioRED Test")
    ax.set_title("BioRED F1 scores: Mixture of Relation to NoRelation samples in training dataset")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(NEGRATIO_OUT, dpi=150)
    print(f"wrote {NEGRATIO_OUT}")


def plot_scaling(data: dict) -> None:
    """Dataset scaling: F1 vs training paper count, one line per source."""
    lines = [
        ("Real BioRED", lambda n: ("real", n, "1"), "o-", "#d62728"),
        ("Synthetic 8B", lambda n: ("synth8b", n, "1"), "o--", "#1f77b4"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, key_of, fmt, colour in lines:
        keys = [key_of(n) for n in PAPER_COUNTS]
        xs, means, stds = series(data, keys, PAPER_COUNTS)
        ax.errorbar(xs, means, yerr=stds, fmt=fmt, color=colour, label=label, capsize=3, markersize=6, linewidth=1.6)
    ax.set_xticks(PAPER_COUNTS)
    ax.set_xlabel("Training papers (n)")
    ax.set_ylabel("Dev-selected all-rels BioRED F1 (real BioRED Test)")
    ax.set_title("Dataset scaling: BioRED F1 vs training paper count\n(multi-seed mean +/- std; 1:1 negatives)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(SCALING_OUT, dpi=150)
    print(f"wrote {SCALING_OUT}")


def plot_qcswap(data: dict) -> None:
    """Data-replacement swap on QC subsets: F1 vs swap fraction, one line per QC arm.

    The gradient is single-axis under real-Dev selection: 0% = the real-subset base row, 25/50/75%
    = the qcswap runs, 100% = the real-Dev QC synthetic endpoint. The 0% and 100% baselines are the
    natural line endpoints, drawn as datapoints like the intermediate fractions.
    """
    # (label, key at swap fraction, marker+style, colour); fractions are 0/25/50/75/100 percent.
    lines = [
        (
            "QC dedup (212): real subset -> QC dedup",
            {0: ("qc", "real_subset", "1"), 25: ("qcswap", "dedup", 25), 50: ("qcswap", "dedup", 50),
             75: ("qcswap", "dedup", 75), 87.5: ("qcswap", "dedup", 87.5), 93.75: ("qcswap", "dedup", 93.75), 96.875: ("qcswap", "dedup", 96.875), 100: ("qc", "dedup_realdev", "1")},
            "s-", "#2ca02c",
        ),
        (
            "QC dedup NoRel (112): real subset -> dedup_norel",
            {0: ("qc", "real_subset_norel", "1"), 25: ("qcswap", "dedup_norel", 25), 50: ("qcswap", "dedup_norel", 50),
             75: ("qcswap", "dedup_norel", 75), 100: ("qc", "dedup_norel_realdev", "1")},
            "^-", "#9467bd",
        ),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for label, key_at, fmt, colour in lines:
        fr = sorted(key_at)  # each line plots its own fractions (dedup adds 87.5)
        keys = [key_at[f] for f in fr]
        xs, means, stds = series(data, keys, fr)
        ax.errorbar(xs, means, yerr=stds, fmt=fmt, color=colour, label=label, capsize=3, markersize=7, linewidth=1.6)
    ax.set_xticks([0, 25, 50, 75, 87.5, 93.75, 96.875, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "87.5", "93.75", "96.875", "100"], rotation=45, fontsize=8)
    ax.set_xlabel("Synthetic text swapped in (% of QC-subset papers)")
    ax.set_ylabel("Dev-selected all-rels BioRED F1 (real BioRED Test)")
    ax.set_title("Data-replacement swap on QC subsets: BioRED F1 vs swap fraction\n(multi-seed mean +/- std; real-Dev selection; 0% = real subset, 100% = QC synthetic)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(QCSWAP_OUT, dpi=150)
    print(f"wrote {QCSWAP_OUT}")


def plot_swap_all(data: dict) -> None:
    """All abstract-swap gradients on one % axis: full 394 (report 13) plus the two QC subsets.

    The x-axis is the fraction of a dataset's papers whose text is swapped from real to synthetic,
    so the three gradients (different absolute sizes) are directly comparable. The full-394
    gradient's swap-100/200/300 land at 100/200/300 of 394 = ~25/51/76 %. Endpoints are datapoints:
    0% = the real endpoint, 100% = the synthetic endpoint of each gradient.
    """
    p = 100.0 / 394.0  # swap-N of 394 papers as a percentage
    # (label, {pct: key}, marker+style, colour).
    lines = [
        (
            "Full 394: Real BioRED -> Synthetic 8B",
            {0: ("real", 394, "1"), 100 * p: ("swap", 100, "1"), 200 * p: ("swap", 200, "1"),
             300 * p: ("swap", 300, "1"), 345 * p: ("swap", 345, "1"), 369 * p: ("swap", 369, "1"), 382 * p: ("swap", 382, "1"), 100: ("synth8b", 394, "1")},
            "o-", "#333333",
        ),
        (
            "QC dedup (212): real subset -> QC dedup",
            {0: ("qc", "real_subset", "1"), 25: ("qcswap", "dedup", 25), 50: ("qcswap", "dedup", 50),
             75: ("qcswap", "dedup", 75), 87.5: ("qcswap", "dedup", 87.5), 93.75: ("qcswap", "dedup", 93.75), 96.875: ("qcswap", "dedup", 96.875), 100: ("qc", "dedup_realdev", "1")},
            "s-", "#2ca02c",
        ),
        (
            "QC dedup NoRel (112): real subset -> dedup_norel",
            {0: ("qc", "real_subset_norel", "1"), 25: ("qcswap", "dedup_norel", 25), 50: ("qcswap", "dedup_norel", 50),
             75: ("qcswap", "dedup_norel", 75), 100: ("qc", "dedup_norel_realdev", "1")},
            "^-", "#9467bd",
        ),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, key_at, fmt, colour in lines:
        pcts = sorted(key_at)
        keys = [key_at[pc] for pc in pcts]
        xs, means, stds = series(data, keys, pcts)
        ax.errorbar(xs, means, yerr=stds, fmt=fmt, color=colour, label=label, capsize=3, markersize=7, linewidth=1.6)
    ax.set_xticks([0, 25, 50, 75, 87.5, 93.75, 96.875, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "87.5", "93.75", "96.875", "100"], rotation=45, fontsize=8)
    ax.set_xlabel("Synthetic text swapped in (% of papers)")
    ax.set_ylabel("BioRED F1 score on real BioRED Test")
    ax.set_title("BioRED F1 scores: Replacing real data with synthetic in training dataset")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(SWAP_ALL_OUT, dpi=150)
    print(f"wrote {SWAP_ALL_OUT}")


# --- Per-dataset negative-ratio P/R/F1 plots (scaling_pr_mean.py visual style) -------------------
# One figure per dataset: mean F1 / precision / recall vs the training relation-mixture ratio.
NEGRATIO_RATIOS: list[tuple[str, str]] = [("1", "1:1"), ("2", "1:2"), ("3", "1:3"), ("all", "1:all")]
# metric field -> (colour, linestyle, label); shared "x" marker, matching plot_scaling_pr_mean.py.
PR_STYLE: dict[str, tuple[str, str, str]] = {
    "biored_f1": ("red", "-", "F1"),
    "biored_precision": ("blue", ":", "Precision"),
    "biored_recall": ("green", "--", "Recall"),
}
# Empirical NoRelation-per-relation ratio at each mixture setting, per dataset, so the x-axis
# is to scale instead of categorical. 1:1/1:2/1:3 come out at ~1.0/1.95/2.7-2.8 (the
# distance-capped negative pool plus per-paper rounding pull 1:3 below its nominal 3.0), and
# 1:all is the full co-mentioned unrelated pool with no distance cap, ~6x the relation count.
# Measured once from the training frames (build_train_df / build_qc_arm, NoRel/positive count).
NEGRATIO_X: dict[str, list[float]] = {
    "real": [1.00, 1.95, 2.78, 6.42],
    "synth8b": [1.00, 1.95, 2.78, 6.42],
    "qc_dedup": [1.00, 1.95, 2.77, 6.19],
    "real_subset": [1.00, 1.95, 2.77, 6.19],
    "qc_dedup_norel": [1.00, 1.93, 2.64, 6.15],
    "real_subset_norel": [1.00, 1.93, 2.64, 6.15],
}
# (label, filename slug, ratio-code -> config key).
NEGRATIO_DATASETS: list[tuple[str, str, object]] = [
    ("Real BioRED (394)", "real", lambda r: ("real", 394, r)),
    ("Synthetic (Qwen3-8B, 394)", "synth8b", lambda r: ("synth8b", 394, r)),
    ("QC dedup (212)", "qc_dedup", lambda r: ("qc", "dedup", r)),
    ("Real subset (212)", "real_subset", lambda r: ("qc", "real_subset", r)),
    ("QC dedup NoRel (112)", "qc_dedup_norel", lambda r: ("qc", "dedup_norel", r)),
    ("Real subset NoRel (112)", "real_subset_norel", lambda r: ("qc", "real_subset_norel", r)),
]


def _mean_std_field(data: dict, key: tuple, field: str) -> tuple[float | None, float]:
    """Seed-averaged mean and std of an all-rels field (F1 / precision / recall) for one key."""
    runs = data.get(key, {})
    vals = [v for s, _ in COLUMNS if (v := _val(runs, s, "all_pairs", field)) is not None]
    if not vals:
        return None, 0.0
    return statistics.fmean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)


def plot_negratio_pr(data: dict) -> None:
    """One F1/precision/recall-vs-mixture figure per dataset, in the scaling-plot style."""
    for label, slug, key_of in NEGRATIO_DATASETS:
        x_pos = NEGRATIO_X[slug]  # to-scale NoRel-per-relation ratio for this dataset
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for field, (colour, linestyle, metric_label) in PR_STYLE.items():
            keys = [key_of(code) for code, _ in NEGRATIO_RATIOS]
            pts = [(x, *_mean_std_field(data, k, field)) for x, k in zip(x_pos, keys)]
            xs = [x for x, m, _ in pts if m is not None]
            means = [m for _, m, _ in pts if m is not None]
            stds = [s for _, m, s in pts if m is not None]
            ax.errorbar(
                xs, means, yerr=stds, color=colour, linestyle=linestyle, marker="x",
                markersize=8, linewidth=1.8, capsize=4, elinewidth=1.0, label=metric_label,
            )
        ax.set_title(f"{label}: multi-seed mean all-rels. metric vs relation mixture", fontsize=10)
        ax.set_xlabel("Relation mixture (Relations : NoRelation), x-axis to scale")
        ax.set_ylabel("BioRED score (all relations, Dev-selected)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([lab for _, lab in NEGRATIO_RATIOS])
        ax.set_xlim(0.5, max(x_pos) + 0.5)
        # Common cutoff across all six datasets (data spans ~0.06-0.61 incl. std),
        # so the panels are directly comparable side by side in the report.
        ax.set_ylim(0.0, 0.64)
        ax.grid(True, alpha=0.3)
        # Single-row legend in the empty band at the bottom of the axes (~0.0-0.1),
        # so no extra vertical space is needed outside the plot.
        ax.legend(loc="lower center", ncol=3, fontsize=9)
        fig.tight_layout()
        out = REPO_ROOT / "finetuning" / f"{PREFIX}negratio_pr_{slug}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


# --- Per-dataset swap-gradient P/R/F1 plots (scaling_pr_mean.py visual style) ---------------------
# One figure per swap gradient: mean F1 / precision / recall vs the fraction of papers swapped
# real -> synthetic. Same PR_STYLE colours as the relation-mixture plots.
_P = 100.0 / 394.0  # swap-N of 394 papers as a percentage
# (title label, filename slug, {swap_pct: config key}).
SWAP_PR_DATASETS: list[tuple[str, str, dict]] = [
    ("Full 394 (Real->Synth 8B)", "full394",
     {0: ("real", 394, "1"), 100 * _P: ("swap", 100, "1"), 200 * _P: ("swap", 200, "1"),
      300 * _P: ("swap", 300, "1"), 345 * _P: ("swap", 345, "1"), 369 * _P: ("swap", 369, "1"), 382 * _P: ("swap", 382, "1"), 100.0: ("synth8b", 394, "1")}),
    ("QC dedup (212)", "qc_dedup",
     {0: ("qc", "real_subset", "1"), 25: ("qcswap", "dedup", 25), 50: ("qcswap", "dedup", 50),
      75: ("qcswap", "dedup", 75), 87.5: ("qcswap", "dedup", 87.5), 93.75: ("qcswap", "dedup", 93.75), 96.875: ("qcswap", "dedup", 96.875), 100: ("qc", "dedup_realdev", "1")}),
    ("QC dedup NoRel (112)", "qc_dedup_norel",
     {0: ("qc", "real_subset_norel", "1"), 25: ("qcswap", "dedup_norel", 25), 50: ("qcswap", "dedup_norel", 50),
      75: ("qcswap", "dedup_norel", 75), 100: ("qc", "dedup_norel_realdev", "1")}),
]


def plot_swap_pr(data: dict) -> None:
    """One F1/precision/recall-vs-swap-fraction figure per swap gradient, in the scaling-plot style."""
    for label, slug, key_at in SWAP_PR_DATASETS:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        pcts = sorted(key_at)
        for field, (colour, linestyle, metric_label) in PR_STYLE.items():
            pts = [(p, *_mean_std_field(data, key_at[p], field)) for p in pcts]
            xs = [p for p, m, _ in pts if m is not None]
            means = [m for _, m, _ in pts if m is not None]
            stds = [s for _, m, s in pts if m is not None]
            ax.errorbar(
                xs, means, yerr=stds, color=colour, linestyle=linestyle, marker="x",
                markersize=8, linewidth=1.8, capsize=4, elinewidth=1.0, label=metric_label,
            )
        ax.set_title(f"{label}: multi-seed mean all-rels. metric vs swap fraction", fontsize=10)
        ax.set_xlabel("Synthetic text swapped in (% of papers)")
        ax.set_ylabel("BioRED score (all relations, Dev-selected)")
        ax.set_xticks([0, 25, 50, 75, 87.5, 93.75, 96.875, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "87.5", "93.75", "96.875", "100"], rotation=45, fontsize=8)
        ax.set_xlim(-5, 105)
        # Common cutoff shared with the negratio_pr panels (data spans ~0.15-0.62
        # incl. std), so all P/R/F1 panels in the report are directly comparable.
        ax.set_ylim(0.0, 0.64)
        ax.grid(True, alpha=0.3)
        # Single-row legend in the empty band at the bottom of the axes (~0.0-0.1),
        # so no extra vertical space is needed outside the plot.
        ax.legend(loc="lower center", ncol=3, fontsize=9)
        fig.tight_layout()
        out = REPO_ROOT / "finetuning" / f"{PREFIX}swap_pr_{slug}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def main() -> None:
    data = gather()
    plot_negratio(data)
    plot_scaling(data)
    plot_qcswap(data)
    plot_swap_all(data)
    plot_negratio_pr(data)
    plot_swap_pr(data)


if __name__ == "__main__":
    main()
