"""Plot the BioRED-style F1 per training epoch from ``epoch_curves.csv``.

Produces, in ``presentation/figures/``:

    * the Test-only curves, one figure per run. Every run is scored on the evaluation set
      that matches its own training data there, which is how the numbers in the deck were
      produced.
    * ``biored_f1_per_epoch_all_runs.png``, the figure embedded in the deck: only the two
      runs trained on the original BioRED training set, both scored on the held-out BioRED
      test set. The runs trained with the other negative sampling are deliberately left
      out, so the legend can name the models alone.
    * the full cross-model comparison, all four runs on the **common** BioRED test set.
      This is the figure that also shows the runs trained with the other negative sampling.
    * the Dev-versus-Test comparison on the common BioRED evaluation set, one figure per
      run plus one combined figure, used to check on which split a checkpoint would have
      been selected.
    * a diagnostic for the perturbated-trained runs: their own evaluation set against the
      common one.

The data comes from ``epoch_curves.py``; this script only reads the CSV, so the plots
can be changed without re-running any model.

Two test variants are drawn where both exist:

    matched_norelation  one sampled NoRelation per gold relation (the balanced split
                        the models were trained on)
    full_norelation     every co-mentioned pair of the evaluated abstracts (the realistic
                        setting; only defined for the original BioRED splits)

Usage::

    nix develop -c python finetuning/plot_epoch_curves.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The figures end up on beamer slides at roughly half their pixel width, so the base font
# has to be large enough to survive that downscaling.
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "legend.fontsize": 11})

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = REPO_ROOT / "finetuning" / "epoch_curves.csv"
FIGURE_DIR = REPO_ROOT / "presentation" / "figures"

ENCODER_NAMES: dict[str, str] = {
    "NeuML/pubmedbert-base-embeddings": "PubMedBERT",
    "bioformers/bioformer-8L": "Bioformer-8L",
}
# Both training sets are BioRED and share 92% of their gold relations; they differ in how
# the NoRelation examples were drawn (91% of the negatives are not shared). The labels name
# that difference, because "perturbed" would suggest corrupted labels, which the models
# never saw: the sample builder drops every label_flip, direction_swap and false_negative row.
DATASET_NAMES: dict[str, str] = {
    "BioRed": "BioRED, distance-matched neg.",
    "BioRedPerturbated": "BioRED, type-restricted neg.",
}
# Legend-sized version of the same labels.
DATASET_SHORT_NAMES: dict[str, str] = {
    "BioRed": "distance-matched neg.",
    "BioRedPerturbated": "type-restricted neg.",
}
# File-name fragments, kept as they were: one of these figures is referenced from the deck.
DATASET_SLUGS: dict[str, str] = {
    "BioRed": "biored",
    "BioRedPerturbated": "perturbed-biored",
}
VARIANT_NAMES: dict[str, str] = {
    "matched_norelation": "matched negatives (1 per gold relation)",
    "full_norelation": "all co-mentioned pairs",
}
VARIANT_STYLES: dict[str, str] = {"matched_norelation": "-", "full_norelation": "--"}
# Dev versus Test: colour carries the split, the line style still carries the test variant.
SPLIT_COLOURS: dict[str, str] = {"Dev": "#B23A1E", "Test": "#2A4D7A"}
SPLIT_ORDER: list[str] = ["Dev", "Test"]
# Rows written before the Dev sweep existed carry no split; they are all Test rows.
DEFAULT_SPLIT = "Test"
# The evaluation set every run is scored on, so the runs can be compared with each other.
COMMON_EVAL_DATASET = "BioRed"
# Own evaluation set versus the common one: colour per run, line style per evaluation set.
EVAL_SET_STYLES: dict[str, str] = {"BioRed": "-", "BioRedPerturbated": ":"}
# Deck colours: accent blue and warn red, one shade per training set.
RUN_COLOURS: dict[tuple[str, str], str] = {
    ("PubMedBERT", "BioRed"): "#2A4D7A",
    ("Bioformer-8L", "BioRed"): "#B23A1E",
    ("PubMedBERT", "BioRedPerturbated"): "#6B9BD1",
    ("Bioformer-8L", "BioRedPerturbated"): "#E0916F",
}
Y_LIMITS = (0.0, 0.75)


def load_curves(csv_file: Path = CSV_FILE) -> pd.DataFrame:
    """Read the sweep CSV and fill in the columns older versions of it did not have."""
    frame = pd.read_csv(csv_file)
    if "split" not in frame.columns:
        frame["split"] = DEFAULT_SPLIT
    frame["split"] = frame["split"].fillna(DEFAULT_SPLIT)
    if "dataset" in frame.columns:
        for column in ("train_dataset", "eval_dataset"):
            if column not in frame.columns:
                frame[column] = frame["dataset"]
            frame[column] = frame[column].fillna(frame["dataset"])
    return frame


def slugify(encoder: str, dataset_raw: str) -> str:
    """Build a file-name fragment from an encoder label and a raw training-set name."""
    slug = DATASET_SLUGS.get(dataset_raw, dataset_raw)
    return f"{encoder}_{slug}".lower().replace(" ", "-").replace("/", "-")


def style_axes(
    axes: plt.Axes,
    title: str,
    epochs: list[int],
    y_label: str = "BioRED-style F1 (pooled over the 8 relation types)",
    y_limits: tuple[float, float] = Y_LIMITS,
) -> None:
    """Apply the shared axis labels, limits, grid and title."""
    axes.set_xlabel("Training epoch")
    axes.set_ylabel(y_label)
    axes.set_title(title)
    axes.set_xticks(epochs)
    axes.set_ylim(*y_limits)
    axes.grid(True, alpha=0.3)


def plot_single_run(frame: pd.DataFrame, encoder: str, dataset_raw: str) -> Path:
    """Draw the per-epoch curve(s) of one run and return the file it was written to."""
    figure, axes = plt.subplots(figsize=(8.0, 4.6))
    dataset = DATASET_NAMES.get(dataset_raw, dataset_raw)
    colour = RUN_COLOURS.get((encoder, dataset_raw), "#2A4D7A")

    for variant, group in frame.groupby("variant", sort=False):
        group = group.sort_values("epoch")
        axes.plot(
            group["epoch"],
            group["biored_f1"],
            marker="o",
            markersize=4,
            linestyle=VARIANT_STYLES.get(variant, "-"),
            color=colour,
            label=VARIANT_NAMES.get(variant, variant),
        )
        # Mark the best epoch, which is the number quoted in the presentation.
        best = group.loc[group["biored_f1"].idxmax()]
        axes.annotate(
            f"{best['biored_f1']:.3f}",
            xy=(best["epoch"], best["biored_f1"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=colour,
        )

    style_axes(axes, f"{encoder} trained on {dataset}: BioRED-style F1 per epoch", sorted(frame["epoch"].unique()))
    axes.legend(title="Test set", loc="lower right", fontsize=10, title_fontsize=10)
    figure.tight_layout()

    output = FIGURE_DIR / f"biored_f1_per_epoch_{slugify(encoder, dataset_raw)}.png"
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def draw_split_curves(axes: plt.Axes, frame: pd.DataFrame, annotate: bool = True) -> None:
    """Draw one run's curves with colour per split and line style per test variant.

    The best epoch of every curve is marked with a star, since that is the epoch a
    selection rule reading this curve would pick.
    """
    for split in SPLIT_ORDER:
        for variant in VARIANT_STYLES:
            group = frame[(frame["split"] == split) & (frame["variant"] == variant)]
            if group.empty:
                continue
            group = group.sort_values("epoch")
            colour = SPLIT_COLOURS.get(split, "#444444")
            suffix = "" if variant == "matched_norelation" else ", all pairs"
            axes.plot(
                group["epoch"],
                group["biored_f1"],
                marker="o",
                markersize=4,
                linestyle=VARIANT_STYLES[variant],
                color=colour,
                label=f"{split}{suffix}",
            )
            best = group.loc[group["biored_f1"].idxmax()]
            axes.plot(
                best["epoch"],
                best["biored_f1"],
                marker="*",
                markersize=13,
                linestyle="none",
                color=colour,
            )
            if annotate:
                # Dev is labelled below its star, Test above, so the two do not collide
                # when both peak at the same epoch.
                axes.annotate(
                    f"e{int(best['epoch'])}: {best['biored_f1']:.3f}",
                    xy=(best["epoch"], best["biored_f1"]),
                    xytext=(0, 9) if split == "Test" else (0, -17),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=colour,
                )


def plot_dev_vs_test_run(frame: pd.DataFrame, encoder: str, dataset_raw: str) -> Path:
    """Draw the Dev and Test curves of one run into one figure."""
    figure, axes = plt.subplots(figsize=(8.0, 4.6))
    dataset = DATASET_NAMES.get(dataset_raw, dataset_raw)
    draw_split_curves(axes, frame)
    style_axes(
        axes,
        f"{encoder} trained on {dataset}: Dev vs. Test on BioRED, F1 per epoch",
        sorted(frame["epoch"].unique()),
    )
    # Shorter than the shared label, which does not fit next to a four-entry legend.
    axes.set_ylabel("BioRED-style F1 (8 relation types)")
    axes.legend(
        title="Evaluation split (dashed: all co-mentioned pairs)",
        loc="lower right",
        fontsize=10,
        title_fontsize=10,
    )
    figure.tight_layout()

    output = FIGURE_DIR / f"dev_vs_test_biored_f1_per_epoch_{slugify(encoder, dataset_raw)}.png"
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def plot_dev_vs_test_all_runs(frame: pd.DataFrame) -> Path:
    """Draw the Dev and Test curves of every run into one 2x2 panel figure."""
    groups = list(frame.groupby(["encoder", "train_dataset"], sort=False))
    figure, axes_grid = plt.subplots(2, 2, figsize=(13.0, 8.0), sharey=True)
    flat_axes: list[plt.Axes] = list(axes_grid.flatten())

    for axes, ((encoder_raw, dataset_raw), group) in zip(flat_axes, groups):
        encoder = ENCODER_NAMES.get(encoder_raw, encoder_raw)
        dataset = DATASET_SHORT_NAMES.get(dataset_raw, dataset_raw)
        draw_split_curves(axes, group, annotate=False)
        style_axes(axes, f"{encoder} / {dataset}", sorted(group["epoch"].unique()))
        axes.set_ylabel("BioRED-style F1")

    for axes in flat_axes[len(groups) :]:
        axes.set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Evaluation split (dashed: all co-mentioned pairs)",
        loc="lower center",
        ncol=4,
        fontsize=11,
        title_fontsize=11,
    )
    figure.suptitle(
        "Dev vs. Test on the common BioRED evaluation set: F1 per training epoch, all four runs\n"
        "(panels: encoder / negatives of the BioRED training set, stars: best epoch of a curve)",
        fontsize=15,
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    output = FIGURE_DIR / "dev_vs_test_biored_f1_per_epoch_all_runs.png"
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


# One curve per (model, metric). Solid is the realistic score, dashed the balanced one and
# dotted the plain micro-F1, so the most optimistic reading is the faintest line.
METRIC_SERIES: list[tuple[str, str, str, str]] = [
    ("micro", "micro_f1_all_classes", "matched_norelation", ":"),
    ("BioRED", "biored_f1", "matched_norelation", "--"),
    ("BioRED all pairs", "biored_f1", "full_norelation", "-"),
]


def plot_f1_metric_comparison(frame: pd.DataFrame, file_name: str) -> Path:
    """Draw the three F1 readings of every model into one figure.

    The legend lists one column per model, each holding the three metrics in the order
    micro, BioRED, BioRED all pairs.
    """
    figure, axes = plt.subplots(figsize=(10.0, 5.2))
    handles: list[plt.Line2D] = []
    labels: list[str] = []

    for encoder_raw in ENCODER_NAMES:
        encoder = ENCODER_NAMES[encoder_raw]
        for metric_name, column, variant, style in METRIC_SERIES:
            group = frame[(frame["encoder"] == encoder_raw) & (frame["variant"] == variant)]
            if group.empty:
                continue
            group = group.sort_values("epoch")
            line, = axes.plot(
                group["epoch"],
                group[column],
                marker="o",
                markersize=4,
                linestyle=style,
                color=RUN_COLOURS.get((encoder, COMMON_EVAL_DATASET), None),
            )
            handles.append(line)
            labels.append(f"{encoder} {metric_name}")

    style_axes(
        axes,
        "Model Performance comparison using different F1 metrics",
        sorted(frame["epoch"].unique()),
        y_label="F1 on the held-out BioRED test set",
        y_limits=(0.0, 0.9),
    )
    axes.legend(
        handles,
        labels,
        title="different f1 scores per model",
        loc="lower center",
        ncol=2,
        fontsize=10,
        title_fontsize=10,
    )
    figure.tight_layout()

    output = FIGURE_DIR / file_name
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def plot_all_runs(
    frame: pd.DataFrame,
    title: str,
    file_name: str,
    name_training_set: bool = True,
) -> Path:
    """Draw every run into one figure: colour per run, line style per test variant.

    With ``name_training_set`` disabled the legend carries the model name alone, which is
    only unambiguous when the frame holds a single training set.
    """
    figure, axes = plt.subplots(figsize=(10.0, 5.2))

    for (encoder_raw, dataset_raw, variant), group in frame.groupby(
        ["encoder", "train_dataset", "variant"], sort=False
    ):
        encoder = ENCODER_NAMES.get(encoder_raw, encoder_raw)
        dataset = DATASET_SHORT_NAMES.get(dataset_raw, dataset_raw)
        group = group.sort_values("epoch")
        suffix = "" if variant == "matched_norelation" else ", all pairs"
        label = f"{encoder} / {dataset}{suffix}" if name_training_set else f"{encoder}{suffix}"
        axes.plot(
            group["epoch"],
            group["biored_f1"],
            marker="o",
            markersize=4,
            linestyle=VARIANT_STYLES.get(variant, "-"),
            color=RUN_COLOURS.get((encoder, dataset_raw), None),
            label=label,
        )

    legend_title = (
        "Model / negatives of the BioRED training set (dashed: all co-mentioned pairs)"
        if name_training_set
        else "Model (dashed: all co-mentioned pairs)"
    )
    style_axes(axes, title, sorted(frame["epoch"].unique()))
    axes.legend(
        title=legend_title,
        loc="lower center",
        ncol=2,
        fontsize=10,
        title_fontsize=10,
    )
    figure.tight_layout()

    output = FIGURE_DIR / file_name
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def plot_own_vs_common_eval_set(frame: pd.DataFrame) -> Path:
    """Draw the type-restricted runs on their own evaluation set and on the common one.

    Only the matched-negatives variant exists for both, so only that variant is drawn.
    """
    figure, axes = plt.subplots(figsize=(9.0, 5.0))

    for (encoder_raw, train_raw, eval_raw), group in frame.groupby(
        ["encoder", "train_dataset", "eval_dataset"], sort=False
    ):
        encoder = ENCODER_NAMES.get(encoder_raw, encoder_raw)
        eval_dataset = DATASET_SHORT_NAMES.get(eval_raw, eval_raw)
        group = group.sort_values("epoch")
        axes.plot(
            group["epoch"],
            group["biored_f1"],
            marker="o",
            markersize=4,
            linestyle=EVAL_SET_STYLES.get(eval_raw, "-"),
            color=RUN_COLOURS.get((encoder, train_raw), None),
            label=f"{encoder}, tested with {eval_dataset}",
        )

    style_axes(
        axes,
        "Models trained with type-restricted negatives: own test set vs. the common one",
        sorted(frame["epoch"].unique()),
    )
    axes.set_ylabel("BioRED-style F1 (matched negatives)")
    axes.legend(
        title="Model (trained with type-restricted neg.), negatives of the test set",
        loc="lower right",
        fontsize=10,
        title_fontsize=10,
    )
    figure.tight_layout()

    output = FIGURE_DIR / "own_vs_common_testset_perturbed_runs.png"
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def main() -> int:
    """Read the CSV and write the Test-only figures plus the Dev-versus-Test figures."""
    if not CSV_FILE.exists():
        raise SystemExit(f"{CSV_FILE} does not exist. Run finetuning/epoch_curves.py first.")

    frame = load_curves()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Legacy view: every run on the evaluation set of its own training data, Test split.
    own_test = frame[(frame["split"] == "Test") & (frame["eval_dataset"] == frame["train_dataset"])]
    # Corrected view: every run on the one common evaluation set.
    common = frame[frame["eval_dataset"] == COMMON_EVAL_DATASET]
    common_test = common[common["split"] == "Test"]
    # Deck view: only the runs trained on the original BioRED training set, on the held-out
    # BioRED test set. One training set, so the legend does not have to name it.
    original_test = common_test[common_test["train_dataset"] == COMMON_EVAL_DATASET]

    written: list[Path] = []
    for (encoder_raw, dataset_raw), group in own_test.groupby(["encoder", "train_dataset"], sort=False):
        encoder = ENCODER_NAMES.get(encoder_raw, encoder_raw)
        written.append(plot_single_run(group, encoder, dataset_raw))
    written.append(
        plot_all_runs(
            original_test,
            "BioRED-style F1 per training epoch on the held-out BioRED test set",
            "biored_f1_per_epoch_all_runs.png",
            name_training_set=False,
        )
    )
    # The same runs read through all three metrics, kept separate so the plain version
    # above stays available to roll back to.
    written.append(
        plot_f1_metric_comparison(original_test, "biored_f1_per_epoch_all_runs_with_micro.png")
    )
    written.append(
        plot_all_runs(
            common_test,
            "BioRED-style F1 per training epoch, all runs on the common BioRED test set",
            "biored_f1_per_epoch_all_runs_common_testset.png",
        )
    )

    for (encoder_raw, dataset_raw), group in common.groupby(["encoder", "train_dataset"], sort=False):
        encoder = ENCODER_NAMES.get(encoder_raw, encoder_raw)
        written.append(plot_dev_vs_test_run(group, encoder, dataset_raw))
    written.append(plot_dev_vs_test_all_runs(common))

    perturbed = frame[
        (frame["split"] == "Test")
        & (frame["train_dataset"] != COMMON_EVAL_DATASET)
        & (frame["variant"] == "matched_norelation")
    ]
    if not perturbed.empty:
        written.append(plot_own_vs_common_eval_set(perturbed))

    for path in written:
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
