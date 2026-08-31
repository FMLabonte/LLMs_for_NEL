"""Meeting 8 tasks 1 to 3: the normalised error-rate plots on the synthetic data.

Asked at the 2026-08-27 meeting: if the generator's error rate were linear in the
number of relations, the normalised curve is flat. Any rise means errors grow
super-linearly, i.e. the generator gets less reliable as the abstract gets more
complex.

So the unit is one synthetic abstract and the quantity is

    error rate = (relation claims the QC model rejects) / (relation claims made)

plotted against the relation count, and again against abstract length, because
the two are correlated and he wants to know which one is doing the work.

Everything here is post-processing of synthetic_relation_scores.csv. No model is
loaded unless --truncation is passed.

Run:
    python error_rate_analysis.py                 # the two plots + the split
    python error_rate_analysis.py --truncation    # adds the 512-token check
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
SCORES = OUT_DIR / "synthetic_relation_scores.csv"
FIG_DIR = OUT_DIR / "figures"

FOUR_CLASS = {"Association", "Positive_Correlation", "Negative_Correlation", "NoRelation"}

# One consistent look for every figure so they can sit next to each other in the
# report without being re-styled.
POINT = dict(s=9, alpha=0.12, color="#4C72B0", linewidths=0, rasterized=True)
LINE = dict(color="#C44E52", marker="o", markersize=5, linewidth=2, zorder=5)
BAND = dict(color="#C44E52", alpha=0.18, zorder=4)


def per_abstract(df: pd.DataFrame, cutoff: float, rares: str) -> pd.DataFrame:
    """Collapse the per-claim scores to one row per synthetic abstract."""
    d = df.copy()
    d["is_rare"] = ~d.relation_type.isin(FOUR_CLASS)
    d["failed"] = d.prob_supported < cutoff
    if rares == "exclude":
        # The QC model never saw these five types in training, so a rejection
        # says more about coverage than about the generator. Drop the claims
        # rather than zero them, otherwise they still inflate the denominator.
        d = d[~d.is_rare]

    key = ["model", "split", "paper_id", "generation"]
    out = (d.groupby(key)
             .agg(n_relations=("rel_idx", "count"),
                  n_failed=("failed", "sum"),
                  mean_prob=("prob_supported", "mean"),
                  abstract_words=("abstract_words", "first"))
             .reset_index())
    out["error_rate"] = out.n_failed / out.n_relations
    return out


MIN_BIN = 25   # below this a bin is noise, and a noisy line is as useless as a cloud


def binned(x: pd.Series, y: pd.Series, edges, min_n: int = MIN_BIN) -> pd.DataFrame:
    """Mean error rate per bin with a 95% normal-approximation interval.

    A raw scatter of these two quantities is the uninformative point cloud that
    the real-data version produced, so the binned mean is the actual answer and
    the scatter is only background.

    Each bin is plotted at the MEDIAN x of the abstracts inside it, not at the
    interval midpoint. The top bin is wide and its members sit at its left edge,
    so a midpoint would draw a long straight segment out into empty space and
    invent a trend that no data supports.
    """
    b = pd.cut(x, edges, include_lowest=True)
    frame = pd.DataFrame({"x": x, "y": y, "bin": b})
    g = frame.groupby("bin", observed=True)
    out = g.y.agg(["count", "mean", "std"]).rename(columns={"count": "n"})
    out["centre"] = g.x.median()
    out["se"] = out["std"] / np.sqrt(out["n"])
    out["lo"] = out["mean"] - 1.96 * out["se"]
    out["hi"] = out["mean"] + 1.96 * out["se"]
    out = out.reset_index(names="bin")
    return out[out.n >= min_n].reset_index(drop=True)


def trend_plot(ax, x, y, edges, xlabel, title, xmax_q: float = 0.99):
    """Scatter as background, binned mean as the answer, flat line as the null."""
    b = binned(x, y, edges)
    # A long thin tail (a handful of abstracts with 60+ relations) would stretch
    # the axis and squash the region where almost all the data lives.
    xmax = float(np.quantile(x, xmax_q))
    beyond = int((x > xmax).sum())

    ax.scatter(x, y, **POINT)
    ax.plot(b.centre, b["mean"], label="mean error rate per bin", **LINE)
    ax.fill_between(b.centre, b.lo, b.hi, **BAND)

    # The flat line is the null: a constant per-claim error rate.
    overall = y.mean()
    ax.axhline(overall, color="#555555", linestyle="--", linewidth=1.2,
               label=f"constant rate ({overall:.1%}) = linear scaling")

    xmin = float(np.quantile(x, 0.005))
    ax.set_xlim(max(0.0, xmin * 0.9), xmax * 1.03)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("errors / relations in the abstract")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    if beyond:
        ax.text(0.99, 0.02, f"{beyond} abstracts beyond {xmax:.0f} not shown",
                transform=ax.transAxes, ha="right", fontsize=7, color="#777777")
    return b


def spearman(x, y) -> tuple[float, float]:
    """Rank correlation plus a large-sample p, without pulling in scipy."""
    rx, ry = pd.Series(x).rank(), pd.Series(y).rank()
    r = np.corrcoef(rx, ry)[0, 1]
    n = len(x)
    z = np.arctanh(r) * np.sqrt((n - 3) / 1.06)
    # Two-sided normal tail via the complementary error function.
    from math import erfc, sqrt
    return float(r), float(erfc(abs(z) / sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=float, default=0.5)
    ap.add_argument("--rares", choices=["include", "exclude"], default="exclude")
    ap.add_argument("--truncation", action="store_true",
                    help="tokenise the abstracts and check the 512-token confound")
    args = ap.parse_args()

    FIG_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(SCORES, dtype={"paper_id": str})
    abs_df = per_abstract(df, args.cutoff, args.rares)
    print(f"{len(df):,} claims -> {len(abs_df):,} synthetic abstracts "
          f"(rares {args.rares}d, cutoff {args.cutoff})")

    rel_edges = [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 100]
    len_edges = [0, 120, 160, 200, 240, 280, 320, 380, 450, 2000]

    # --- Task 1: normalised error rate against relation count ----------------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    rel_bins = trend_plot(ax, abs_df.n_relations, abs_df.error_rate, rel_edges,
                          "relations asserted in the abstract",
                          "Normalised error rate vs relation count (synthetic, QC model)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_rate_vs_relations.png", dpi=200)
    plt.close(fig)

    # --- Task 2: the twin plot against length --------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    len_bins = trend_plot(ax, abs_df.abstract_words, abs_df.error_rate, len_edges,
                          "abstract length (words)",
                          "Normalised error rate vs abstract length (synthetic, QC model)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_rate_vs_length.png", dpi=200)
    plt.close(fig)

    # --- Which of the two is driving it --------------------------------------
    # Relation count and length are correlated, so hold one roughly fixed and
    # see whether the other still moves the error rate.
    abs_df["len_band"] = pd.qcut(abs_df.abstract_words, 3,
                                 labels=["short", "medium", "long"])
    abs_df["rel_band"] = pd.qcut(abs_df.n_relations, 3,
                                 labels=["few", "medium", "many"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for band, colour in zip(["short", "medium", "long"], ["#4C72B0", "#DD8452", "#55A868"]):
        sub = abs_df[abs_df.len_band == band]
        b = binned(sub.n_relations, sub.error_rate, rel_edges)
        axes[0].plot(b.centre, b["mean"], marker="o", markersize=4, color=colour,
                     label=f"{band} abstracts")
    axes[0].set_xlim(0, float(np.quantile(abs_df.n_relations, 0.98)) * 1.03)
    axes[0].set_xlabel("relations asserted")
    axes[0].set_ylabel("errors / relations")
    axes[0].set_title("Relation count, holding length band fixed")

    for band, colour in zip(["few", "medium", "many"], ["#4C72B0", "#DD8452", "#55A868"]):
        sub = abs_df[abs_df.rel_band == band]
        b = binned(sub.abstract_words, sub.error_rate, len_edges)
        axes[1].plot(b.centre, b["mean"], marker="o", markersize=4, color=colour,
                     label=f"{band} relations")
    axes[1].set_xlim(float(np.quantile(abs_df.abstract_words, 0.01)) * 0.95,
                     float(np.quantile(abs_df.abstract_words, 0.99)) * 1.03)
    axes[1].set_xlabel("abstract length (words)")
    axes[1].set_title("Length, holding relation-count band fixed")

    for ax in axes:
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_rate_disentangled.png", dpi=200)
    plt.close(fig)

    r_rel, p_rel = spearman(abs_df.n_relations, abs_df.error_rate)
    r_len, p_len = spearman(abs_df.abstract_words, abs_df.error_rate)

    stats = {
        "n_claims": int(len(df)),
        "n_abstracts": int(len(abs_df)),
        "cutoff": args.cutoff,
        "rares": args.rares,
        "overall_error_rate": round(float(abs_df.error_rate.mean()), 4),
        "spearman_error_rate_vs_relations": [round(r_rel, 4), float(f"{p_rel:.3g}")],
        "spearman_error_rate_vs_length": [round(r_len, 4), float(f"{p_len:.3g}")],
        "spearman_relations_vs_length": [round(spearman(abs_df.n_relations,
                                                        abs_df.abstract_words)[0], 4)],
        "by_relation_count": rel_bins.assign(
            bin=rel_bins.bin.astype(str)).round(4).to_dict("records"),
        "by_length": len_bins.assign(
            bin=len_bins.bin.astype(str)).round(4).to_dict("records"),
    }

    # --- The confound worth ruling out before claiming the curve -------------
    if args.truncation:
        from transformers import AutoTokenizer
        from filter_synthetic import build_rows, MODEL_DIR

        rows = build_rows()
        uniq = rows.drop_duplicates(["model", "split", "paper_id", "generation"])
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        lens = [len(tok(a, add_special_tokens=True)["input_ids"])
                for a in uniq.abstract.tolist()]
        uniq = uniq.assign(abstract_tokens=lens)
        merged = abs_df.merge(
            uniq[["model", "split", "paper_id", "generation", "abstract_tokens"]],
            on=["model", "split", "paper_id", "generation"], how="left")
        # The prompt is "Relation: ...\nContext: <abstract>", so the abstract
        # loses its tail whenever the whole thing passes 512 word pieces.
        merged["truncated"] = merged.abstract_tokens > 480
        share = float(merged.truncated.mean())
        stats["truncation"] = {
            "max_len": 512,
            "share_abstracts_near_or_over_limit": round(share, 4),
            "error_rate_truncated": round(float(merged[merged.truncated].error_rate.mean()), 4)
            if merged.truncated.any() else None,
            "error_rate_intact": round(float(merged[~merged.truncated].error_rate.mean()), 4),
            "median_tokens": int(merged.abstract_tokens.median()),
            "p95_tokens": int(merged.abstract_tokens.quantile(0.95)),
        }
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.hist(merged.abstract_tokens, bins=60, color="#4C72B0")
        ax.axvline(512, color="#C44E52", linestyle="--",
                   label="QC model max_length = 512")
        ax.set_xlabel("abstract length (PubMedBERT word pieces)")
        ax.set_ylabel("synthetic abstracts")
        ax.set_title("Do long abstracts get truncated before the QC model sees them?")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "truncation_check.png", dpi=200)
        plt.close(fig)
        merged.to_csv(OUT_DIR / "abstract_error_rates.csv", index=False)
    else:
        abs_df.to_csv(OUT_DIR / "abstract_error_rates.csv", index=False)

    (OUT_DIR / "error_rate_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"\noverall error rate {stats['overall_error_rate']:.1%}")
    print(f"spearman vs relation count {r_rel:+.3f} (p={p_rel:.2g})")
    print(f"spearman vs length         {r_len:+.3f} (p={p_len:.2g})")
    print("\nby relation count:")
    print(rel_bins[["bin", "n", "mean"]].to_string(index=False))
    print("\nby length:")
    print(len_bins[["bin", "n", "mean"]].to_string(index=False))
    if args.truncation:
        print("\ntruncation:", json.dumps(stats["truncation"], indent=2))
    print(f"\nwrote {FIG_DIR}/, abstract_error_rates.csv, error_rate_stats.json")


if __name__ == "__main__":
    main()
