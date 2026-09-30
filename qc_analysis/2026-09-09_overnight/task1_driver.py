"""Task 1: which drives the generator's error rate, longer texts or more relations?

Recomputed from scratch on 2026-09-09. Nothing is carried over from the earlier markdown
notes; the only input is the scored-claim table, and the row counts are printed so a
result computed on an empty or badly joined frame cannot look like a clean zero.

Fred's method rule, given twice at meeting 9: fix as many variables as possible and vary
exactly one. So the answer is not a correlation. It is a grid. Bin the abstracts by
length and by relation count, then

  * read ACROSS a row  = length held inside one band, relation count varies
  * read DOWN a column = relation count held inside one band, length varies

and compare the two swings. Both swings are measured on the same cells of the same grid,
so nothing else differs between them.

Run:
    python task1_driver.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# These figures end up scaled to about 60% inside a two-column-width LaTeX float, so
# everything is set larger than it looks here. Fred's complaint at meeting 9 was that
# the earlier version could not be read at a glance; small type is the same defect.
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 11, "figure.titlesize": 16})

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / "LLMs_for_NEL"
SCORES = REPO / "qc_filtering" / "synthetic_relation_scores.csv"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

FOUR_CLASS = ["Association", "Positive_Correlation", "Negative_Correlation", "NoRelation"]
CUTOFF = 0.5
MIN_CELL = 25   # a cell below this is noise and is not drawn


def per_abstract(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.relation_type.isin(FOUR_CLASS)].copy()
    d["failed"] = d.prob_supported < CUTOFF
    key = ["model", "split", "paper_id", "generation"]
    out = (d.groupby(key)
             .agg(n_relations=("rel_idx", "count"),
                  n_failed=("failed", "sum"),
                  abstract_words=("abstract_words", "first"))
             .reset_index())
    out["error_rate"] = out.n_failed / out.n_relations
    return out


def main() -> dict:
    raw = pd.read_csv(SCORES, dtype={"paper_id": str})
    print(f"loaded {len(raw):,} scored claims from {SCORES.name}")
    assert len(raw) == 35658, f"unexpected claim count {len(raw)}"
    kept = raw.relation_type.isin(FOUR_CLASS)
    print(f"  {kept.sum():,} claims of the four modelled types, "
          f"{(~kept).sum():,} of the five rare types dropped")

    a = per_abstract(raw)
    print(f"  -> {len(a):,} synthetic abstracts")
    assert len(a) > 3000, "abstract collapse produced too few rows"
    print(f"  relations per abstract: median {a.n_relations.median():.0f}, "
          f"range {a.n_relations.min()} to {a.n_relations.max()}")
    print(f"  length in words:        median {a.abstract_words.median():.0f}, "
          f"range {a.abstract_words.min()} to {a.abstract_words.max()}")
    print(f"  overall claim-level rejection rate "
          f"{(raw[kept].prob_supported < CUTOFF).mean():.4f}")
    print(f"  mean per-abstract error rate {a.error_rate.mean():.4f}")

    # --- the grid ---------------------------------------------------------------
    # Quartile edges on each axis, computed from the data rather than chosen, so the
    # two axes are cut the same way and neither is favoured.
    rel_q = [a.n_relations.quantile(q) for q in (0.25, 0.5, 0.75)]
    len_q = [a.abstract_words.quantile(q) for q in (0.25, 0.5, 0.75)]
    rel_edges = [-0.5] + list(dict.fromkeys(rel_q)) + [a.n_relations.max() + 0.5]
    len_edges = [-0.5] + list(dict.fromkeys(len_q)) + [a.abstract_words.max() + 0.5]

    def labels(series, edges, unit):
        """Label each bin by the actual smallest and largest value inside it, so no
        two labels share a boundary number and nothing suggests an empty range."""
        cut = pd.cut(series, edges)
        out = []
        for iv in cut.cat.categories:
            v = series[cut == iv]
            if len(v) == 0:
                out.append("empty")
                continue
            out.append(f"{int(v.min())}-{int(v.max())}{unit}")
        return out

    a["rel_bin"] = pd.cut(a.n_relations, rel_edges,
                          labels=labels(a.n_relations, rel_edges, ""))
    a["len_bin"] = pd.cut(a.abstract_words, len_edges,
                          labels=labels(a.abstract_words, len_edges, " w"))

    grid = a.pivot_table(index="len_bin", columns="rel_bin", values="error_rate",
                         aggfunc="mean", observed=True)
    counts = a.pivot_table(index="len_bin", columns="rel_bin", values="error_rate",
                           aggfunc="size", observed=True)
    grid = grid.where(counts >= MIN_CELL)

    print("\ncell counts (abstracts):")
    print(counts.to_string())
    print("\nmean error rate per cell:")
    print(grid.round(3).to_string())

    # Swings: first to last column within a row, first to last row within a column.
    row_swings, col_swings = {}, {}
    for r in grid.index:
        row = grid.loc[r].dropna()
        if len(row) > 1:
            row_swings[str(r)] = float(row.iloc[-1] - row.iloc[0])
    for c in grid.columns:
        col = grid[c].dropna()
        if len(col) > 1:
            col_swings[str(c)] = float(col.iloc[-1] - col.iloc[0])

    rs = np.array(list(row_swings.values()))
    cs = np.array(list(col_swings.values()))
    print(f"\nvary RELATION COUNT, length held fixed: {rs.min():+.3f} to {rs.max():+.3f}, "
          f"mean {rs.mean():+.3f}  (n={len(rs)} length bands)")
    print(f"vary LENGTH, relation count held fixed:  {cs.min():+.3f} to {cs.max():+.3f}, "
          f"mean {cs.mean():+.3f}  (n={len(cs)} relation bands)")
    ratio = float(rs.mean() / cs.mean()) if cs.mean() else float("nan")
    print(f"ratio of mean swings: {ratio:.2f}x")

    # --- a second, assumption-free check on the same question -------------------
    # Partial Spearman is overkill; instead regress the error rate on both, each
    # standardised, and report the two coefficients. Same data, no binning at all.
    x1 = (a.n_relations - a.n_relations.mean()) / a.n_relations.std()
    x2 = (a.abstract_words - a.abstract_words.mean()) / a.abstract_words.std()
    X = np.column_stack([np.ones(len(a)), x1, x2])
    beta, *_ = np.linalg.lstsq(X, a.error_rate.values, rcond=None)
    print(f"\nOLS on standardised predictors (no binning): "
          f"error_rate = {beta[0]:.3f} {beta[1]:+.3f}*z(relations) {beta[2]:+.3f}*z(length)")
    print(f"  correlation between the two predictors: "
          f"{np.corrcoef(a.n_relations, a.abstract_words)[0,1]:+.3f}")

    # --- the figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), layout="constrained",
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    m = grid.values.astype(float)
    im = ax.imshow(m, cmap="YlOrRd", vmin=0, vmax=float(np.nanmax(m)) * 1.05,
                   aspect="auto")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([str(c) for c in grid.columns])
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([str(i) for i in grid.index])
    ax.set_xlabel("relations the abstract was asked to express")
    ax.set_ylabel("abstract length (words)")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isnan(m[i, j]):
                continue
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=16,
                    color="black" if m[i, j] < np.nanmax(m) * 0.6 else "white")
    ax.set_title("Mean error rate in each cell\n"
                 "read ACROSS a row: only the relation count changes\n"
                 "read DOWN a column: only the length changes", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="errors / relations")

    ax = axes[1]
    ax.bar([0], [rs.mean()], yerr=[[rs.mean() - rs.min()], [rs.max() - rs.mean()]],
           color="#C44E52", width=0.55, capsize=6)
    ax.bar([1], [cs.mean()], yerr=[[cs.mean() - cs.min()], [cs.max() - cs.mean()]],
           color="#4C72B0", width=0.55, capsize=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["more relations\n(length held fixed)",
                        "longer text\n(relations held fixed)"], fontsize=13)
    ax.set_ylabel("rise in error rate")
    for i, v, hi in [(0, rs.mean(), rs.max()), (1, cs.mean(), cs.max())]:
        ax.text(i, hi + 0.018, f"{v:+.2f}", ha="center", fontweight="bold", fontsize=18)
    ax.set_ylim(0, max(rs.max(), cs.max()) * 1.35)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"The relation count moves it {ratio:.1f}x more\n"
                 "bar = mean, whisker = min and max band",
                 fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("What drives the generator's error rate: the number of relations, "
                 "not the length of the text",
                 fontsize=16, fontweight="bold")
    fig.savefig(FIG / "task1_driver_grid.png", dpi=200)
    plt.close(fig)
    print(f"\nwrote {FIG/'task1_driver_grid.png'}")

    # --- the strictest version of the same question ------------------------------
    # Banding still leaves variation inside a band. So hold the relation count to a
    # single exact integer and ask whether length still moves the error rate, then
    # hold length to a 10-word window and ask whether the relation count still does.
    def rho(x, y):
        return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])

    fixed_rel = [(int(k), len(g), rho(g.abstract_words, g.error_rate))
                 for k, g in a.groupby("n_relations") if len(g) >= 80]
    a["lwin"] = (a.abstract_words // 10) * 10
    fixed_len = [(int(k), len(g), rho(g.n_relations, g.error_rate))
                 for k, g in a.groupby("lwin") if len(g) >= 80]

    def wmean(t):
        n = np.array([x[1] for x in t], dtype=float)
        r = np.array([x[2] for x in t])
        return float((r * n / n.sum()).sum())

    wr_len, wr_rel = wmean(fixed_rel), wmean(fixed_len)
    print(f"\nrelation count held to ONE exact value ({len(fixed_rel)} values, "
          f"{sum(x[1] for x in fixed_rel):,} abstracts):")
    print(f"  correlation of length with the error rate = {wr_len:+.3f}, "
          f"positive in {sum(1 for x in fixed_rel if x[2] > 0)}/{len(fixed_rel)} of them")
    print(f"length held to a 10-word window ({len(fixed_len)} windows, "
          f"{sum(x[1] for x in fixed_len):,} abstracts):")
    print(f"  correlation of relation count with the error rate = {wr_rel:+.3f}, "
          f"positive in {sum(1 for x in fixed_len if x[2] > 0)}/{len(fixed_len)} of them")

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y1 = [x[2] for x in fixed_rel]
    y2 = [x[2] for x in fixed_len]
    ax.scatter(np.random.default_rng(0).normal(0, 0.045, len(y1)), y1, s=46,
               color="#4C72B0", alpha=0.8, zorder=3, label="one dot = one band")
    ax.scatter(1 + np.random.default_rng(1).normal(0, 0.045, len(y2)), y2, s=46,
               color="#C44E52", alpha=0.8, zorder=3)
    ax.hlines(wr_len, -0.25, 0.25, color="#4C72B0", linewidth=3, zorder=4)
    ax.hlines(wr_rel, 0.75, 1.25, color="#C44E52", linewidth=3, zorder=4)
    ax.text(0.30, wr_len, f"  mean {wr_len:+.2f}", ha="left", va="center",
            fontweight="bold", fontsize=15, color="#4C72B0")
    ax.text(1.30, wr_rel, f"  mean {wr_rel:+.2f}", ha="left", va="center",
            fontweight="bold", fontsize=15, color="#C44E52")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["effect of LENGTH,\nrelation count fixed to one exact value",
                        "effect of RELATION COUNT,\nlength fixed to a 10-word window"],
                       fontsize=9)
    ax.set_xlim(-0.45, 1.95)
    ax.set_ylim(-0.42, 0.55)
    ax.set_ylabel("rank correlation with the error rate")
    ax.set_title("Hold the other variable fixed and only one effect survives",
                 fontsize=14, fontweight="bold")
    ax.legend(frameon=False, fontsize=11, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "task1_conditional.png", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG/'task1_conditional.png'}")

    stats = {
        "n_claims_total": int(len(raw)),
        "n_claims_four_class": int(kept.sum()),
        "n_abstracts": int(len(a)),
        "cutoff": CUTOFF,
        "mean_abstract_error_rate": round(float(a.error_rate.mean()), 4),
        "grid_counts": counts.to_dict(),
        "grid_error_rate": {str(k): {str(kk): (None if pd.isna(vv) else round(float(vv), 4))
                                     for kk, vv in v.items()}
                            for k, v in grid.to_dict("index").items()},
        "swing_vary_relations_hold_length": {k: round(v, 4) for k, v in row_swings.items()},
        "swing_vary_length_hold_relations": {k: round(v, 4) for k, v in col_swings.items()},
        "mean_swing_relations": round(float(rs.mean()), 4),
        "mean_swing_length": round(float(cs.mean()), 4),
        "ratio": round(ratio, 3),
        "ols_standardised": {"intercept": round(float(beta[0]), 4),
                             "z_relations": round(float(beta[1]), 4),
                             "z_length": round(float(beta[2]), 4)},
        "corr_relations_length": round(float(np.corrcoef(a.n_relations,
                                                         a.abstract_words)[0, 1]), 4),
        "exact_fix": {
            "relcount_fixed_rho_length": round(wr_len, 4),
            "relcount_fixed_bands": len(fixed_rel),
            "relcount_fixed_positive": sum(1 for x in fixed_rel if x[2] > 0),
            "relcount_fixed_abstracts": int(sum(x[1] for x in fixed_rel)),
            "length_fixed_rho_relcount": round(wr_rel, 4),
            "length_fixed_bands": len(fixed_len),
            "length_fixed_positive": sum(1 for x in fixed_len if x[2] > 0),
            "length_fixed_abstracts": int(sum(x[1] for x in fixed_len)),
            "per_band_relcount_fixed": [[k, n, round(r, 4)] for k, n, r in fixed_rel],
            "per_band_length_fixed": [[k, n, round(r, 4)] for k, n, r in fixed_len],
        },
    }
    (HERE / "task1_stats.json").write_text(json.dumps(stats, indent=2, default=str) + "\n")
    a.to_csv(HERE / "task1_abstract_level.csv", index=False)
    return stats


if __name__ == "__main__":
    main()
