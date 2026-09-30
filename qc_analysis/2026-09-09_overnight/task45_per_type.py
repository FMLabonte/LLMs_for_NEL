"""Tasks 4 and 5: error rate per relation type, and the two-sided real/synthetic table.

Task 4 asks for the error rate per relation type on the synthetic data. Task 5 asks for
the same thing computed twice, once on the real BioRED test set where the ground truth is
known and once on the synthetic data, split by relation type and again by entity pair.

They are one computation seen from two sides, so they live in one script.

The logic, stated once:

  * On the real test set each claim carries a label. label=1 means the claim is true of
    the text, label=0 means a perturbation made it false. So two rates are measurable:
      - catch rate  = share of the label=0 claims the QC model rejects. How good the QC
                      model is at finding a wrong claim of this type.
      - false alarm = share of the label=1 claims the QC model rejects. What the QC model
                      costs on claims that are actually fine.
  * On the synthetic data there is no label. Every claim in the prompt is one the
    generator was told to express, so if the generator did its job the claim is true and
    the row is the synthetic counterpart of a real label=1 gold. The measurable quantity
    is the rejection rate.
  * Therefore rejection(synthetic) - false alarm(real gold) is the part of the synthetic
    rejection rate that the QC model's own error does not explain. That excess is the
    estimate of how often the generator actually got that relation type wrong.

This is the comparison Fred asked for: "if there are strong differences between the two
then you know where the LLM probably struggles".

Everything is recomputed from the two prediction tables. Row counts are asserted.

Run:
    python task45_per_type.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Figures are scaled down inside LaTeX floats, so type is set larger than it looks
# here. Unreadable legends were the exact defect Fred called out at meeting 9.
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 11, "figure.titlesize": 15})

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / "LLMs_for_NEL"
WORK = HERE.parent
SYN_POS = REPO / "qc_filtering" / "synthetic_relation_scores.csv"
SYN_NEG = HERE / "norelation_scores.csv"          # written by score_norelation.py
REAL_PRED = WORK / "2026-07-22_next_tasks" / "qc_run2_test_predictions.csv"
REAL_ROWS = WORK / "2026-07-08_meeting" / "data" / "qc_test.csv"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

CUTOFF = 0.5
FOUR = ["Association", "Positive_Correlation", "Negative_Correlation", "NoRelation"]
SHORT = {"Association": "Assoc", "Positive_Correlation": "Pos_Corr",
         "Negative_Correlation": "Neg_Corr", "NoRelation": "NoRelation"}
TYPE_SHORT = {"GeneOrGeneProduct": "Gene", "ChemicalEntity": "Chem",
              "DiseaseOrPhenotypicFeature": "Disease", "SequenceVariant": "Variant",
              "OrganismTaxon": "Species", "CellLine": "CellLine"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval. Normal approximation breaks down on the small cells here."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pair_label(ta: str, tb: str) -> str:
    """Unordered entity-type pair. Association carries no direction and the correlations
    are counted in both directions across the corpus, so ordering the pair would split
    one biological pairing into two rows for no reason."""
    x, y = sorted([TYPE_SHORT.get(ta, ta), TYPE_SHORT.get(tb, tb)])
    return f"{x}-{y}"


def load_real() -> pd.DataFrame:
    pred = pd.read_csv(REAL_PRED)
    rows = pd.read_csv(REAL_ROWS)
    print(f"real test predictions: {len(pred):,} rows")
    print(f"real test source rows: {len(rows):,} rows")
    assert len(pred) == len(rows) == 15199, "real test set changed size"
    # The two files are the same table in the same order; run2_infer_test.py wrote the
    # predictions straight from qc_test.csv without reordering. Verified below rather
    # than assumed, because a silent misalignment here would poison every number.
    for c in ["pmid", "relation_type", "entity_a_id", "entity_b_id", "label",
              "perturbation"]:
        same = (pred[c].astype(str).values == rows[c].astype(str).values).mean()
        assert same == 1.0, f"real test files disagree on {c} ({same:.4f} match)"
    print("  alignment check: pmid, relation_type, entity ids, label and perturbation "
          "all match row for row")
    df = rows.copy()
    df["prob"] = pred["prob"].values
    df["rejected"] = df.prob < CUTOFF
    df["pair"] = [pair_label(a, b) for a, b in zip(df.entity_a_type, df.entity_b_type)]
    print(f"  label=1 (true claims): {int((df.label==1).sum()):,}   "
          f"label=0 (perturbed, false claims): {int((df.label==0).sum()):,}")
    p = (~df[df.label == 1].rejected).mean()
    tp = df[(df.label == 1) & (~df.rejected)].shape[0]
    fp = df[(df.label == 0) & (~df.rejected)].shape[0]
    fn = df[(df.label == 1) & (df.rejected)].shape[0]
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    print(f"  overall run-2 test scores recomputed: P={prec:.4f} R={rec:.4f} "
          f"F1={2*prec*rec/(prec+rec):.4f}  (published run 2: 0.684 / 0.730 / 0.706)")
    del p
    return df


def load_synthetic() -> pd.DataFrame:
    pos = pd.read_csv(SYN_POS, dtype={"paper_id": str})
    pos = pos[pos.relation_type.isin(FOUR)].copy()
    pos["source"] = "stated positive"
    print(f"synthetic stated positive claims: {len(pos):,}")
    if SYN_NEG.exists():
        neg = pd.read_csv(SYN_NEG, dtype={"paper_id": str})
        neg["source"] = "implicit NoRelation"
        print(f"synthetic implicit NoRelation claims: {len(neg):,}")
        cols = ["model", "split", "paper_id", "generation", "relation_type",
                "type_a", "type_b", "prob_supported", "source"]
        df = pd.concat([pos[cols], neg[cols]], ignore_index=True)
    else:
        print("WARNING: norelation_scores.csv missing, NoRelation side unavailable")
        df = pos[["model", "split", "paper_id", "generation", "relation_type",
                  "type_a", "type_b", "prob_supported", "source"]]
    df["rejected"] = df.prob_supported < CUTOFF
    df["pair"] = [pair_label(a, b) for a, b in zip(df.type_a, df.type_b)]
    return df


def rate_table(df: pd.DataFrame, group: str, mask, name: str) -> pd.DataFrame:
    sub = df[mask]
    g = sub.groupby(group, observed=True).rejected.agg(["sum", "count"])
    g.columns = [f"{name}_rejected", f"{name}_n"]
    g[f"{name}_rate"] = g[f"{name}_rejected"] / g[f"{name}_n"]
    ci = [wilson(int(k), int(n)) for k, n in zip(g[f"{name}_rejected"], g[f"{name}_n"])]
    g[f"{name}_lo"] = [c[0] for c in ci]
    g[f"{name}_hi"] = [c[1] for c in ci]
    return g


def build(real: pd.DataFrame, syn: pd.DataFrame, group: str, min_n: int) -> pd.DataFrame:
    catch = rate_table(real, group, real.label == 0, "catch")
    alarm = rate_table(real, group, real.label == 1, "alarm")
    srej = rate_table(syn, group, slice(None) if False else syn.index == syn.index, "syn")
    t = catch.join(alarm, how="outer").join(srej, how="outer")
    t = t[(t.catch_n.fillna(0) >= min_n) & (t.alarm_n.fillna(0) >= min_n)
          & (t.syn_n.fillna(0) >= min_n)]
    t["excess"] = t.syn_rate - t.alarm_rate
    return t.sort_values("syn_n", ascending=False)


def main():
    print("=" * 78)
    real = load_real()
    print("-" * 78)
    syn = load_synthetic()
    print("=" * 78)

    out: dict = {}

    # ---- by relation type ------------------------------------------------------
    t = build(real, syn, "relation_type", min_n=30)
    t = t.reindex([r for r in FOUR if r in t.index])
    show = t[["catch_n", "catch_rate", "alarm_n", "alarm_rate", "syn_n", "syn_rate",
              "excess"]].copy()
    print("\nTASK 5, split by relation type")
    print("catch_rate = share of deliberately-wrong REAL claims the QC model rejects")
    print("alarm_rate = share of TRUE real claims the QC model wrongly rejects")
    print("syn_rate   = share of SYNTHETIC claims of this type the QC model rejects")
    print("excess     = syn_rate - alarm_rate, the part the QC model's own error "
          "does not explain\n")
    print(show.round(4).to_string())
    out["by_relation_type"] = json.loads(t.round(4).reset_index().to_json(orient="records"))

    # ---- by entity pair --------------------------------------------------------
    tp = build(real, syn, "pair", min_n=50)
    print("\n\nTASK 5, split by entity-type pair (unordered, cells with n>=50 on all "
          "three sides)\n")
    print(tp[["catch_n", "catch_rate", "alarm_n", "alarm_rate", "syn_n", "syn_rate",
              "excess"]].round(4).to_string())
    out["by_entity_pair"] = json.loads(tp.round(4).reset_index().to_json(orient="records"))

    # ---- task 4 figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4), layout="constrained")

    ax = axes[0]
    idx = list(t.index)
    x = np.arange(len(idx))
    w = 0.27
    for off, col, lab, colour in [
            (-w, "catch_rate", "real test set: wrong claims caught", "#55A868"),
            (0.0, "alarm_rate", "real test set: true claims wrongly rejected", "#4C72B0"),
            (w, "syn_rate", "synthetic data: claims rejected", "#C44E52")]:
        lo = t[col] - t[col.replace("_rate", "_lo")]
        hi = t[col.replace("_rate", "_hi")] - t[col]
        ax.bar(x + off, t[col], w, yerr=[lo, hi], capsize=3, color=colour, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(i, i) for i in idx], fontsize=12)
    ax.set_ylabel("share of claims rejected by the QC model")
    ax.set_ylim(0, 1.34)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Per relation type, all three rates on one axis\n"
                 "whiskers are 95% Wilson intervals", fontsize=13)

    ax = axes[1]
    ax.bar(x, t.excess, 0.55, color=["#C44E52" if v > 0 else "#999999" for v in t.excess])
    for i, v in enumerate(t.excess):
        ax.text(i, v + (0.012 if v >= 0 else -0.03), f"{v:+.2f}", ha="center",
                fontweight="bold", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(i, i) for i in idx], fontsize=12)
    ax.set_ylabel("synthetic rejection minus real false-alarm rate")
    ax.set_ylim(min(0, t.excess.min()) - 0.055, max(t.excess.max() + 0.055, 0.08))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("How much of the synthetic rejection the QC model\n"
                 "does not explain: the generator's own error", fontsize=13)

    fig.suptitle("Error rate per relation type, real test set against synthetic data",
                 fontsize=15, fontweight="bold")
    fig.savefig(FIG / "task4_per_relation_type.png", dpi=200)
    plt.close(fig)

    # ---- entity pair figure ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.0), layout="constrained")
    tpp = tp.sort_values("excess")
    y = np.arange(len(tpp))
    ax.barh(y - 0.2, tpp.alarm_rate, 0.38, color="#4C72B0",
            label="real test set: true claims wrongly rejected")
    ax.barh(y + 0.2, tpp.syn_rate, 0.38, color="#C44E52",
            label="synthetic data: claims rejected")
    for i, (a_, s_) in enumerate(zip(tpp.alarm_rate, tpp.syn_rate)):
        ax.text(max(a_, s_) + 0.012, i, f"{s_-a_:+.2f}", va="center", fontsize=14,
                fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i}  (n={int(n)})" for i, n in zip(tpp.index, tpp.syn_n)],
                       fontsize=12)
    ax.set_xlabel("share of claims rejected by the QC model")
    ax.set_xlim(0, min(1.0, max(tpp.syn_rate.max(), tpp.alarm_rate.max()) + 0.16))
    ax.legend(frameon=False, fontsize=13, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Same split by entity-type pair; the number is the excess\n"
                 "the QC model's own error does not explain",
                 fontsize=15, fontweight="bold")
    fig.savefig(FIG / "task5_per_entity_pair.png", dpi=200)
    plt.close(fig)

    # ---- source split on the synthetic side ------------------------------------
    if "implicit NoRelation" in set(syn.source):
        s = syn.groupby("source").rejected.agg(["sum", "count"])
        s["rate"] = s["sum"] / s["count"]
        print("\n\nTASK 3, synthetic claims by where the claim comes from:")
        print(s.round(4).to_string())
        out["synthetic_by_source"] = json.loads(s.round(4).reset_index()
                                                .to_json(orient="records"))

    (HERE / "task45_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    t.reset_index().to_csv(HERE / "task45_by_relation_type.csv", index=False)
    tp.reset_index().to_csv(HERE / "task45_by_entity_pair.csv", index=False)
    print(f"\nwrote task45_stats.json and two CSVs, figures in {FIG}")


if __name__ == "__main__":
    main()
