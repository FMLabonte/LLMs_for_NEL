"""Task 5, the controlled version: the SAME relation claim judged on real text and on
generated text.

The first cut of this table compared the QC model's behaviour on the real BioRED test set
with its behaviour on all of the synthetic data. Two things varied at once: the text
changed AND the set of papers, entity pairs and relation types changed. Fred's rule at
meeting 9 was to fix as many variables as possible and vary exactly one, so that cut can
only be a background check.

Here the unit is a relation triple (paper, entity id pair, relation type) that exists on
both sides:

  * on the real side it is a gold row of the perturbed BioRED test set, scored against the
    real abstract,
  * on the synthetic side it is the same triple stated in the generation prompt, scored
    against the abstract Qwen wrote for that paper.

Same paper, same entities, same relation type, same ground truth, same QC model. The only
thing that differs is who wrote the text. The difference in rejection rate is therefore
attributable to the generated text and to nothing else.

Synthetic claims carry surface names, not concept ids, so the names are resolved to BioRED
concept ids through the PubTator annotations of the same paper, the same way the implicit
NoRelation pairs were built.

Run:
    python task5_matched.py
"""
from __future__ import annotations

import json
import re
import sys
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
sys.path.insert(0, str(REPO))
from pubtator_parser import parse_pubtator  # noqa: E402

SYN_POS = REPO / "qc_filtering" / "synthetic_relation_scores.csv"
SYN_NEG = HERE / "norelation_scores.csv"
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


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def name_maps() -> dict[str, dict[str, str]]:
    m: dict[str, dict[str, str]] = {}
    for f in ["Train.PubTator", "Dev.PubTator", "Test.PubTator"]:
        _, anns, _ = parse_pubtator(REPO / "Data" / "BioRED" / f)
        for pmid, g in anns.groupby("pmid"):
            d = m.setdefault(str(pmid), {})
            for _, r in g.iterrows():
                d.setdefault(str(r["mention"]).strip().lower(), str(r["mesh_id"]))
    return m


def pair_label(ta, tb):
    x, y = sorted([TYPE_SHORT.get(ta, ta), TYPE_SHORT.get(tb, tb)])
    return f"{x}-{y}"


def main():
    maps = name_maps()

    # ---- real side ------------------------------------------------------------
    pred = pd.read_csv(REAL_PRED)
    rows = pd.read_csv(REAL_ROWS)
    assert len(pred) == len(rows) == 15199
    assert (pred.pmid.values == rows.pmid.values).all()
    real = rows.copy()
    real["prob"] = pred["prob"].values
    real["rejected"] = real.prob < CUTOFF
    gold = real[real.label == 1].copy()
    gold["key"] = [f"{p}|{'|'.join(sorted([str(a), str(b)]))}|{r}"
                   for p, a, b, r in zip(gold.pmid, gold.entity_a_id,
                                         gold.entity_b_id, gold.relation_type)]
    print(f"real gold claims on the BioRED test set: {len(gold):,} "
          f"({gold.key.nunique():,} distinct triples, {gold.pmid.nunique()} papers)")

    # ---- synthetic side, stated positives -------------------------------------
    syn = pd.read_csv(SYN_POS, dtype={"paper_id": str})
    syn = syn[(syn.split == "test") & (syn.relation_type.isin(FOUR))].copy()
    print(f"synthetic stated claims on the same 100 test papers: {len(syn):,}")

    def to_id(pid, name):
        return maps.get(str(pid), {}).get(str(name).strip().lower())

    syn["id_a"] = [to_id(p, n) for p, n in zip(syn.paper_id, syn.entity_a)]
    syn["id_b"] = [to_id(p, n) for p, n in zip(syn.paper_id, syn.entity_b)]
    unresolved = syn.id_a.isna() | syn.id_b.isna()
    print(f"  claims whose entity names do not resolve to a BioRED concept id: "
          f"{int(unresolved.sum()):,} ({unresolved.mean():.1%}), dropped")
    syn = syn[~unresolved].copy()
    syn["key"] = [f"{p}|{'|'.join(sorted([a, b]))}|{r}"
                  for p, a, b, r in zip(syn.paper_id, syn.id_a, syn.id_b,
                                        syn.relation_type)]
    syn["rejected"] = syn.prob_supported < CUTOFF

    # ---- the matched set ------------------------------------------------------
    shared = set(gold.key) & set(syn.key)
    print(f"\ntriples present on BOTH sides: {len(shared):,}")
    assert len(shared) > 200, "matched set too small to report"
    g = gold[gold.key.isin(shared)].copy()
    s = syn[syn.key.isin(shared)].copy()
    print(f"  real rows: {len(g):,}   synthetic rows: {len(s):,} "
          f"(2 generators x 3 generations per paper)")
    print(f"  papers covered: {g.pmid.nunique()}")

    key2type = dict(zip(g.key, g.relation_type))
    key2pair = dict(zip(g.key, [pair_label(a, b) for a, b in
                                zip(g.entity_a_type, g.entity_b_type)]))
    s["pair"] = s.key.map(key2pair)
    g["pair"] = g.key.map(key2pair)

    print(f"\nOVERALL, same triples, only the text differs:")
    print(f"  rejected when the text is the REAL abstract     : "
          f"{g.rejected.mean():.4f}  ({int(g.rejected.sum())}/{len(g)})")
    print(f"  rejected when the text is the GENERATED abstract: "
          f"{s.rejected.mean():.4f}  ({int(s.rejected.sum())}/{len(s)})")
    print(f"  difference: {s.rejected.mean()-g.rejected.mean():+.4f}")

    # per-triple, so paper size cannot dominate: average the synthetic verdict
    # within a triple first, then compare to the single real verdict.
    per = (s.groupby("key").rejected.mean().rename("syn")
           .to_frame().join(g.set_index("key").rejected.rename("real")))
    print(f"  paired over {len(per)} triples: real {per.real.mean():.4f}, "
          f"synthetic {per.syn.mean():.4f}, difference {per.syn.mean()-per.real.mean():+.4f}")
    both = per.dropna()
    disagree_worse = int(((both.syn > both.real)).sum())
    disagree_better = int(((both.syn < both.real)).sum())
    print(f"  triples where the generated text is rejected MORE: {disagree_worse}, "
          f"LESS: {disagree_better}, same: {len(both)-disagree_worse-disagree_better}")

    def table(groupcol):
        rr = g.groupby(groupcol, observed=True).rejected.agg(["sum", "count"])
        ss = s.groupby(groupcol, observed=True).rejected.agg(["sum", "count"])
        t = pd.DataFrame({
            "real_n": rr["count"], "real_rate": rr["sum"] / rr["count"],
            "syn_n": ss["count"], "syn_rate": ss["sum"] / ss["count"]})
        t["diff"] = t.syn_rate - t.real_rate
        t["real_lo"], t["real_hi"] = zip(*[wilson(int(k), int(n))
                                           for k, n in zip(rr["sum"], rr["count"])])
        t["syn_lo"], t["syn_hi"] = zip(*[wilson(int(k), int(n))
                                         for k, n in zip(ss["sum"], ss["count"])])
        return t[t.real_n >= 25]

    t_rel = table("relation_type").reindex(
        [r for r in FOUR if r in table("relation_type").index])
    print("\nBY RELATION TYPE (same triples on both sides)\n")
    print(t_rel[["real_n", "real_rate", "syn_n", "syn_rate", "diff"]].round(4).to_string())

    t_pair = table("pair").sort_values("diff")
    print("\nBY ENTITY-TYPE PAIR (same triples on both sides)\n")
    print(t_pair[["real_n", "real_rate", "syn_n", "syn_rate", "diff"]].round(4).to_string())

    # ---- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.3), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, t, title in [
            (axes[0], t_rel, "by relation type"),
            (axes[1], t_pair, "by entity-type pair")]:
        idx = list(t.index)
        x = np.arange(len(idx))
        ax.bar(x - 0.2, t.real_rate, 0.38, color="#4C72B0",
               yerr=[t.real_rate - t.real_lo, t.real_hi - t.real_rate], capsize=3,
               label="claim judged against the REAL abstract")
        ax.bar(x + 0.2, t.syn_rate, 0.38, color="#C44E52",
               yerr=[t.syn_rate - t.syn_lo, t.syn_hi - t.syn_rate], capsize=3,
               label="same claim judged against the GENERATED abstract")
        for i, (a_, b_) in enumerate(zip(t.real_hi, t.syn_hi)):
            d = t.syn_rate.iloc[i] - t.real_rate.iloc[i]
            ax.text(i, max(a_, b_) + 0.035, f"{d:+.2f}", ha="center",
                    fontsize=15, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(i, i) for i in idx], fontsize=14,
                           rotation=0 if len(idx) < 5 else 20)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("share rejected by the QC model" if ax is axes[0] else "")
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=13, loc="upper left")
    fig.suptitle("Same relation claims, same QC model, only the text differs "
                 f"({len(shared):,} matched triples on {g.pmid.nunique()} BioRED test papers)",
                 fontsize=12, fontweight="bold")
    fig.savefig(FIG / "task5_matched.png", dpi=200)
    plt.close(fig)

    out = {
        "n_matched_triples": len(shared),
        "n_real_rows": int(len(g)), "n_synthetic_rows": int(len(s)),
        "papers": int(g.pmid.nunique()),
        "overall_real_rejection": round(float(g.rejected.mean()), 4),
        "overall_synthetic_rejection": round(float(s.rejected.mean()), 4),
        "paired_real": round(float(per.real.mean()), 4),
        "paired_synthetic": round(float(per.syn.mean()), 4),
        "triples_worse_on_synthetic": disagree_worse,
        "triples_better_on_synthetic": disagree_better,
        "by_relation_type": json.loads(t_rel.round(4).reset_index().to_json(orient="records")),
        "by_entity_pair": json.loads(t_pair.round(4).reset_index().to_json(orient="records")),
    }
    (HERE / "task5_matched_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    t_rel.reset_index().to_csv(HERE / "task5_matched_relation_type.csv", index=False)
    t_pair.reset_index().to_csv(HERE / "task5_matched_entity_pair.csv", index=False)
    print(f"\nwrote task5_matched_stats.json and figures/task5_matched.png")


if __name__ == "__main__":
    main()
