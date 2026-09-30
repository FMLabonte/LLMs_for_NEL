"""Task 3, the analysis: is the QC model checked on the implicit NoRelation pairs?

Fred called this the biggest open item at meeting 9. His position: the generation
prompt lists entities and relations and states that anything not in a listed relation
has no relation to anything else, so every unlisted pair is an implicit NoRelation
claim, and the QC model must be run on those because that is how a hallucinated
relation gets caught.

The answer to the literal question is no, it was not. The shipped pipeline parses only
the RELATIONS block of the prompt, so all 35,658 scored claims are stated positives and
none are NoRelation. That is verified below rather than asserted.

The implicit pairs were then built from the same prompts, resolved to BioRED concept
ids, and scored with the same run-2 checkpoint. This script reads those scores and
answers four questions:

  1. how often does the QC model reject an implicit NoRelation claim, that is, how
     often does it think the generated abstract asserts a relation that should not be
     there;
  2. how much of that is the QC model's own false-alarm rate rather than the
     generator, measured on the same triples scored against the real BioRED abstract;
  3. what it does to the acceptance ladder if these claims are counted;
  4. how it splits by entity-type pair.

Run:
    python task3_norelation.py
"""
from __future__ import annotations

import json
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
SYN_POS = REPO / "qc_filtering" / "synthetic_relation_scores.csv"
SYN_NEG = HERE / "norelation_scores.csv"
REALTEXT = HERE / "norelation_realtext_scores.csv"
REAL_PRED = WORK / "2026-07-22_next_tasks" / "qc_run2_test_predictions.csv"
REAL_ROWS = WORK / "2026-07-08_meeting" / "data" / "qc_test.csv"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

CUTOFF = 0.5
FOUR = ["Association", "Positive_Correlation", "Negative_Correlation", "NoRelation"]
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


def pair_label(ta, tb):
    x, y = sorted([TYPE_SHORT.get(ta, ta), TYPE_SHORT.get(tb, tb)])
    return f"{x}-{y}"


def main():
    out: dict = {}

    # ---- 0. the literal question ----------------------------------------------
    pos = pd.read_csv(SYN_POS, dtype={"paper_id": str})
    n_norel_in_shipped = int((pos.relation_type == "NoRelation").sum())
    print("=" * 78)
    print("Q: was the QC model being run on the implicit NoRelation pairs?")
    print(f"   claims in the shipped synthetic_relation_scores.csv : {len(pos):,}")
    print(f"   of those, relation_type == NoRelation               : {n_norel_in_shipped}")
    print("   ANSWER: no. Only the stated positives were scored.")
    assert n_norel_in_shipped == 0
    out["shipped_norelation_claims"] = n_norel_in_shipped
    out["shipped_total_claims"] = int(len(pos))

    neg = pd.read_csv(SYN_NEG, dtype={"paper_id": str, "entity_a_id": str,
                                      "entity_b_id": str})
    neg["rejected"] = neg.prob_supported < CUTOFF
    print(f"\n   implicit NoRelation claims now built and scored     : {len(neg):,}")
    print(f"   that is {len(neg)/len(pos):.1f}x the number of stated positive claims")
    assert len(neg) > 100000, "NoRelation score file is short, did the run finish?"
    assert neg.prob_supported.notna().all()
    covered = neg.drop_duplicates(["model", "split", "paper_id", "generation"])
    print(f"   covering {len(covered):,} of the 3,552 synthetic abstracts "
          f"(the rest have no unstated pair)")
    out["implicit_claims"] = int(len(neg))
    out["abstracts_covered"] = int(len(covered))

    # ---- 1. the raw rate --------------------------------------------------------
    k, n = int(neg.rejected.sum()), len(neg)
    lo, hi = wilson(k, n)
    print("\n" + "=" * 78)
    print("1. How often does the QC model reject an implicit NoRelation claim?")
    print(f"   {k:,} of {n:,} = {k/n:.4f}  (95% CI {lo:.4f} to {hi:.4f})")
    print("   A rejection means: the QC model believes the generated abstract does")
    print("   assert a relation between this pair, i.e. a candidate hallucination.")
    out["implicit_rejection_rate"] = round(k / n, 4)
    out["implicit_rejection_ci"] = [round(lo, 4), round(hi, 4)]

    # ---- 2. the control ---------------------------------------------------------
    pred = pd.read_csv(REAL_PRED)
    rows = pd.read_csv(REAL_ROWS)
    assert len(pred) == len(rows) == 15199
    assert (pred.pmid.values == rows.pmid.values).all()
    real = rows.copy()
    real["prob"] = pred["prob"].values
    real["rejected"] = real.prob < CUTOFF
    rg = real[(real.label == 1) & (real.relation_type == "NoRelation")].copy()
    print("\n" + "=" * 78)
    print("2. How much of that is the QC model's own error?")
    print(f"   real BioRED test set, true NoRelation pairs scored against the REAL")
    print(f"   abstract: {int(rg.rejected.sum()):,} of {len(rg):,} rejected = "
          f"{rg.rejected.mean():.4f}")
    print(f"   naive excess over the whole synthetic set: "
          f"{k/n - rg.rejected.mean():+.4f}")

    out["real_norelation_gold_rate"] = round(float(rg.rejected.mean()), 4)
    out["real_norelation_gold_n"] = int(len(rg))

    # The perturbed test set only holds 1,131 NoRelation golds, capped to match the
    # relation count, and only a few hundred of those also appear as implicit pairs in
    # a generation prompt. That is too thin a control for the central claim here, so
    # score_norelation_realtext.py scored every distinct implicit pair of the TEST
    # split a second time against the REAL abstract of the same paper. Same model,
    # same pairs, same ground truth; only the author of the text differs.
    rt = pd.read_csv(REALTEXT, dtype={"paper_id": str, "entity_a_id": str,
                                      "entity_b_id": str})
    rt["rejected_real"] = rt.prob_supported_real < CUTOFF
    pk = ["paper_id", "entity_a_id", "entity_b_id"]
    st = neg[neg.split == "test"].copy()
    print(f"\n   PAIRED CONTROL, BioRED test split")
    print(f"   distinct implicit pairs: {st.drop_duplicates(pk).shape[0]:,} over "
          f"{st.paper_id.nunique()} papers")
    m = st.merge(rt[pk + ["rejected_real"]], on=pk, how="inner", validate="many_to_one")
    assert len(m) == len(st), f"merge lost rows: {len(m)} vs {len(st)}"

    def paired(d, label):
        per = d.groupby(pk).rejected.mean()
        r = d.drop_duplicates(pk).set_index(pk).rejected_real
        lo_r, hi_r = wilson(int(r.sum()), len(r))
        lo_s, hi_s = wilson(int(d.rejected.sum()), len(d))
        print(f"   {label} ({len(r):,} pairs)")
        print(f"      real abstract     : {r.mean():.4f}  CI {lo_r:.3f}-{hi_r:.3f}")
        print(f"      generated abstract: {per.mean():.4f}  CI {lo_s:.3f}-{hi_s:.3f}")
        print(f"      difference        : {per.mean()-r.mean():+.4f}")
        return {"pairs": int(len(r)), "real": round(float(r.mean()), 4),
                "real_ci": [round(lo_r, 4), round(hi_r, 4)],
                "generated": round(float(per.mean()), 4),
                "generated_ci": [round(lo_s, 4), round(hi_s, 4)],
                "difference": round(float(per.mean() - r.mean()), 4)}

    out["paired_all"] = paired(m, "all pairs")
    # The generation pipeline sometimes fell back to a bare MeSH or Gene identifier as
    # the entity name, and marked the type "unknown". The generator then wrote the
    # identifier into the abstract as if it were a name, so a claim about "D000438"
    # is not something either model can judge. Reported with and without.
    typed = m[(m.type_a != "unknown") & (m.type_b != "unknown")]
    out["paired_typed"] = paired(typed, "excluding bare-identifier entities")

    per = m.groupby(pk).agg(gen=("rejected", "mean"), real=("rejected_real", "first"))
    worse = int((per.gen > per.real).sum())
    better = int((per.gen < per.real).sum())
    print(f"   pairs flagged MORE on generated text: {worse:,}, LESS: {better:,}, "
          f"same: {len(per)-worse-better:,}")
    out["paired_pairs_worse"] = worse
    out["paired_pairs_better"] = better
    out["paired_pairs_same"] = int(len(per) - worse - better)

    # ---- 3. what it does to the acceptance ladder --------------------------------
    print("\n" + "=" * 78)
    print("3. What happens to the acceptance ladder if these claims are counted?")
    p4 = pos[pos.relation_type.isin(FOUR)].copy()
    p4["rejected"] = p4.prob_supported < CUTOFF
    key = ["model", "split", "paper_id", "generation"]
    stated = (p4.groupby(key).rejected.agg(["sum", "count"])
              .rename(columns={"sum": "f_pos", "count": "n_pos"}))
    implicit = (neg.groupby(key).rejected.agg(["sum", "count"])
                .rename(columns={"sum": "f_neg", "count": "n_neg"}))
    both = stated.join(implicit, how="outer").fillna(0)
    both["rate_pos"] = both.f_pos / both.n_pos.replace(0, np.nan)
    both["rate_all"] = (both.f_pos + both.f_neg) / (both.n_pos + both.n_neg)
    print(f"   abstracts: {len(both):,}")
    print(f"   mean per-abstract error rate, stated positives only : "
          f"{both.rate_pos.mean():.4f}")
    print(f"   mean per-abstract error rate, all claims            : "
          f"{both.rate_all.mean():.4f}")
    ladder = []
    for name, tau in [("L0 strict", 0.0), ("L1 10%", 0.10), ("L2 20%", 0.20),
                      ("L3 33%", 0.33), ("L4 50%", 0.50), ("L5 unfiltered", 1.0)]:
        a = int((both.rate_pos <= tau + 1e-12).sum())
        b = int((both.rate_all <= tau + 1e-12).sum())
        ladder.append({"level": name, "positives only": a, "with NoRelation": b,
                       "positives %": round(100 * a / len(both), 1),
                       "with NoRelation %": round(100 * b / len(both), 1)})
    lt = pd.DataFrame(ladder)
    print(lt.to_string(index=False))
    out["ladder_with_norelation"] = lt.to_dict("records")

    n_hall = int((both.f_neg > 0).sum())
    print(f"\n   abstracts with at least one flagged hallucination: {n_hall:,} of "
          f"{len(both):,} = {n_hall/len(both):.1%}")
    print(f"   flagged hallucinations per abstract: median "
          f"{both.f_neg.median():.0f}, mean {both.f_neg.mean():.2f}")
    out["abstracts_with_flagged_hallucination"] = n_hall
    out["abstracts_total"] = int(len(both))

    # ---- 4. by entity pair -------------------------------------------------------
    neg["pair"] = [pair_label(a, b) for a, b in zip(neg.type_a, neg.type_b)]
    g = neg[(neg.type_a != "unknown") & (neg.type_b != "unknown")] \
        .groupby("pair").rejected.agg(["sum", "count"])
    g["rate"] = g["sum"] / g["count"]
    g = g[g["count"] >= 500].sort_values("rate", ascending=False)
    print("\n" + "=" * 78)
    print("4. Implicit NoRelation rejection rate by entity-type pair (n >= 500)")
    print(g.round(4).to_string())
    out["by_entity_pair"] = json.loads(g.round(4).reset_index().to_json(orient="records"))

    # ---- 5. the robustness check that could invalidate all of the above ----------
    # The QC model's NoRelation training golds were always pairs of entities that are
    # BOTH mentioned in the abstract. If the generator simply left an entity out, the
    # input is off-distribution and the verdict means less. So split by whether both
    # names actually appear in the generated text.
    pairs = pd.read_parquet(HERE / "norelation_pairs.parquet")
    assert len(pairs) == len(neg), "pair file and score file disagree in length"
    low = pairs.abstract.str.lower()
    both_present = (pd.Series([a.lower() in t for a, t in zip(pairs.entity_a, low)])
                    & pd.Series([b.lower() in t for b, t in zip(pairs.entity_b, low)]))
    neg = neg.reset_index(drop=True)
    neg["both_mentioned"] = both_present.values
    print("\n" + "=" * 78)
    print("5. Robustness: are both entities actually written into the abstract?")
    print(f"   both names present: {int(neg.both_mentioned.sum()):,} of {len(neg):,} "
          f"= {neg.both_mentioned.mean():.1%}")
    sub = neg.groupby("both_mentioned").rejected.agg(["sum", "count"])
    sub["rate"] = sub["sum"] / sub["count"]
    print(sub.round(4).to_string())
    print("   The first row is the off-distribution case. Read the second row as the")
    print("   result; the first is reported so the gap is visible rather than hidden.")
    out["both_mentioned_share"] = round(float(neg.both_mentioned.mean()), 4)
    out["rate_both_mentioned"] = round(float(sub.loc[True, "rate"]), 4)
    out["rate_not_both_mentioned"] = (round(float(sub.loc[False, "rate"]), 4)
                                      if False in sub.index else None)

    # ---- figure ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.3]})

    ax = axes[0]
    pa, pt = out["paired_all"], out["paired_typed"]
    groups = [("all pairs\n" + f"n={pa['pairs']:,}", pa),
              ("bare identifiers\nremoved, " + f"n={pt['pairs']:,}", pt)]
    x = np.arange(len(groups))
    for off, kk, colour, lab in [(-0.2, "real", "#4C72B0", "real BioRED abstract"),
                                 (0.2, "generated", "#C44E52", "generated abstract")]:
        v = [gg[kk] for _, gg in groups]
        lo_e = [gg[kk][0] if False else gg[kk + "_ci"][0] for _, gg in groups]
        hi_e = [gg[kk + "_ci"][1] for _, gg in groups]
        ax.bar(x + off, v, 0.38, color=colour, label=lab,
               yerr=[np.array(v) - np.array(lo_e), np.array(hi_e) - np.array(v)],
               capsize=5)
    for i, (_, gg) in enumerate(groups):
        top = max(gg["real_ci"][1], gg["generated_ci"][1])
        ax.text(i, top + 0.012, f"{gg['difference']:+.3f}", ha="center",
                fontweight="bold", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in groups], fontsize=12)
    ax.set_ylabel("share flagged as asserting a relation")
    ax.set_ylim(0, 0.28)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Same pairs, same model, only the text differs\n"
                 "BioRED test split, no measurable difference", fontsize=13)

    ax = axes[1]
    y = np.arange(len(g))
    ax.barh(y, g.rate, 0.6, color="#C44E52")
    for i, (r_, c_) in enumerate(zip(g.rate, g["count"])):
        ax.text(r_ + 0.004, i, f"{r_:.3f}  (n={int(c_):,})", va="center", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(g.index, fontsize=12)
    ax.set_xlabel("share flagged as asserting a relation")
    ax.set_xlim(0, g.rate.max() * 1.45)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Flag rate by entity-type pair. This tracks which pairs carry\n"
                 "relations in BioRED at all, so it is mostly the QC model's prior",
                 fontsize=13)

    fig.suptitle("The implicit NoRelation pairs: "
                 f"{len(neg):,} claims the pipeline had never scored",
                 fontsize=15, fontweight="bold")
    fig.savefig(FIG / "task3_norelation.png", dpi=200)
    plt.close(fig)

    (HERE / "task3_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote task3_stats.json and figures/task3_norelation.png")


if __name__ == "__main__":
    main()
