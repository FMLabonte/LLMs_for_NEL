"""Fred's meeting-10 question: is Chemical-Gene hard in itself, or is it just mostly
Negative_Correlation?

He asked for the label distribution per entity pair, as percentages over the four
classes. That distribution alone does not answer his question, so this script does both
halves:

  1. the distribution he asked for, over {Association, Positive_Correlation,
     Negative_Correlation, NoRelation}, per unordered entity-type pair;
  2. the test the distribution is for. If a pair only looks hard because of its class
     mix, then the false-alarm rate predicted from that mix should match the one
     actually measured on the pair. Where the measured rate runs above the predicted
     one, the pair is hard beyond its mix.

Both halves are QC EVALUATION: the perturbed BioRED test split, where every claim
carries a label, so a rejected true claim is a false alarm and the rate is a real error
rate. The synthetic side is reported alongside for reference and is QC APPLICATION, so
it gives a rejection rate and nothing more.

Conditions, as always: run 2 weights, test split, cut-off 0.5, per claim not per
abstract, and rows are grouped by the CLAIMED relation type rather than the true one.

Run:
    /opt/homebrew/Caskroom/miniconda/base/bin/python entity_pair_labels.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LAB = HERE.parent.parent
WORK = LAB / "work"
REAL_PRED = WORK / "2026-07-22_next_tasks" / "qc_run2_test_predictions.csv"
REAL_ROWS = WORK / "2026-07-08_meeting" / "data" / "qc_test.csv"
SYN_POS = LAB / "LLMs_for_NEL" / "qc_filtering" / "synthetic_relation_scores.csv"
SYN_NEG = WORK / "2026-09-09_overnight" / "norelation_scores.csv"

CUTOFF = 0.5
FOUR = ["Association", "Positive_Correlation", "Negative_Correlation", "NoRelation"]
SHORT = {"Association": "Assoc", "Positive_Correlation": "Pos",
         "Negative_Correlation": "Neg", "NoRelation": "NoRel"}
TYPE_SHORT = {"GeneOrGeneProduct": "Gene", "ChemicalEntity": "Chem",
              "DiseaseOrPhenotypicFeature": "Disease", "SequenceVariant": "Variant",
              "OrganismTaxon": "Species", "CellLine": "CellLine"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pair_label(ta, tb):
    x, y = sorted([TYPE_SHORT.get(ta, ta), TYPE_SHORT.get(tb, tb)])
    return f"{x}-{y}"


def load_real():
    pred = pd.read_csv(REAL_PRED)
    rows = pd.read_csv(REAL_ROWS)
    assert len(pred) == len(rows) == 15199, "real test set changed size"
    for c in ["pmid", "relation_type", "entity_a_id", "entity_b_id", "label"]:
        assert (pred[c].astype(str).values == rows[c].astype(str).values).all(), \
            f"real test files disagree on {c}"
    df = rows.copy()
    df["prob"] = pred["prob"].values
    df["rejected"] = df.prob < CUTOFF
    df["pair"] = [pair_label(a, b) for a, b in zip(df.entity_a_type, df.entity_b_type)]
    return df


def main():
    df = load_real()
    gold = df[df.label == 1].copy()          # true claims, so rejection = false alarm
    print(f"QC evaluation, test split: {len(df):,} claims, "
          f"{len(gold):,} of them true\n")

    # ---- 1. the distribution Fred asked for -------------------------------------
    print("=" * 86)
    print("1. LABEL DISTRIBUTION PER ENTITY PAIR, percentages over the four classes")
    print("=" * 86)
    ct = pd.crosstab(gold.pair, gold.relation_type)
    for c in FOUR:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[FOUR]
    ct["n"] = ct.sum(axis=1)
    ct = ct.sort_values("n", ascending=False)
    pct = ct[FOUR].div(ct.n, axis=0) * 100

    hdr = f"{'entity pair':<16}{'n':>7}" + "".join(f"{SHORT[c]:>9}" for c in FOUR)
    print(hdr)
    print("-" * len(hdr))
    for p in ct.index:
        print(f"{p:<16}{ct.loc[p,'n']:>7}" +
              "".join(f"{pct.loc[p,c]:>8.1f}%" for c in FOUR))
    tot = ct[FOUR].sum()
    print("-" * len(hdr))
    print(f"{'all':<16}{int(ct.n.sum()):>7}" +
          "".join(f"{tot[c]/ct.n.sum()*100:>8.1f}%" for c in FOUR))

    # ---- 2. does the mix explain the difficulty? --------------------------------
    print()
    print("=" * 86)
    print("2. IS THE PAIR HARD IN ITSELF? observed false alarms against the rate its")
    print("   class mix predicts (QC evaluation, so these are real error rates)")
    print("=" * 86)
    by_class = gold.groupby("relation_type").rejected.agg(["sum", "count"])
    by_class["rate"] = by_class["sum"] / by_class["count"]
    print("per-class false-alarm rate, pooled over all pairs:")
    for c in FOUR:
        if c in by_class.index:
            print(f"   {SHORT[c]:<7} {by_class.loc[c,'rate']:.3f}  "
                  f"({int(by_class.loc[c,'sum']):,} of {int(by_class.loc[c,'count']):,})")

    rows_out = []
    for p in ct.index:
        g = gold[gold.pair == p]
        obs_k, obs_n = int(g.rejected.sum()), len(g)
        obs = obs_k / obs_n
        exp = sum((pct.loc[p, c] / 100) * by_class.loc[c, "rate"]
                  for c in FOUR if c in by_class.index)
        lo, hi = wilson(obs_k, obs_n)
        rows_out.append({
            "pair": p, "n": obs_n,
            **{SHORT[c]: round(float(pct.loc[p, c]), 1) for c in FOUR},
            "observed": round(obs, 3), "predicted_from_mix": round(float(exp), 3),
            "residual": round(obs - float(exp), 3),
            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
            "mix_explains": "yes" if lo <= exp <= hi else "no",
        })
    out = pd.DataFrame(rows_out)

    print(f"\n{'entity pair':<16}{'n':>7}{'observed':>10}{'predicted':>11}"
          f"{'residual':>10}{'95% CI':>18}  mix explains")
    print("-" * 86)
    for r in rows_out:
        print(f"{r['pair']:<16}{r['n']:>7}{r['observed']:>10.3f}"
              f"{r['predicted_from_mix']:>11.3f}{r['residual']:>+10.3f}"
              f"{'[' + format(r['ci_lo'], '.3f') + ', ' + format(r['ci_hi'], '.3f') + ']':>18}"
              f"  {r['mix_explains']}")

    print("\nReading: 'mix explains yes' means the rate predicted from the pair's class")
    print("mix falls inside the pair's own confidence interval, so the pair carries no")
    print("difficulty beyond the classes it happens to contain. A positive residual with")
    print("'no' is a pair that is genuinely harder than its mix.")

    # ---- the synthetic side, for reference ---------------------------------------
    pos = pd.read_csv(SYN_POS, dtype={"paper_id": str})
    pos = pos[pos.relation_type.isin(FOUR)].copy()
    neg = pd.read_csv(SYN_NEG, dtype={"paper_id": str})
    cols = ["relation_type", "type_a", "type_b", "prob_supported"]
    syn = pd.concat([pos[cols], neg[cols]], ignore_index=True)
    syn["rejected"] = syn.prob_supported < CUTOFF
    syn["pair"] = [pair_label(a, b) for a, b in zip(syn.type_a, syn.type_b)]
    s = syn.groupby("pair").rejected.agg(["sum", "count"])
    s["rejection_rate"] = (s["sum"] / s["count"]).round(3)
    out = out.merge(s[["count", "rejection_rate"]].rename(
        columns={"count": "syn_n", "rejection_rate": "syn_rejection_rate"}),
        left_on="pair", right_index=True, how="left")

    out.to_csv(HERE / "entity_pair_labels.csv", index=False)
    counts = ct.reset_index()
    counts.to_csv(HERE / "entity_pair_counts.csv", index=False)

    # ---- appendix table ----------------------------------------------------------
    tex = [
        r"\begin{table}[H]\centering",
        r"\caption{Label distribution per entity-type pair on the true claims of the",
        r"perturbed BioRED test split, as percentages over the four classes, with the",
        r"QC model's false-alarm rate on that pair beside the rate predicted from the",
        r"pair's class mix alone. A residual near zero means the pair is no harder than",
        r"the classes it contains. QC evaluation throughout, run 2 weights, cut-off",
        r"0.5.}",
        r"\label{tab:pairlabels}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"pair & $n$ & Assoc & Pos & Neg & NoRel & observed & predicted & residual \\",
        r"\midrule",
    ]
    for r in rows_out:
        tex.append(
            rf"{r['pair']} & {r['n']:,} & {r['Assoc']:.1f}\% & {r['Pos']:.1f}\% & "
            rf"{r['Neg']:.1f}\% & {r['NoRel']:.1f}\% & {r['observed']:.3f} & "
            rf"{r['predicted_from_mix']:.3f} & {r['residual']:+.3f} \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE / "table.tex").write_text("\n".join(tex) + "\n")

    stats = {
        "stage": "QC evaluation (perturbed BioRED test split)",
        "cutoff": CUTOFF,
        "true_claims": int(len(gold)),
        "per_class_false_alarm": {SHORT[c]: round(float(by_class.loc[c, "rate"]), 4)
                                  for c in FOUR if c in by_class.index},
        "pairs": rows_out,
    }
    (HERE / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"\nwritten: entity_pair_labels.csv, entity_pair_counts.csv, table.tex, stats.json")


if __name__ == "__main__":
    main()
