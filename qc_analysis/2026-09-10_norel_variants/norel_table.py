"""The NoRelation comparison table for the report.

Three questions with three different answers, and the table has to keep them apart:

  1. did the shipped filter reject on NoRelation at all?  No, never.
  2. does the new filter reject on NoRelation?            Yes, and on nothing else.
  3. is that rejection attributable to the generator?     No.

Question 3 is the one that needs both stages on the same claims, per the rule that any
claim about the GENERATOR needs a labelled arm next to the unlabelled one:

  * QC EVALUATION arm: the same implicit pairs scored against the REAL BioRED abstract.
    A pair that BioRED does not annotate as related is a NoRelation by the same
    closed-world assumption the NoRelation golds already use, so a rejection here is a
    false alarm and the rate is a real error rate.
  * QC APPLICATION arm: the same pairs scored against the generated abstract. No labels,
    so this is a rejection rate and nothing else.

The difference between the two is the only part attributable to generation.

Writes tables.tex (body + appendix) and table_stats.json.

Run:
    /opt/homebrew/Caskroom/miniconda/base/bin/python norel_table.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LAB = HERE.parent.parent
QC = LAB / "LLMs_for_NEL" / "qc_filtering"
OVERNIGHT = LAB / "work" / "2026-09-09_overnight"

CUTOFF = 0.5
KEY = ["model", "split", "paper_id", "generation"]
TOP3 = {"Association", "Positive_Correlation", "Negative_Correlation"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    d = pd.read_csv(QC / "abstract_decisions_norel.csv", dtype={"paper_id": str})
    # The CSV shipped these as columns until 2026-09-13, when it was cut back to
    # `passed` (dynamic) and `passed_strict`. Both are still exactly derivable from
    # the raw counts, so this analysis is unchanged.
    d["keep_stated_only"] = d.failed_stated == 0
    d["keep_norel_aware"] = d.passed_strict
    pos = pd.read_csv(QC / "synthetic_relation_scores.csv", dtype={"paper_id": str})
    neg = pd.read_csv(OVERNIGHT / "norelation_scores.csv", dtype={"paper_id": str})
    real = pd.read_csv(OVERNIGHT / "norelation_realtext_scores.csv", dtype={"paper_id": str})
    neg["failed"] = neg.prob_supported < CUTOFF
    real["failed_real"] = real.prob_supported_real < CUTOFF
    for c in ("entity_a_id", "entity_b_id"):
        neg[c] = neg[c].astype(str)
        real[c] = real[c].astype(str)

    s: dict = {"stage_note": "real-text arm = QC evaluation, generated arm = QC application"}

    # ---- 1. did the shipped filter ever see a NoRelation claim? ------------------
    n_norel_shipped = int((pos.relation_type == "NoRelation").sum())
    assert n_norel_shipped == 0
    print("=" * 78)
    print("1. THE SHIPPED FILTER")
    print("=" * 78)
    print(f"claims it judged                  : {len(pos):,}")
    print(f"of those, NoRelation              : {n_norel_shipped}")
    print("So the shipped filter never rejected anything on NoRelation. It could not.")
    s["shipped_claims"] = int(len(pos))
    s["shipped_norelation_claims"] = n_norel_shipped

    # ---- 2. does the new filter reject on NoRelation? ----------------------------
    print("\n" + "=" * 78)
    print("2. THE NoRelation-AWARE FILTER: what the extra rejections are made of")
    print("=" * 78)
    lost = d[d.keep_stated_only & ~d.keep_norel_aware]
    print(f"abstracts                         : {len(d):,}")
    print(f"kept, stated relations only       : {int(d.keep_stated_only.sum()):,} "
          f"({d.keep_stated_only.mean()*100:.1f}%)")
    print(f"kept, NoRelation aware            : {int(d.keep_norel_aware.sum()):,} "
          f"({d.keep_norel_aware.mean()*100:.1f}%)")
    print(f"newly rejected                    : {len(lost):,}")
    print(f"  of those, stated claims all pass: {int((lost.failed_stated == 0).sum()):,} "
          f"({(lost.failed_stated == 0).mean()*100:.0f}%)")
    print(f"  median implicit failures each   : {lost.failed_implicit.median():.0f}")
    print("So yes: every extra rejection is a NoRelation rejection and nothing else.")
    s["kept_stated_only"] = int(d.keep_stated_only.sum())
    s["kept_norel_aware"] = int(d.keep_norel_aware.sum())
    s["newly_rejected"] = int(len(lost))
    s["newly_rejected_purely_norelation"] = int((lost.failed_stated == 0).sum())

    # ---- 3. the control: same pairs, real text vs generated text -----------------
    print("\n" + "=" * 78)
    print("3. THE CONTROL: is that rejection attributable to the generator?")
    print("=" * 78)
    pk = KEY + ["entity_a_id", "entity_b_id"]
    m = real.merge(neg[pk + ["failed"]].rename(columns={"failed": "failed_gen"}),
                   on=pk, how="inner")
    kr, kg, n = int(m.failed_real.sum()), int(m.failed_gen.sum()), len(m)
    lr, hr = wilson(kr, n)
    lg, hg = wilson(kg, n)
    print(f"paired pairs {n:,} over {m.groupby(KEY).ngroups} abstracts, test split")
    print(f"  REAL BioRED text, false alarms (QC evaluation) : {kr/n:.4f} "
          f"[{lr:.4f}, {hr:.4f}]")
    print(f"  GENERATED text, rejections (QC application)    : {kg/n:.4f} "
          f"[{lg:.4f}, {hg:.4f}]")
    print(f"  difference attributable to generation          : {(kg-kr)/n:+.4f}")
    b = int((m.failed_gen & ~m.failed_real).sum())
    c = int((~m.failed_gen & m.failed_real).sum())
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    print(f"  McNemar on the same pairs: gen-only {b}, real-only {c}, chi2 {chi2:.2f} "
          f"(3.84 = p 0.05)")

    ab = m.groupby(KEY).agg(fr=("failed_real", "sum"), fg=("failed_gen", "sum")).reset_index()
    print(f"\nabstract level, judged on the implicit pairs alone:")
    print(f"  REAL BioRED abstract survives  : {int((ab.fr==0).sum())} of {len(ab)} "
          f"({(ab.fr==0).mean()*100:.1f}%)")
    print(f"  generated abstract survives    : {int((ab.fg==0).sum())} of {len(ab)} "
          f"({(ab.fg==0).mean()*100:.1f}%)")
    print("\nThe rule discards three quarters of abstracts that cannot contain an")
    print("invented relation, at the same rate as the generated ones.")
    s["paired_pairs"] = n
    s["paired_abstracts"] = int(m.groupby(KEY).ngroups)
    s["realtext_false_alarm_rate"] = round(kr / n, 4)
    s["generated_rejection_rate"] = round(kg / n, 4)
    s["attributable_to_generation"] = round((kg - kr) / n, 4)
    s["mcnemar_chi2"] = round(chi2, 2)
    s["survive_real"] = round(float((ab.fr == 0).mean()), 3)
    s["survive_generated"] = round(float((ab.fg == 0).mean()), 3)

    # full-corpus implicit rejection rate, for the record
    ka, na = int(neg.failed.sum()), len(neg)
    la, ha = wilson(ka, na)
    print(f"\nfor the record, all {na:,} implicit pairs against generated text: "
          f"{ka/na:.4f} [{la:.4f}, {ha:.4f}]")
    print("(higher than the paired 0.14 because the paired subset is test split only)")
    s["all_implicit_rejection_rate"] = round(ka / na, 4)
    s["all_implicit_pairs"] = na

    # ---- the tables --------------------------------------------------------------
    per = []
    for (mdl, sp), grp in d.groupby(["model", "split"]):
        per.append({
            "model": mdl.replace("qwen3_", "Qwen3-").replace("b", "B"), "split": sp,
            "abstracts": len(grp),
            "stated": int(grp.keep_stated_only.sum()),
            "stated_pct": grp.keep_stated_only.mean() * 100,
            "norel": int(grp.keep_norel_aware.sum()),
            "norel_pct": grp.keep_norel_aware.mean() * 100,
            "lost": int((grp.keep_stated_only & ~grp.keep_norel_aware).sum()),
            "med_k_all": grp.n_implicit.median(),
            "med_k_kept": grp[grp.keep_norel_aware].n_implicit.median(),
        })
    per_df = pd.DataFrame(per)
    per_df.to_csv(HERE / "norel_by_model_split.csv", index=False)

    # The table Houman asked for: how many abstracts pass, with and without the
    # implicit NoRelation pairs counted, per generator and split, with a total row.
    body = [
        r"\begin{table}[H]\centering",
        r"\caption{Abstracts passing the filter when only the relations the prompt lists",
        r"are judged, and when the implicit NoRelation pairs are judged as well. The",
        r"rule is the same in both columns, reject the abstract if any claim fails at",
        r"0.5; only the set of claims looked at changes. Every abstract in the last",
        r"column fails on NoRelation alone, with its stated relations intact. These are",
        r"QC application figures, so they are rejection rates and carry no error",
        r"interpretation.}",
        r"\label{tab:norel}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"generator & split & abstracts & stated only & with NoRelation & newly rejected \\",
        r"\midrule",
    ]
    for r_ in per:
        body.append(
            rf"{r_['model']} & {r_['split']} & {r_['abstracts']:,} & "
            rf"{r_['stated']} ({r_['stated_pct']:.1f}\%) & "
            rf"{r_['norel']} ({r_['norel_pct']:.1f}\%) & {r_['lost']} \\")
    body += [
        r"\midrule",
        rf"\multicolumn{{2}}{{l}}{{all}} & {len(d):,} & "
        rf"{int(d.keep_stated_only.sum()):,} ({d.keep_stated_only.mean()*100:.1f}\%) & "
        rf"{int(d.keep_norel_aware.sum()):,} ({d.keep_norel_aware.mean()*100:.1f}\%) & "
        rf"{len(lost):,} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    # The control belongs next to it, otherwise the table above reads as a quality
    # result rather than as a false-alarm result.
    app = [
        r"\begin{table}[H]\centering",
        rf"\caption{{Control for Table~\ref{{tab:norel}}. The same {n:,} implicit pairs",
        r"scored against the real BioRED abstract of the same paper and against the",
        r"generated one. The real-text arm is a QC evaluation measurement, since a pair",
        r"BioRED does not annotate as related is a NoRelation under the same closed-world",
        r"assumption used to build the NoRelation golds, so its rejections are false",
        r"alarms. The generated arm is QC application. The two arms agree, so the",
        r"rejections in Table~\ref{tab:norel} are not attributable to generation.}",
        r"\label{tab:norelcontrol}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r" & real BioRED text & generated text \\",
        r"\midrule",
        rf"implicit pairs rejected & {kr/n:.3f} & {kg/n:.3f} \\",
        rf"abstracts passing on implicit pairs alone & {int((ab.fr==0).sum())} of {len(ab)} "
        rf"({(ab.fr==0).mean()*100:.1f}\%) & {int((ab.fg==0).sum())} of {len(ab)} "
        rf"({(ab.fg==0).mean()*100:.1f}\%) \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    (HERE / "tables.tex").write_text("\n".join(body) + "\n\n" + "\n".join(app) + "\n")
    (HERE / "table_stats.json").write_text(json.dumps(s, indent=2) + "\n")

    print("\n" + "=" * 78)
    print("PER MODEL AND SPLIT")
    print("=" * 78)
    print(per_df.to_string(index=False))
    print(f"\nwritten: tables.tex, norel_by_model_split.csv, table_stats.json")


if __name__ == "__main__":
    main()
