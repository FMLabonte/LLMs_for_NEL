"""Side finding from the Task 4 filtering run: some entities reach the generator
as bare numeric identifiers instead of names.

Fred's generation prompt lists entities under "use these exact names". For a
subset of them the name slot holds an NCBI Gene ID (54624, 840, 83810, ...), so
the generator was asked to write an abstract using "54624" as an entity name, and
in about half those cases it did exactly that:

    "CTR9 binds to Rtf1 and an unknown factor designated 54624"

This script measures how widespread that is and, more usefully, whether the QC
model rejects those claims on its own. If it does, the filter is catching a
generation defect nobody was looking for, which is worth a line in the report.

Run after filter_synthetic.py:
    python generator_bug_check.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_synthetic import build_rows

OUT = Path(__file__).resolve().parent
CUTOFF = 0.5


def main():
    scores = pd.read_csv(OUT / "synthetic_relation_scores.csv", dtype={"paper_id": str})
    rows = build_rows()[["model", "split", "paper_id", "generation", "rel_idx", "abstract"]]
    df = scores.merge(rows, on=["model", "split", "paper_id", "generation", "rel_idx"], how="left")

    isnum = lambda s: str(s).strip().isdigit()
    df["numeric_entity"] = df.entity_a.map(isnum) | df.entity_b.map(isnum)
    idname = df.apply(lambda r: r.entity_a if isnum(r.entity_a) else r.entity_b, axis=1)
    df["id_in_text"] = [nm in ab if isinstance(ab, str) else False
                        for nm, ab in zip(idname, df.abstract)]
    df["supported"] = df.prob_supported >= CUTOFF

    total, affected = len(df), int(df.numeric_entity.sum())
    lines = [
        "# Side finding: raw gene IDs used as entity names in the generation prompts",
        "",
        f"Measured over all {total:,} relation claims scored on 2026-08-24.",
        "",
        f"- **{affected:,} claims ({affected/total*100:.1f}%)** name at least one entity by a bare",
        f"  numeric identifier rather than a text name.",
        f"- **{int(df[df.numeric_entity].id_in_text.sum()):,}** of those "
        f"({df[df.numeric_entity].id_in_text.mean()*100:.1f}%) have the raw identifier written",
        "  literally into the generated abstract.",
        f"- Papers affected: **{df[df.numeric_entity].paper_id.nunique()}** of {df.paper_id.nunique()}.",
        "",
        "## Does the QC model reject them by itself?",
        "",
        "Share of claims the QC model calls supported, at cut-off 0.5:",
        "",
        "| claim group | claims | called supported | mean probability |",
        "|---|---|---|---|",
    ]
    for label, mask in [
        ("normal entity names", ~df.numeric_entity),
        ("numeric id as a name", df.numeric_entity),
        ("  of those, id written into the abstract", df.numeric_entity & df.id_in_text),
        ("  of those, id not in the abstract", df.numeric_entity & ~df.id_in_text),
    ]:
        sub = df[mask]
        lines.append(f"| {label} | {len(sub):,} | {sub.supported.mean()*100:.1f}% | "
                     f"{sub.prob_supported.mean():.3f} |")

    normal = df[~df.numeric_entity].supported.mean()
    broken = df[df.numeric_entity].supported.mean()
    lines += [
        "",
        f"The gap is **{(normal - broken)*100:+.1f} points** of acceptance rate between normal "
        "claims and claims whose entity arrived as a bare identifier.",
        "",
        "## How to read that, in both directions",
        "",
        "The flattering reading is that the filter removes these almost entirely without "
        "having been designed to, which is evidence it keys on something real rather than "
        "rejecting at random.",
        "",
        "The unflattering reading matters more for the report. Part of what the filter is "
        "catching here is a **data defect, not a failure of the generator to express a "
        "relation**. An entity called `54624` has no readable mention to support, so the "
        "claim is unsupported for a trivial reason. Those claims should not be counted as "
        "evidence that the QC model detects subtle unsupported relations, and if the "
        "identifier bug is fixed upstream the filter's measured value will drop a little.",
        "",
        "Either way Fred should hear it: the entity-name lookup falls back to the "
        "identifier for some entities, and the generator faithfully writes the number into "
        "the abstract. It affects the unfiltered synthetic training data that BERT 1 has "
        "already been trained on.",
    ]
    (OUT / "GENERATOR_BUG.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
