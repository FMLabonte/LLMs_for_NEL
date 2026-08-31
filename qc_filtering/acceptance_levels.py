"""Meeting 8 task 4: the QC-settings experiment, several levels of acceptance.

Requested at the 2026-08-27 meeting: "we still need the experiment using the QC model with
different settings, different levels of acceptance, several to train the base
model on and see if this leads to any improvements. If you don't have this
experiment, the report is missing its main question."

So this produces a LADDER of training sets, from strict to permissive, each one
a subset of the synthetic data, plus a same-size random control for each so
"the QC model picked well" can be told apart from "the set got smaller".

The knob is the tolerated error rate per abstract, not a step function on the
relation count. An abstract is kept when

    (claims the QC model rejects) / (claims made) <= tau

tau = 0 is the strict rule (any detected error rejects the abstract) and tau = 1
keeps everything, i.e. the unfiltered baseline Chris already trained on. Using a
rate rather than a raw count is deliberate: the normalised error-rate plot shows
the generator's error rate rising with relation count, so a fixed budget of
allowed errors punishes long abstracts twice.

The old strict/dynamic rules from decide.py are carried along as named levels so
the 2026-08-24 numbers stay comparable.

Outputs, into acceptance_levels/:
  acceptance_levels.csv    one row per synthetic abstract, kept flag per level
  LEVELS.md                the ladder, sizes and what Chris trains on
  materialize.py runs separately and writes the actual JSONs.

Run:
    python acceptance_levels.py
"""
import json
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
SCORES = OUT_DIR / "synthetic_relation_scores.csv"
LEVEL_DIR = OUT_DIR / "acceptance_levels"

FOUR_CLASS = {"Association", "Positive_Correlation", "Negative_Correlation", "NoRelation"}
CUTOFF = 0.5

# The ladder handed to Chris. Ordered strict to permissive so the kept counts
# are monotone and the levels nest.
LEVELS = [
    ("L0_strict",    "rate", 0.0,  "no detected error at all, the 2026-08-24 shipped rule"),
    ("L1_rate10",    "rate", 0.10, "up to 10% of the claims may be rejected"),
    ("L2_rate20",    "rate", 0.20, "up to 20%"),
    ("L3_rate33",    "rate", 0.33, "up to a third"),
    ("L4_rate50",    "rate", 0.50, "up to half"),
    ("L5_unfiltered", "rate", 1.00, "everything, the baseline Chris already has"),
    ("D_dynamic",    "dynamic", None, "the step function from decide.py, kept for continuity"),
]


def allowed_errors(n_relations: int) -> int:
    """The 2026-07-22 step function, reproduced so the old level still matches."""
    if n_relations <= 8:
        return 0
    if n_relations <= 12:
        return 1
    return 2


def per_abstract(df: pd.DataFrame) -> pd.DataFrame:
    """One row per synthetic abstract, with the error count and rate.

    Rare-type claims are dropped rather than scored. The QC model was never
    trained on the five rare BioRED types, so a rejection there measures a
    coverage gap rather than the generator. This matches the rares=exclude choice
    shipped on 2026-08-24.
    """
    d = df.copy()
    d["is_rare"] = ~d.relation_type.isin(FOUR_CLASS)
    n_rare = (d.groupby(["model", "split", "paper_id", "generation"])
                .is_rare.sum().rename("n_rare"))
    d = d[~d.is_rare]
    d["failed"] = d.prob_supported < CUTOFF

    key = ["model", "split", "paper_id", "generation"]
    out = (d.groupby(key)
             .agg(n_relations=("rel_idx", "count"),
                  n_failed=("failed", "sum"),
                  min_prob=("prob_supported", "min"),
                  mean_prob=("prob_supported", "mean"),
                  abstract_words=("abstract_words", "first"))
             .reset_index()
             .merge(n_rare.reset_index(), on=key, how="left"))
    out["error_rate"] = out.n_failed / out.n_relations
    return out


def apply_levels(abs_df: pd.DataFrame) -> pd.DataFrame:
    out = abs_df.copy()
    for name, kind, tau, _ in LEVELS:
        if kind == "rate":
            out[name] = out.error_rate <= tau + 1e-9
        else:
            out[name] = out.n_failed <= out.n_relations.map(allowed_errors)
    return out


def main():
    LEVEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(SCORES, dtype={"paper_id": str})
    abs_df = apply_levels(per_abstract(df))
    print(f"{len(df):,} claims -> {len(abs_df):,} synthetic abstracts, cutoff {CUTOFF}")

    abs_df.to_csv(LEVEL_DIR / "acceptance_levels.csv", index=False)

    names = [n for n, _, _, _ in LEVELS]
    rows = []
    for name, kind, tau, note in LEVELS:
        kept = abs_df[name]
        train = abs_df[(abs_df.split == "train") & kept]
        rows.append({
            "level": name,
            "tolerated error rate": "step fn" if kind == "dynamic" else f"{tau:.0%}",
            "abstracts kept": int(kept.sum()),
            "kept %": round(float(kept.mean()) * 100, 1),
            "train abstracts": len(train),
            "papers in train": int(train.paper_id.nunique()),
            "note": note,
        })
    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))

    # Nesting check. If a permissive level ever dropped an abstract a stricter
    # one kept, the ladder would not be a ladder and the comparison would be
    # confounded by which abstracts moved rather than by how many.
    rate_names = [n for n, k, _, _ in LEVELS if k == "rate"]
    nested = all(
        bool((abs_df[a] & ~abs_df[b]).sum() == 0)
        for a, b in zip(rate_names, rate_names[1:])
    )
    print(f"\nrate levels nest correctly: {nested}")

    per_split = (abs_df.melt(id_vars=["model", "split"], value_vars=names,
                             var_name="level", value_name="kept")
                       .groupby(["level", "split"]).kept.agg(["size", "sum"])
                       .assign(pct=lambda x: (x["sum"] / x["size"] * 100).round(1))
                       .reset_index())

    doc = [
        "# QC acceptance levels, the settings experiment",
        "",
        "Meeting 8 task 4. Several filtered versions of the synthetic training data at",
        "different levels of acceptance, so the relation classifier can be trained on each",
        "to see whether any of them beats the unfiltered baseline.",
        "",
        f"Source: `synthetic_relation_scores.csv`, {len(df):,} claims scored by the run-2 QC",
        f"model over {len(abs_df):,} synthetic abstracts. Decision cut-off {CUTOFF}. Claims of the",
        "five rare BioRED types are dropped, not scored, because the QC model was never",
        "trained on them.",
        "",
        "An abstract is kept at level tau when the share of its claims that the QC model",
        "rejects is at most tau. The levels nest, so a more permissive level keeps a superset",
        "of a stricter one, and the only thing changing between two runs is how much",
        "suspected noise is tolerated.",
        "",
        "## The ladder",
        "",
        table.to_markdown(index=False),
        "",
        f"Rate levels nest correctly: **{nested}**.",
        "",
        "## Kept per split",
        "",
        per_split.pivot(index="level", columns="split", values="pct").to_markdown(),
        "",
        "Numbers are the percentage of abstracts kept.",
        "",
        "## What to train",
        "",
        "For each level, two runs:",
        "",
        "1. the filtered set,",
        "2. a same-size random control drawn from the unfiltered pool at seed 42.",
        "",
        "The control is what separates a real filtering effect from a set-size effect. If",
        "the filtered run beats its control the QC model is selecting useful data; if the",
        "two match, all that happened was the set got smaller.",
        "",
        "`L5_unfiltered` is the full set and is its own control, so it needs one run only.",
        "That run is the baseline Chris already has from 2026-08-04.",
        "",
        "## Getting the files",
        "",
        "The JSONs are not committed: the ladder is roughly 100 MB of near-duplicate data",
        "and it is fully determined by `acceptance_levels.csv`. Build any level locally:",
        "",
        "```",
        "python materialize.py --level L2_rate20          # filtered + random control",
        "python materialize.py --level all --splits train dev",
        "```",
        "",
        "Output lands in `acceptance_levels/<level>/` in the source schema, so it drops",
        "straight into `load_synthetic_abstracts` the same way the 2026-08-24 files did.",
    ]
    (LEVEL_DIR / "LEVELS.md").write_text("\n".join(doc) + "\n")

    (LEVEL_DIR / "levels.json").write_text(json.dumps(
        {"cutoff": CUTOFF, "nested": nested,
         "levels": [{"name": n, "kind": k, "tau": t, "note": d} for n, k, t, d in LEVELS],
         "table": rows}, indent=2) + "\n")

    print(f"\nwrote {LEVEL_DIR}/acceptance_levels.csv, LEVELS.md, levels.json")


if __name__ == "__main__":
    main()
