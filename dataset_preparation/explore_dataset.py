"""Data exploration for the perturbed BioRED dataset (proposal section 5.1).

Reports, for each split, the things the proposal commits to inspecting on every
build:
  - row counts per perturbation type and per relation type
  - label-1 share
  - abstract length distribution
  - entity-pair counts per abstract: annotated vs co-mentioned-but-unrelated
    (implicit NoRelation), plus the corpus-level negative count
  - byte-reproducibility for a fixed seed

Outputs a stats.json and a set of figures into the directory given as argv[1].
Run from the repo root:
    python dataset_preparation/explore_dataset.py <output_dir> [--rare-classes]
"""
from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations  # noqa: F401  (kept for clarity; count is computed directly)
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `build_dataset` / `data_loader` import cleanly

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from build_dataset import build_perturbed_dataframe  # noqa: E402
from data_loader import load_biored_split  # noqa: E402

SEED = 42
SPLITS = ["train", "dev", "test"]
PERT_ORDER = ["gold", "label_flip", "direction_swap",
              "fp_co_related", "fp_co_standalone", "fp_external", "false_negative"]


def desc(series) -> dict:
    s = series.astype(float)
    return {
        "n": int(s.size),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "mean": round(float(s.mean()), 2),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "total": int(s.sum()),
    }


def entity_pairs_per_abstract(split: str) -> pd.DataFrame:
    """For each abstract: #entities, #co-mentioned pairs, #annotated pairs,
    #implicit-no-relation pairs (co-mentioned minus annotated). Uses raw BioRED
    (all relation types), which is the pool the fp_* / false_negative families
    sample from."""
    from perturbations import _split_multi_id_annotations
    _, anns, rels = load_biored_split(split)
    anns = _split_multi_id_annotations(anns)  # match the pipeline: count individual IDs
    ents_by_pmid = anns.groupby("pmid")["mesh_id"].apply(lambda s: set(x for x in s if pd.notna(x)))
    ann_pairs_by_pmid: dict[str, set] = {}
    for pmid, grp in rels.groupby("pmid"):
        pairs = set()
        for a, b in zip(grp["id_1"], grp["id_2"]):
            if pd.isna(a) or pd.isna(b):
                continue
            pairs.add(tuple(sorted((str(a), str(b)))))
        ann_pairs_by_pmid[pmid] = pairs
    rows = []
    for pmid, ents in ents_by_pmid.items():
        n = len(ents)
        comention = n * (n - 1) // 2
        annp = {p for p in ann_pairs_by_pmid.get(pmid, set()) if p[0] in ents and p[1] in ents}
        n_ann = len(annp)
        rows.append((pmid, n, comention, n_ann, max(comention - n_ann, 0)))
    return pd.DataFrame(rows, columns=["pmid", "n_entities", "comention_pairs",
                                       "annotated_pairs", "implicit_norel_pairs"])


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else HERE
    rare = "--rare-classes" in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    stats: dict = {"config": {"seed": SEED, "keep_rare_classes": rare,
                              "default_kept_relations": ["Association",
                                                         "Positive_Correlation",
                                                         "Negative_Correlation"]}}
    dfs = {sp: build_perturbed_dataframe(sp, seed=SEED, keep_rare_classes=rare) for sp in SPLITS}

    for sp, df in dfs.items():
        df["label"] = df["label"].astype(int)
        s: dict = {}
        s["rows"] = int(len(df))
        s["label1_count"] = int((df["label"] == 1).sum())
        s["label1_share"] = round(float((df["label"] == 1).mean()), 4)
        s["per_perturbation"] = {k: int(df["perturbation"].value_counts().get(k, 0)) for k in PERT_ORDER}
        s["per_relation_type"] = {k: int(v) for k, v in df["relation_type"].value_counts().items()}
        s["abstracts_in_build"] = int(df["pmid"].nunique())

        # abstract length (whitespace tokens) over the unique abstracts in the build
        ab = df.drop_duplicates("pmid")["abstract"].fillna("")
        s["abstract_len_words"] = desc(ab.str.split().map(len))

        # entity pairs per abstract (raw BioRED)
        ep = entity_pairs_per_abstract(sp)
        s["entity_pairs_per_abstract"] = {
            "n_abstracts": int(len(ep)),
            "n_entities": desc(ep["n_entities"]),
            "annotated_pairs": desc(ep["annotated_pairs"]),
            "implicit_norel_pairs": desc(ep["implicit_norel_pairs"]),
            "negatives_vs_positives_ratio": round(
                float(ep["implicit_norel_pairs"].sum()) / max(int(ep["annotated_pairs"].sum()), 1), 1),
        }
        stats[sp] = s

        if sp == "train":
            _figures(df, ep, fig_dir)

    # reproducibility: same seed -> byte-identical CSV
    csv_a = build_perturbed_dataframe("train", seed=SEED, keep_rare_classes=rare).to_csv(index=False)
    csv_b = build_perturbed_dataframe("train", seed=SEED, keep_rare_classes=rare).to_csv(index=False)
    h_a = hashlib.sha256(csv_a.encode()).hexdigest()
    stats["reproducibility"] = {
        "train_two_builds_byte_identical": csv_a == csv_b,
        "sha256_prefix": h_a[:16],
    }

    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {out/'stats.json'} and figures to {fig_dir}")


def _bar(ax, labels, values, title, ylabel="rows"):
    ax.bar(range(len(labels)), values, color="#4C72B0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)


def _figures(df: pd.DataFrame, ep: pd.DataFrame, fig_dir: Path) -> None:
    # 1. rows per perturbation type
    pc = [int(df["perturbation"].value_counts().get(k, 0)) for k in PERT_ORDER]
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar(ax, PERT_ORDER, pc, "Train rows per perturbation type")
    fig.tight_layout(); fig.savefig(fig_dir / "01_rows_per_perturbation.png", dpi=130); plt.close(fig)

    # 2. rows per relation type
    rc = df["relation_type"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    _bar(ax, list(rc.index), [int(v) for v in rc.values], "Train rows per relation type")
    fig.tight_layout(); fig.savefig(fig_dir / "02_rows_per_relation.png", dpi=130); plt.close(fig)

    # 3. label distribution
    lc = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 4))
    _bar(ax, [f"label={i}" for i in lc.index], [int(v) for v in lc.values], "Train label distribution")
    fig.tight_layout(); fig.savefig(fig_dir / "03_label_distribution.png", dpi=130); plt.close(fig)

    # 4. abstract length histogram
    wl = df.drop_duplicates("pmid")["abstract"].fillna("").str.split().map(len)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(wl, bins=30, color="#55A868")
    ax.set_title("Abstract length (whitespace tokens), train abstracts", fontsize=10)
    ax.set_xlabel("words"); ax.set_ylabel("abstracts")
    fig.tight_layout(); fig.savefig(fig_dir / "04_abstract_length.png", dpi=130); plt.close(fig)

    # 5. annotated vs implicit-no-relation pairs per abstract
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([ep["annotated_pairs"], ep["implicit_norel_pairs"]], bins=20,
            label=["annotated", "implicit no-relation"], color=["#4C72B0", "#C44E52"])
    ax.set_title("Entity pairs per abstract (train)", fontsize=10)
    ax.set_xlabel("pairs per abstract"); ax.set_ylabel("abstracts"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "05_entity_pairs_per_abstract.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
