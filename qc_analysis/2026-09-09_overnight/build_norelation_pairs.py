"""Task 3, step 1: build the implicit NoRelation pairs for every synthetic abstract.

Fred's position at meeting 9: the generation prompt lists a set of entities and a set of
relations, and states that any entity not in a listed relation has no relation to
anything else. So every entity pair in the prompt that is NOT in the RELATIONS block is
an implicit NoRelation pair, and the QC model has to be checked on those too. That is
the only way a hallucinated relation gets caught.

The universe of entities for a paper is

    U = (names in the ENTITIES block) union (names appearing in the RELATIONS block)

because the two blocks do not agree: the RELATIONS block sometimes uses a different
surface name for the same BioRED concept than the ENTITIES block does (for example
"SCN5A" in RELATIONS and "Na(v)1.5" in ENTITIES, both concept 6331). Names are therefore
resolved to BioRED concept identifiers through the original PubTator annotations of the
same paper, and the universe is deduplicated by identifier. Without that step the pair
list would contain bogus "NoRelation" pairs between two names for one concept.

Output: one row per (model, split, paper, generation, unordered entity pair), with the
abstract text, ready for scoring.

Run:
    python build_norelation_pairs.py            # writes norelation_pairs.parquet
    python build_norelation_pairs.py --report   # counts only, writes nothing
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / "LLMs_for_NEL"
SYN_DIR = REPO / "Data" / "Synthetic abstracts"
BIORED = REPO / "Data" / "BioRED"

sys.path.insert(0, str(REPO))
from pubtator_parser import parse_pubtator  # noqa: E402

ENT_RE = re.compile(r"^-\s*(.+?)\s*\(([^()]+)\)\s*$")
REL_RE = re.compile(r"^-\s*(.+?)\s*\(([^()]+)\)\s*--\[(\w+)\]-->\s*(.+?)\s*\(([^()]+)\)\s*$")
ENT_HEAD = "ENTITIES (use these exact names):"
REL_HEAD = "RELATIONS (these are the ONLY relations to include):"

SPLIT_FILES = {"train": "Train.PubTator", "dev": "Dev.PubTator", "test": "Test.PubTator"}


def clean_abstract(text: str) -> str:
    """Identical to qc_filtering/filter_synthetic.py, so both score the same string."""
    text = re.sub(r"^\s*\**\s*Abstract:?\s*\**\s*", "", text.strip(), flags=re.I)
    return re.sub(r"\*\*", "", text).strip()


def first_block(prompt: str, header: str) -> list[str]:
    """Lines of the FIRST block under `header`.

    Only the first matters: the prompt repeats both headers inside its few-shot
    EXAMPLE FORMAT section, and those belong to other papers.
    """
    start = prompt.find(header)
    if start == -1:
        return []
    body = prompt[start:].split("\n", 1)[1]
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            break
        out.append(line)
    return out


def load_annotation_maps() -> dict[str, dict[str, tuple[str, str]]]:
    """pmid -> {lowercased mention text: (concept_id, entity_type)} for all of BioRED."""
    maps: dict[str, dict[str, tuple[str, str]]] = {}
    for fname in SPLIT_FILES.values():
        _, anns, _ = parse_pubtator(BIORED / fname)
        for pmid, grp in anns.groupby("pmid"):
            m = maps.setdefault(str(pmid), {})
            for _, row in grp.iterrows():
                m.setdefault(str(row["mention"]).strip().lower(),
                             (str(row["mesh_id"]), str(row["entity_type"])))
    return maps


def build(report_only: bool = False) -> pd.DataFrame:
    ann_maps = load_annotation_maps()
    print(f"BioRED annotation maps for {len(ann_maps)} papers")

    rows = []
    diag = dict(papers=0, unresolved_names=0, total_names=0,
                merged_by_id=0, papers_with_merge=0)

    for path in sorted(SYN_DIR.glob("results_qwen3_*.json")):
        model, split = re.match(r"results_(qwen3_\d+b)_(\w+)\.json", path.name).groups()
        papers = json.load(open(path))
        for p in papers:
            pid = str(p["paper_id"])
            amap = ann_maps.get(pid, {})
            ent_lines = first_block(p["prompt"], ENT_HEAD)
            rel_lines = first_block(p["prompt"], REL_HEAD)

            block_ents = [ENT_RE.match(l).groups() for l in ent_lines if ENT_RE.match(l)]
            rels = [REL_RE.match(l).groups() for l in rel_lines if REL_RE.match(l)]
            if len(rels) != p["num_relations"]:
                print(f"  WARNING {path.name} {pid}: parsed {len(rels)} relations, "
                      f"num_relations={p['num_relations']}")

            def key(name: str, etype: str) -> str:
                """Concept id if BioRED knows the surface form, else the name itself."""
                hit = amap.get(name.strip().lower())
                if hit is not None:
                    return hit[0]
                diag["unresolved_names"] += 1
                return f"NAME::{name.strip().lower()}"

            # Universe, deduplicated by concept id. Prefer the ENTITIES-block surface
            # name: constraint 1 of the prompt tells the generator to write exactly
            # those, so that is the string most likely to be in the abstract.
            universe: dict[str, tuple[str, str]] = {}
            for name, etype in block_ents:
                diag["total_names"] += 1
                universe.setdefault(key(name, etype), (name, etype))
            before_rel = len(universe)
            seen_rel_names = 0
            for a, ta, _r, b, tb in rels:
                for name, etype in ((a, ta), (b, tb)):
                    diag["total_names"] += 1
                    seen_rel_names += 1
                    universe.setdefault(key(name, etype), (name, etype))
            n_distinct_names = len({n for n, _ in block_ents} |
                                   {a for a, _, _, _, _ in rels} |
                                   {b for _, _, _, b, _ in rels})
            if n_distinct_names > len(universe):
                diag["merged_by_id"] += n_distinct_names - len(universe)
                diag["papers_with_merge"] += 1
            del before_rel, seen_rel_names

            # Pairs that the prompt states a relation for, keyed by concept id.
            stated = set()
            for a, ta, _r, b, tb in rels:
                stated.add(frozenset((key(a, ta), key(b, tb))))

            ids = sorted(universe)
            cand = [(x, y) for x, y in combinations(ids, 2)
                    if frozenset((x, y)) not in stated]
            diag["papers"] += 1
            if not cand:
                continue

            for gen_name, abstract in p["synthetic_abstracts"].items():
                text = clean_abstract(abstract)
                nwords = len(text.split())
                for x, y in cand:
                    na, ta = universe[x]
                    nb, tb = universe[y]
                    rows.append({
                        "model": model, "split": split, "paper_id": pid,
                        "generation": gen_name,
                        "num_relations": p["num_relations"],
                        "n_universe": len(ids),
                        "n_norel_pairs": len(cand),
                        "abstract_words": nwords,
                        "entity_a_id": x, "entity_a": na, "type_a": ta,
                        "relation_type": "NoRelation",
                        "entity_b_id": y, "entity_b": nb, "type_b": tb,
                        "abstract": text,
                    })

    df = pd.DataFrame(rows)
    print(f"\npapers processed: {diag['papers']}")
    print(f"surface names seen: {diag['total_names']:,}, "
          f"not resolvable to a BioRED concept id: {diag['unresolved_names']:,} "
          f"({diag['unresolved_names']/max(diag['total_names'],1):.1%})")
    print(f"names merged because two names share one concept id: {diag['merged_by_id']:,} "
          f"across {diag['papers_with_merge']} paper-files")
    print(f"implicit NoRelation claims to score: {len(df):,}")
    if len(df):
        per = df.drop_duplicates(["model", "split", "paper_id", "generation"])
        print(f"synthetic abstracts covered: {len(per):,}")
        print(f"NoRelation pairs per abstract: median {per.n_norel_pairs.median():.0f}, "
              f"mean {per.n_norel_pairs.mean():.1f}, max {per.n_norel_pairs.max()}")
    if not report_only and len(df):
        out = HERE / "norelation_pairs.parquet"
        df.to_parquet(out, index=False)
        print(f"wrote {out}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.parse_args()
    build(report_only=ap.parse_args().report)
