"""Adapter: bring LLM-generated synthetic abstracts into the BioRED PubTator shape.

The synthetic datasets (``Data/Synthetic abstracts/results_qwen3_*_{train,test}.json``)
contain, per paper, the original BioRED ``paper_id`` and one or more freshly
generated abstract texts (``generation_1``, ``generation_2``, ...). They carry no
character-level entity annotations, so they cannot be parsed like a PubTator file.

Everything the relation classifier needs beyond the raw text - entity mentions,
entity types and the gold relations - already exists in the original BioRED
annotations, and the generation prompts were derived from exactly those. This
module therefore reuses the BioRED metadata verbatim and only swaps the abstract
*text*: for every paper and every requested generation it emits one virtual
document (``pmid`` suffixed with ``#g<n>``) whose ``meta`` row carries the
synthetic abstract while its ``anns`` and ``rels`` rows are copied unchanged from
the matching real paper.

The output is a ``(meta, anns, rels)`` triple with the exact same columns as
:func:`pubtator_parser.parse_pubtator`, so it drops straight into
:func:`dataset_preparation.prepare_pure_biored.build_pure_biored_samples` (and the
notebook's ``build_samples``) without any downstream branching. The only variable
that changes relative to a real-BioRED run is the Context text of each sample.

Note on negatives: ``build_pure_biored_samples`` selects distance-matched
``NoRelation`` pairs from the annotation offsets, which are the original (real)
offsets shared by all generations of a paper. Each generation therefore receives
the same gold + negative entity pairs, differing only in the abstract text - clean
context augmentation, not a change of the pair set.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pubtator_parser import parse_pubtator

GENERATION_KEY = "generation_{index}"
VIRTUAL_PMID = "{pmid}#g{index}"


def load_synthetic_abstracts(json_path: str | Path) -> dict[str, dict[str, str]]:
    """Load a synthetic-abstract JSON file as ``paper_id -> {generation_key: text}``.

    Args:
        json_path: Path to a ``results_qwen3_*_{train,test}.json`` file.

    Returns:
        Mapping from ``paper_id`` (str) to its ``synthetic_abstracts`` dict.
    """
    records = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return {str(record["paper_id"]): record["synthetic_abstracts"] for record in records}


def build_synthetic_parsed(
    json_path: str | Path,
    pubtator_file: str | Path,
    generations: Sequence[int] = (1, 2, 3),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a BioRED-shaped ``(meta, anns, rels)`` triple with synthetic abstracts.

    The BioRED entity/relation metadata of ``pubtator_file`` is reused unchanged;
    only the abstract text is replaced by the synthetic generations. Each requested
    generation becomes its own virtual document so all of them can be trained on at
    once (context augmentation).

    Args:
        json_path: Synthetic-abstract JSON file providing the generated texts.
        pubtator_file: BioRED PubTator file that supplies entity/relation metadata.
            Must be the split the synthetic papers were drawn from (Train or Test).
        generations: 1-based generation indices to emit per paper. Papers missing a
            requested generation contribute only the generations they have.

    Returns:
        ``(meta, anns, rels)`` DataFrames with the same columns as
        :func:`pubtator_parser.parse_pubtator`, keyed by virtual pmids of the form
        ``"<pmid>#g<index>"``.
    """
    meta, anns, rels = parse_pubtator(pubtator_file)
    synthetic = load_synthetic_abstracts(json_path)

    title_by_pmid: dict[str, str] = dict(zip(meta["pmid"].astype(str), meta["title"]))
    anns_by_pmid = {pmid: group for pmid, group in anns.groupby(anns["pmid"].astype(str), sort=False)}
    rels_by_pmid = {pmid: group for pmid, group in rels.groupby(rels["pmid"].astype(str), sort=False)}

    common_pmids: list[str] = [pmid for pmid in meta["pmid"].astype(str) if pmid in synthetic]

    meta_rows: list[dict[str, str]] = []
    anns_parts: list[pd.DataFrame] = []
    rels_parts: list[pd.DataFrame] = []

    for pmid in common_pmids:
        abstracts = synthetic[pmid]
        for index in generations:
            key = GENERATION_KEY.format(index=index)
            text = abstracts.get(key)
            if not text:
                continue
            virtual_pmid = VIRTUAL_PMID.format(pmid=pmid, index=index)

            meta_rows.append(
                {"pmid": virtual_pmid, "title": title_by_pmid.get(pmid, ""), "abstract": text}
            )
            if pmid in anns_by_pmid:
                part = anns_by_pmid[pmid].copy()
                part["pmid"] = virtual_pmid
                anns_parts.append(part)
            if pmid in rels_by_pmid:
                part = rels_by_pmid[pmid].copy()
                part["pmid"] = virtual_pmid
                rels_parts.append(part)

    meta_out = pd.DataFrame(meta_rows, columns=list(meta.columns))
    anns_out = (
        pd.concat(anns_parts, ignore_index=True) if anns_parts else anns.iloc[0:0].copy()
    )
    rels_out = (
        pd.concat(rels_parts, ignore_index=True) if rels_parts else rels.iloc[0:0].copy()
    )
    return meta_out, anns_out, rels_out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python synthetic_abstracts.py <synthetic.json> <pubtator_file> [n_generations]")
        sys.exit(1)

    n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    meta_df, anns_df, rels_df = build_synthetic_parsed(sys.argv[1], sys.argv[2], tuple(range(1, n_gen + 1)))
    papers = meta_df["pmid"].str.split("#").str[0].nunique()
    print(f"virtual documents : {len(meta_df)} ({papers} papers x up to {n_gen} generations)")
    print(f"annotations       : {len(anns_df)}")
    print(f"relations         : {len(rels_df)}")
    print(meta_df.head(3)[["pmid", "abstract"]].to_string(index=False))
