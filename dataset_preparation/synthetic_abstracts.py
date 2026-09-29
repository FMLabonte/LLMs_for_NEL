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
import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pubtator_parser import parse_pubtator

GENERATION_KEY = "generation_{index}"
VIRTUAL_PMID = "{pmid}#g{index}"

# Markdown wrapper that Qwen3-8B (and similar LLMs) add to generated abstracts but
# that never appears in real PubMed text: a leading ``**Abstract:**`` heading,
# ``**bold**`` / ``*italic*`` emphasis, and multi-newline block structure. Left in
# place these are a deterministic domain cue a classifier can shortcut on
# (see finetuning/07_ABSTRACT_SWAP.md). ``clean_markdown_text`` removes them.
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_ITALIC = re.compile(r"\*(.+?)\*", re.DOTALL)
_MD_LEAD_ABSTRACT = re.compile(r"^\s*abstract\s*:?\s*", re.IGNORECASE)
_WS_RUN = re.compile(r"\s+")


def clean_markdown_text(text: str) -> str:
    """De-Markdown a single synthetic abstract to match plain-text real abstracts.

    Keeps the visible words; removes only the syntax and the structural
    ``Abstract`` heading, then collapses whitespace to a single line. Non-string
    input (e.g. NaN) is returned unchanged so it is safe to map over a column.
    """
    if not isinstance(text, str):
        return text
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = text.replace("*", "")
    text = _MD_LEAD_ABSTRACT.sub("", text)
    text = _WS_RUN.sub(" ", text).strip()
    return text


def clean_markdown_columns(
    df: pd.DataFrame, columns: Sequence[str] = ("abstract",)
) -> pd.DataFrame:
    """Return a copy of ``df`` with :func:`clean_markdown_text` applied per column.

    General runtime cleanup for any training/eval frame that carries synthetic
    abstract text. Columns absent from ``df`` are skipped. The source data files
    are never modified; this operates on the in-memory frame only.
    """
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(clean_markdown_text)
    return out


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
    allowed: set[tuple[str, int]] | None = None,
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
        allowed: Optional whitelist of ``(paper_id, generation_index)`` tuples
            (``paper_id`` as str, index as int). When given, only generations present
            in this set are emitted; ``generations`` still bounds which indices are
            considered. Used to feed a QC-filtered selection of abstracts.

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
            if allowed is not None and (pmid, index) not in allowed:
                continue
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
