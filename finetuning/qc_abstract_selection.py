"""Load and filter the synthetic-abstract QC decisions.

The QC file (``qc_filtering/abstract_decisions.csv``) holds one row per
generated abstract with a quality evaluation.
This module loads it and selects the abstracts that pass QC for a given
model and split, optionally deduplicating to one generation per abstract.
"""

import json
from pathlib import Path

import pandas as pd

DEFAULT_CSV: Path = Path(__file__).resolve().parent.parent / "qc_filtering" / "abstract_decisions.csv"
# NoRelation-aware QC decisions (built under the dynamic rule): same one-row-per-generation
# grain as DEFAULT_CSV, but the ``passed`` keep decision also judges the implicit NoRelation
# pairs the prompt leaves out. It carries no ``mean_prob``; see :func:`load_norel_decisions`.
NOREL_CSV: Path = Path(__file__).resolve().parent.parent / "qc_filtering" / "abstract_decisions_norel.csv"
# Key shared by both decision CSVs and the synthetic-abstract JSONs.
_DECISION_KEY: list[str] = ["model", "split", "paper_id", "generation"]


def load_decisions(csv_path: Path = DEFAULT_CSV, expected_rule: str | None = None) -> pd.DataFrame:
    """Load the QC decisions CSV into a dataframe.

    :param csv_path: path to the decisions CSV.
    :param expected_rule: if given, assert the CSV was produced by ``decide.py`` with
        this ``rule`` (e.g. ``"dynamic"``). The rule is stamped into the CSV since
        2026-08; passing this guards against rule drift, i.e. training on a selection
        the CSV was silently regenerated with. Raises ValueError on mismatch.
    """
    df: pd.DataFrame = pd.read_csv(csv_path)
    if expected_rule is not None:
        if "rule" not in df.columns:
            raise ValueError(
                f"{csv_path} has no 'rule' column (pre-provenance CSV); cannot confirm "
                f"it was produced with rule={expected_rule!r}. Re-run qc_filtering/decide.py."
            )
        rules: set[str] = set(df["rule"].unique())
        if rules != {expected_rule}:
            raise ValueError(f"{csv_path} was produced with rule(s) {rules}, expected {{{expected_rule!r}}}.")
    return df


def load_norel_decisions(
    norel_csv: Path = NOREL_CSV, prob_csv: Path = DEFAULT_CSV
) -> pd.DataFrame:
    """Load the NoRelation-aware QC decisions, attaching ``mean_prob`` from the stated-only CSV.

    The NoRelation-aware CSV marks each generation with ``passed`` (the dynamic-rule keep
    decision that also judges the implicit NoRelation pairs), but carries no QC probability,
    so it cannot pick a best generation per paper on its own. The per-generation ``mean_prob``
    used for that deduplication lives only in the stated-only decisions CSV, keyed identically
    by ``(model, split, paper_id, generation)``. Joining them lets the NoRelation-aware arm
    deduplicate with the exact same highest-``mean_prob`` rule as the original QC-dedup arm;
    the only difference between the two arms is which CSV supplies the ``passed`` column.

    :param norel_csv: path to the NoRelation-aware decisions CSV.
    :param prob_csv: path to the stated-only decisions CSV supplying ``mean_prob``.
    :return: the NoRelation-aware dataframe with a ``mean_prob`` column added.
    """
    df: pd.DataFrame = pd.read_csv(norel_csv)
    if "passed" not in df.columns:
        raise ValueError(f"{norel_csv} has no 'passed' column; wrong CSV?")
    probs: pd.DataFrame = pd.read_csv(prob_csv)[_DECISION_KEY + ["mean_prob"]]
    merged: pd.DataFrame = df.merge(probs, on=_DECISION_KEY, how="left")
    missing: pd.DataFrame = merged.loc[merged["passed"] & merged["mean_prob"].isna()]
    if len(missing):
        raise ValueError(
            f"{len(missing)} passing NoRelation-aware rows have no mean_prob in {prob_csv} "
            f"(the two CSVs disagree on the (model, split, paper_id, generation) key); "
            f"e.g. {missing[_DECISION_KEY].head(3).to_dict('records')}"
        )
    return merged


def verify_allowed_against_synthetic(
    allowed: set[tuple[str, int]],
    synth_json_path: Path,
    model: str,
    split: str,
) -> None:
    """Guard the QC-selection -> text-JSON handoff against silent mismatches.

    Two distinct footguns, because :func:`dataset_preparation.synthetic_abstracts.
    build_synthetic_parsed` *silently* skips any ``allowed`` key the JSON lacks:

    1. **Model mixup.** The 4B and 8B generators share the same ``(paper_id,
       generation)`` key space (same BioRED papers, generations 1-3); only the text
       differs. So a selection made with ``QC_MODEL="qwen3_8b"`` fed to the *4B* JSON
       would key-match perfectly and train on the wrong text with no error. Model
       identity survives only in the JSON *filename*, so this asserts the path is
       ``results_{model}_{split}.json``.
    2. **Stale/partial JSON.** Any selected key genuinely absent from the JSON (a JSON
       missing papers the CSV scored) would be dropped silently; this reports it.

    Raises ValueError on either. Call it right before ``build_synthetic_parsed``.

    :param allowed: selection from :func:`qc_allowed_generations`.
    :param synth_json_path: the synthetic-abstract JSON that will supply the text.
    :param model: the ``QC_MODEL`` the selection was made with, e.g. ``"qwen3_8b"``.
    :param split: the split the selection is for, e.g. ``"train"``.
    """
    from dataset_preparation.synthetic_abstracts import GENERATION_KEY

    path = Path(synth_json_path)
    expected_name = f"results_{model}_{split}.json"
    if path.name != expected_name:
        raise ValueError(
            f"QC selection is for model={model!r} split={split!r} (expects {expected_name}), "
            f"but the text JSON is {path.name!r}; the generators share a (paper_id, generation) "
            f"key space, so this mixup would train on the wrong text silently."
        )

    prefix = GENERATION_KEY.format(index="")  # "generation_"
    text_keys: set[tuple[str, int]] = {
        (str(p["paper_id"]), int(key.split("_")[-1]))
        for p in json.loads(path.read_text())
        for key in p["synthetic_abstracts"]
        if key.startswith(prefix)
    }
    dangling: set[tuple[str, int]] = allowed - text_keys
    if dangling:
        raise ValueError(
            f"{len(dangling)} QC-selected abstracts are absent from {path.name} "
            f"(stale or partial JSON?), e.g. {sorted(dangling)[:3]}"
        )


def select_abstracts(
    df: pd.DataFrame,
    model: str,
    split: str,
    deduplicate: bool = True,
) -> set[tuple[int, str]]:
    """Select the ``(paper_id, generation)`` tuples passing QC.

    Filters for ``passed == True`` and the given ``model`` and ``split``.
    When ``deduplicate`` is set, only the highest-``mean_prob`` generation is
    kept per ``paper_id``.

    :param df: the loaded QC decisions dataframe.
    :param model: model name to keep, e.g. ``"qwen3_8b"``.
    :param split: split name to keep, e.g. ``"train"``.
    :param deduplicate: keep only one generation per paper if True.
    :return: set of ``(paper_id, generation)`` tuples.
    """
    scope = (df["model"] == model) & (df["split"] == split)
    if not scope.any():
        raise ValueError(
            f"no rows for model={model!r}, split={split!r}; "
            f"models={sorted(df['model'].unique())}, splits={sorted(df['split'].unique())}"
        )
    selected: pd.DataFrame = df.loc[scope & df["passed"]]

    if deduplicate:
        # Keep the generation with the highest mean_prob per paper_id.
        idx = selected.groupby("paper_id")["mean_prob"].idxmax()
        selected = selected.loc[idx]

    return set(zip(selected["paper_id"], selected["generation"]))


def qc_allowed_generations(
    df: pd.DataFrame,
    model: str,
    split: str,
    deduplicate: bool = True,
) -> set[tuple[str, int]]:
    """Return QC-passed ``(paper_id, generation_index)`` tuples for the adapter.

    Same selection as :func:`select_abstracts`, but shaped for
    :func:`dataset_preparation.synthetic_abstracts.build_synthetic_parsed`:
    ``paper_id`` as ``str`` and the generation as its 1-based ``int`` index (parsed
    from the ``"generation_<n>"`` label).

    :param df: the loaded QC decisions dataframe.
    :param model: model name to keep, e.g. ``"qwen3_8b"``.
    :param split: split name to keep, e.g. ``"train"``.
    :param deduplicate: keep only the highest-``mean_prob`` generation per paper.
    :return: set of ``(str(paper_id), int(generation_index))`` tuples.
    """
    pairs = select_abstracts(df, model=model, split=split, deduplicate=deduplicate)
    return {(str(paper_id), int(str(generation).split("_")[-1])) for paper_id, generation in pairs}


if __name__ == "__main__":
    decisions: pd.DataFrame = load_decisions()

    for model_name in ("qwen3_8b", "qwen3_4b"):
        tuples = select_abstracts(decisions, model=model_name, split="train", deduplicate=True)
        print(f"{model_name}: {len(tuples)} abstracts (train, deduplicated)")
