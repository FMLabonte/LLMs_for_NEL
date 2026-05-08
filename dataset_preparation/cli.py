"""
CLI entry point for the Phase 1 BioRED data pipeline.

Run `python cli.py --help` to see all commands.

Typical workflow:
    python cli.py inspect             # check raw BioRED stats
    python cli.py build               # generate perturbed CSVs in ../output/
    python cli.py stats               # per-perturbation breakdown of the CSVs
    python cli.py peek -n 3           # eyeball some sample rows
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from build_dataset import build_split
from data_loader import SPLIT_FILES, load_biored_split, summarize
from perturbations import BIORED_RELATION_TYPES, SYMMETRIC_RELATIONS

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "output"

app = typer.Typer(
    help="BioRED Phase 1 data pipeline: load, perturb, inspect.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# ---------------------------------------------------------------------------
# Shared types / helpers
# ---------------------------------------------------------------------------

class Split(str, Enum):
    train = "train"
    dev   = "dev"
    test  = "test"
    all   = "all"


class Perturbation(str, Enum):
    gold             = "gold"
    label_flip       = "label_flip"
    direction_swap   = "direction_swap"
    fp_co_related    = "fp_co_related"
    fp_co_standalone = "fp_co_standalone"
    fp_external      = "fp_external"
    false_negative   = "false_negative"


def _resolve_splits(split: Split) -> list[str]:
    return list(SPLIT_FILES) if split is Split.all else [split.value]


def _csv_path(split: str, output_dir: Path) -> Path:
    return output_dir / f"biored_{split}_samples.csv"


def _load_built_csv(split: str, output_dir: Path) -> pd.DataFrame:
    path = _csv_path(split, output_dir)
    if not path.exists():
        raise typer.BadParameter(
            f"No built CSV at {path}. Run `python cli.py build {split}` first."
        )
    return pd.read_csv(path, dtype=str).assign(label=lambda d: d["label"].astype(int))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def inspect(
    split: Split = typer.Argument(Split.all, help="Which split to inspect."),
) -> None:
    """Show raw BioRED statistics: papers, entities, relations, types."""
    splits = _resolve_splits(split)

    table = Table(title="BioRED raw stats", show_lines=False)
    table.add_column("split", style="cyan", no_wrap=True)
    table.add_column("papers", justify="right")
    table.add_column("with rels", justify="right")
    table.add_column("entities", justify="right")
    table.add_column("relations", justify="right")

    summaries = {}
    for s in splits:
        meta, anns, rels = load_biored_split(s)
        info = summarize(meta, anns, rels)
        summaries[s] = info
        table.add_row(
            s,
            str(info["papers"]),
            str(info["papers_with_rels"]),
            str(info["entities"]),
            str(info["relations"]),
        )

    console.print(table)

    if "train" in summaries:
        info = summaries["train"]
        console.print(f"\n[bold]Entity types (from train):[/bold]   {info['entity_types']}")
        console.print(f"[bold]Relation types (from train):[/bold] {info['relation_types']}")
        console.print(
            f"[dim]Symmetric relations (direction-swap skipped):[/dim] "
            f"{sorted(SYMMETRIC_RELATIONS)}"
        )


@app.command()
def build(
    split: Split = typer.Argument(Split.all, help="Which split to build."),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for perturbations."),
    no_type_restrict: bool = typer.Option(
        False,
        "--no-type-restrict",
        help="Disable type-restricted false-positive sampling "
             "(allows nonsense pairs like Species/CellLine).",
    ),
    out: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--out",
        "-o",
        help="Output directory for the CSVs.",
    ),
) -> None:
    """Build the perturbed (gold + 4-perturbation) CSV(s)."""
    out.mkdir(parents=True, exist_ok=True)
    splits = _resolve_splits(split)

    for s in splits:
        console.print(f"\n[bold cyan]Building {s}[/bold cyan]")
        df = build_split(
            s,
            seed=seed,
            type_restricted_false_positives=not no_type_restrict,
        )
        path = _csv_path(s, out)
        df.to_csv(path, index=False)
        _print_stats_table(df, s)
        console.print(f"[green]Wrote[/green] {path} ([dim]{len(df)} rows[/dim])")


@app.command()
def stats(
    split: Split = typer.Argument(Split.all, help="Which split's CSV to summarize."),
    out: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--out", "-o", help="Where the CSVs live.",
    ),
) -> None:
    """Per-perturbation distribution from already-built CSV(s)."""
    for s in _resolve_splits(split):
        df = _load_built_csv(s, out)
        _print_stats_table(df, s)


@app.command()
def peek(
    split: Split = typer.Argument(Split.train, help="Which split to peek into."),
    n: int = typer.Option(5, "--n", "-n", help="How many rows to show."),
    perturbation: Optional[Perturbation] = typer.Option(
        None, "--perturbation", "-p", help="Filter by perturbation type.",
    ),
    label: Optional[int] = typer.Option(
        None, "--label", "-l", help="Filter by label (0 = wrong, 1 = correct).",
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed for sampling."),
    out: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--out", "-o", help="Where the CSVs live.",
    ),
) -> None:
    """Print a few random sample rows from a built CSV (for eyeballing)."""
    df = _load_built_csv(split.value, out)

    filtered = df
    if perturbation is not None:
        filtered = filtered[filtered["perturbation"] == perturbation.value]
    if label is not None:
        filtered = filtered[filtered["label"] == label]

    if len(filtered) == 0:
        console.print("[yellow]No rows match those filters.[/yellow]")
        raise typer.Exit(code=1)

    sample = filtered.sample(min(n, len(filtered)), random_state=seed)

    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        console.rule(f"[bold]{i}/{n}[/bold] [cyan]{row['perturbation']}[/cyan] (label={row['label']})")
        console.print(f"[dim]PMID:[/dim] {row['pmid']}")
        abstract = row["abstract"]
        snippet = abstract[:240] + ("..." if len(abstract) > 240 else "")
        console.print(f"[dim]Abstract:[/dim] {snippet}")
        console.print(
            f"[dim]Triple:[/dim] "
            f"([magenta]{row['entity_a_text']}[/magenta] "
            f"[{row['entity_a_type']}], "
            f"[bold yellow]{row['relation_type']}[/bold yellow], "
            f"[magenta]{row['entity_b_text']}[/magenta] "
            f"[{row['entity_b_type']}])"
        )


@app.command()
def validate() -> None:
    """Sanity checks on raw BioRED before training: known relation types, etc."""
    ok = True
    for s in SPLIT_FILES:
        _, _, rels = load_biored_split(s)
        observed = set(rels["relation_type"].unique())
        unknown = observed - set(BIORED_RELATION_TYPES)
        if unknown:
            console.print(f"[red][{s}] unknown relation types: {sorted(unknown)}[/red]")
            ok = False
        else:
            console.print(f"[green][{s}] all relation types known.[/green]")

    if not ok:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Internal: shared rendering
# ---------------------------------------------------------------------------

def _print_stats_table(df: pd.DataFrame, split: str) -> None:
    table = Table(title=f"{split}: perturbation breakdown", show_lines=False)
    table.add_column("perturbation", style="cyan")
    table.add_column("count", justify="right")
    table.add_column("label", justify="right")

    counts = df.groupby("perturbation")[["label"]].agg(["count", "mean"])
    counts.columns = ["count", "label_mean"]
    counts = counts.sort_index()

    for ptype, row in counts.iterrows():
        label_str = "1 (correct)" if row["label_mean"] == 1.0 else "0 (incorrect)"
        table.add_row(ptype, str(int(row["count"])), label_str)

    table.add_section()
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{len(df)}[/bold]", "")

    n_correct = int((df["label"] == 1).sum())
    n_wrong = int((df["label"] == 0).sum())
    table.add_row("correct", str(n_correct), "")
    table.add_row("incorrect", str(n_wrong), "")
    console.print(table)


if __name__ == "__main__":
    app()
