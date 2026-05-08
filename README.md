# LLMs_for_NEL: Synthetic Data Quality Control for Biomedical RE

A binary classifier that filters LLM-generated synthetic biomedical training data by detecting hallucinated relations. Built on BioRED gold annotations plus 4 controlled perturbations.

## Table of contents

- [The Broader Idea](#the-broader-idea)
  - [The Solution](#the-solution)
  - [Overall workflow](#overall-workflow)
  - [Research question(s)](#research-questions)
  - [Example walkthrough](#example)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Phase 1 CLI](#phase-1-cli)
  - [Commands](#commands)
  - [Examples](#examples)
  - [How to add new commands](#how-to-add-new-commands)
- [Repository structure](#repository-structure)
- [Direct parser usage (without the CLI)](#example-to-run-the-demo)

## The Broader Idea
We want to train Relation Extraction (RE) models for biomedical texts, which is a relatively hard task requiring expensive human annotators. The problem is that only a limited amount of training data is available. The idea is to use LLMs to create synthetic data by prompting them with a list of relations that should be described in an abstract, then having the model generate the abstract. However, this creates a chicken-and-egg problem: no models exist to verify that the LLM actually follows the input and provides it correctly in the synthetic abstract.
### The Solution
We need a way to verify this synthetic data. Luckily, we have around 500 annotated abstracts that we can treat as a gold standard. Using these, we can switch annotations and text, treating the annotated abstract as if it were the generation of a model. We can then introduce perturbations in the annotations to introduce errors in a controlled way—for example, flipping a label from negative to positive. We can then train a small model to identify those errors by providing the text and the expected label, turning this into a binary classification task.
Producing synthetic data is cheap, and with an effective filter, we can ensure data quality.
### overall workflow
BioRED data -> Introduce errors -> train model to detect them -> use filter on synthetic data -> does it improve perfromance ?


### Research question(s)
The main research questions of this project and its follow-up work are:
- Does filtering synthetic data improve model training?
- How large/diverse must the set of seed abstracts be to construct good filters?
- What errors are the hardest to spot?
- Which model architectures are most effective for this task ? 



#### Example 
We observe that under hypoxia a downregulation of VEGFA increases the actovation level of the SRY gene. -> From this we can take that VEGFA is negatively correlated SRY since its deactivation leads to an increase of the other. expressed as an annotation VEGFA negative_correlation SRY 
##### Step 1
Now we can treat this annotation and text pair as input and output in put being Relation: [VEGFA negative_correlation SRY] and our theoretical synthetic data creation model would have written out: text: [We observe that under hypoxia a downregulation of VEGFA increases the actovation level of the SRY gene.] <- this reversed pair is our gold sample

##### Example Gold Sample: 
**Relation:** [VEGFA negative_correlation SRY] **Text:** [We observe that under hypoxia a downregulation of VEGFA increases the actovation level of the SRY gene.] **Label:** [true]

##### Step 2
We want to see if we can train a model that would spot a missmatch between input relation pair and output Sentence. changing sentences is an controlable manner is hard, Changing annotations is not. 
VEGFA negative_correlation SRY can be changed to VEGFA positive_correlation SRY. Now the model is given this and the input text, with the quetion is this tripplet correctly represented by the text ?

##### Example introduced error:
**Relation:** [VEGFA positive_correlation SRY] **Text:** [We observe that under hypoxia a downregulation of VEGFA increases the actovation level of the SRY gene.] **Label:** [wrong]

##### Step 3
Using this we can create different errors: 
Swapping labels, Connecting enteties that arent connected, Connecting enteties with enteties that are not in the text etc. and figure out which of those the models can spot and how accurately.


## Installation

**Recommended: conda with the provided environment file.**

```bash
conda env create -f cpu-env.yml
conda activate cpu-only
```

This installs Python 3.12, PyTorch (CPU), pandas, Typer, and Rich. Once active, all commands below assume your shell is in this env.

If you only need the data-pipeline parts and not the model training, `pip install pandas typer rich` is enough.

## Quick start

```bash
conda activate cpu-only
cd dataset_preparation/
python cli.py inspect          # raw BioRED stats (papers, entities, relations)
python cli.py build            # generate perturbed CSVs in ../output/
python cli.py stats            # per-perturbation breakdown of the CSVs
python cli.py peek -n 3        # eyeball a few sample rows
```

## Phase 1 CLI

Single entry point at [dataset_preparation/cli.py](dataset_preparation/cli.py). Run `python cli.py --help` for the full list. Built with [Typer](https://typer.tiangolo.com/), so every command and flag is type-validated.

### Commands

| Command | What it does |
|---|---|
| `inspect [SPLIT]` | Raw BioRED stats: papers, entities, relations, observed entity & relation types. |
| `build [SPLIT]` | Generate perturbed CSVs (gold + 4 perturbations per gold relation). Flags: `--seed N`, `--no-type-restrict`, `--out PATH`. |
| `stats [SPLIT]` | Per-perturbation breakdown of an already-built CSV. |
| `peek [SPLIT]` | Print a few random sample rows. Filters: `-p PERTURBATION`, `-l LABEL` (0/1), `-n N`. |
| `validate` | Sanity check that the relation types observed in BioRED all match the hardcoded list in `perturbations.py`. |

`SPLIT` is `train` / `dev` / `test` / `all` (defaults to `all` for most commands; `train` for `peek`).

### Examples

```bash
# Stats for just the train split
python cli.py inspect train

# Build with a different seed and disable type-restricted false-positive sampling
python cli.py build train --seed 7 --no-type-restrict

# Look at 5 random direction-swap perturbations from dev
python cli.py peek dev -n 5 -p direction_swap

# Look at correctly-labeled gold samples from train
python cli.py peek train -p gold -l 1
```

The 4 perturbation types you can filter on:

| Perturbation | Label | What changes |
|---|---|---|
| `gold` | 1 | unchanged BioRED relation |
| `label_flip` | 0 | relation type changed to a different one |
| `direction_swap` | 0 | entity_A and entity_B swapped (skipped for symmetric relations: `Association`, `Comparison`) |
| `false_positive` | 0 | invented a relation between two co-mentioned, currently-unrelated entities |
| `false_negative` | 0 | claimed `NoRelation` for a relation that actually exists |

### How to add new commands

The CLI is a single file, [dataset_preparation/cli.py](dataset_preparation/cli.py). To add a new subcommand, follow the same pattern as the existing five.

**Steps:**

1. Write a function decorated with `@app.command()`. The function name becomes the command name.
2. Use **type hints** for every parameter; Typer reads them to build the parser.
3. Use `typer.Argument(...)` for positional args and `typer.Option(...)` for flags. Always include a `help="..."` string so it shows up in `--help`.
4. For enum-like choices (a split, a perturbation type), define an `Enum` subclass at the top of `cli.py` and use it as the type hint. Typer auto-validates and prints the allowed values in `--help`.
5. For pretty output, use the shared `console` (Rich). For tables, use `rich.table.Table`. For errors, raise `typer.BadParameter("...")` (for bad input) or `typer.Exit(code=1)` (for runtime failures).
6. Keep heavy lifting in `data_loader.py`, `perturbations.py`, or `build_dataset.py`. The CLI file should only **orchestrate** and **render**, not contain logic.

**Minimal example:**

```python
@app.command()
def histogram(
    split: Split = typer.Argument(Split.train, help="Which split."),
    column: str = typer.Argument("entity_a_type", help="Column to count."),
) -> None:
    """Count occurrences of values in COLUMN of a built CSV, sorted descending."""
    df = _load_built_csv(split.value, DEFAULT_OUTPUT_DIR)
    if column not in df.columns:
        raise typer.BadParameter(f"Unknown column: {column!r}")

    counts = df[column].value_counts()

    table = Table(title=f"{split.value}: {column}")
    table.add_column(column, style="cyan")
    table.add_column("count", justify="right")
    for value, count in counts.items():
        table.add_row(str(value), str(count))
    console.print(table)
```

After saving, `python cli.py --help` lists the new command automatically. Run `python cli.py histogram --help` to see its arguments.

**Conventions in this repo:**

- Helper functions used only inside `cli.py` start with `_` and live at the bottom of the file.
- Data-side logic does not depend on Typer or Rich. Keep them out of `data_loader.py`, `perturbations.py`, and `build_dataset.py` so those modules stay reusable from notebooks.
- Default to `Split.all` for stat-style commands (inspect, stats, validate), and `Split.train` for sampling commands (peek).
- Every command must have a one-line docstring; that's the description shown in the top-level `--help`.

## Repository structure

```
LLMs_for_NEL/
├── README.md                        ← you are here
├── cpu-env.yml                      ← conda environment spec
├── pubtator_parser.py               ← raw PubTator -> 3 DataFrames (meta, anns, rels)
├── Data/
│   ├── BioRED/                      ← main dataset (Train/Dev/Test in PubTator + JSON + XML)
│   ├── CDR_Data/                    ← Chemical-Disease RE (older baseline)
│   └── MedMention/                  ← entity mentions only (reference)
├── dataset_preparation/
│   ├── __init__.py
│   ├── cli.py                       ← Typer entry point (see above)
│   ├── data_loader.py               ← BioRED loader (wraps pubtator_parser)
│   ├── perturbations.py             ← TrainingSample dataclass + perturbation generators
│   └── build_dataset.py             ← library: build_split, assert_known_relation_types
└── output/                          ← generated CSVs (gitignored)
```

## Example to run the demo.:

`python3 pubtator_parser.py Data/BioRED/Dev.PubTator `

Example usage of the custom data loader this allows you to load all 3 datasets that we are interested in uniformely. Making it easier to work with down the line
```python
from pubtator_parser import parse_pubtator, save_dataframes, load_dataframes, enrich_relations

# Parse — now returns 3 DataFrames
meta, anns, rels = parse_pubtator("Data/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt")

# Join annotations with metadata
combined = anns.merge(meta, on="pmid")

# Filter to just Chemical entities (example)
chemicals = anns[anns["entity_type"] == "Chemical"]

# Enrich relations with human-readable mention names
rels_named = enrich_relations(rels, anns)

# Save — pass rels as the third argument
save_dataframes(meta, anns, rels, prefix="CDR_test", output_dir="output/")

# Load returns 3 DataFrames metadata(abstract and PID), Annotations, Relations 
meta, anns, rels = load_dataframes(prefix="CDR_test", input_dir="output/")
``` 