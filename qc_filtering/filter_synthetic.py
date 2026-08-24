"""Task 4, step 1: score every relation claim in Fred's synthetic abstracts with the
run-2 QC model, so the synthetic sets can be split into QC-kept and QC-rejected.

One row per (model, split, paper, generation, relation). The abstract-level pass/fail
decision is NOT taken here, it is applied afterwards in decide.py, so the decision
rule can be changed without re-running 46 minutes of inference.

Run:
    python filter_synthetic.py                 # everything, ~46 min on this CPU
    python filter_synthetic.py --limit 5       # 5 papers per file, smoke test
"""
import argparse, json, os, re, time, zipfile
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OUT_DIR = Path(__file__).resolve().parent
REPO = OUT_DIR.parent
SYN_DIR = REPO / "Data/Synthetic abstracts"
OUT_CSV = OUT_DIR / "synthetic_relation_scores.csv"

# The QC checkpoint is 438 MB and is deliberately not in the repo. Set
# QC_MODEL_DIR, or ask Houman for the run-2 model directory.
MODEL_DIR = Path(os.environ.get("QC_MODEL_DIR", "") or
                 "../work/qc_model_run2/qc-pubmedbert-final")

MAX_LEN = 512   # must match the run-2 notebook; 256 over-truncates long abstracts

# "- name_a (TypeA) --[Relation_Type]--> name_b (TypeB)"
REL_RE = re.compile(r"^-\s*(.+?)\s*\(([^()]+)\)\s*--\[(\w+)\]-->\s*(.+?)\s*\(([^()]+)\)\s*$")


def parse_relations(prompt: str) -> list[dict]:
    """Pull the RELATIONS block out of the generation prompt.

    The prompt also contains few-shot EXAMPLE blocks with their own RELATIONS
    sections, so only the first block counts: it is the one this paper's
    abstracts were told to express.
    """
    start = prompt.find("RELATIONS (these are the ONLY relations to include):")
    if start == -1:
        return []
    body = prompt[start:].split("\n", 1)[1]
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            break               # blank line ends the block, before EXAMPLE FORMAT
        m = REL_RE.match(line)
        if m:
            a, ta, rel, b, tb = m.groups()
            out.append({"entity_a": a, "type_a": ta, "relation_type": rel,
                        "entity_b": b, "type_b": tb})
    return out


def clean_abstract(text: str) -> str:
    """Strip the '**Abstract:**' heading and markdown bold the generator adds."""
    text = re.sub(r"^\s*\**\s*Abstract:?\s*\**\s*", "", text.strip(), flags=re.I)
    return re.sub(r"\*\*", "", text).strip()


def _ensure_synthetic_dir() -> Path:
    """Unpack Data/Synthetic abstracts.zip next to itself if it has not been."""
    if not SYN_DIR.exists():
        zip_path = REPO / "Data/Synthetic abstracts.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"neither {SYN_DIR} nor {zip_path} exists")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(REPO / "Data")
    return SYN_DIR


def build_rows(limit: int = 0) -> pd.DataFrame:
    rows, skipped = [], []
    for path in sorted(_ensure_synthetic_dir().glob("results_qwen3_*.json")):
        model, split = re.match(r"results_(qwen3_\d+b)_(\w+)\.json", path.name).groups()
        papers = json.load(open(path))
        if limit:
            papers = papers[:limit]
        for p in papers:
            rels = parse_relations(p["prompt"])
            if len(rels) != p["num_relations"]:
                skipped.append((path.name, p["paper_id"], len(rels), p["num_relations"]))
            for gen_name, abstract in p["synthetic_abstracts"].items():
                abstract = clean_abstract(abstract)
                for i, r in enumerate(rels):
                    rows.append({
                        "model": model, "split": split, "paper_id": str(p["paper_id"]),
                        "generation": gen_name, "rel_idx": i,
                        "num_relations": p["num_relations"], "num_entities": p["num_entities"],
                        "abstract_words": len(abstract.split()),
                        **r, "abstract": abstract,
                    })
    if skipped:
        print(f"WARNING: relation-count mismatch on {len(skipped)} papers "
              f"(parsed vs num_relations). First 5: {skipped[:5]}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="papers per file, 0 = all")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_rows(args.limit)
    print(f"{len(df):,} relation claims to score "
          f"({df.paper_id.nunique()} papers, {df.groupby(['model','split']).ngroups} file groups)")

    torch.set_num_threads(torch.get_num_threads())
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # Same prompt shape the QC model was trained and evaluated with.
    texts = (
        "Relation: " + df.entity_a + " -> " + df.relation_type + " -> " + df.entity_b
        + "\nContext: " + df.abstract
    ).tolist()

    probs, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            enc = tok(texts[i:i + args.batch], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt")
            probs.extend(torch.softmax(model(**enc).logits, dim=-1)[:, 1].tolist())
            if (i // args.batch) % 20 == 0:
                el = time.time() - t0
                done = i + args.batch
                rate = done / max(el, 1e-9)
                left = (len(texts) - done) / max(rate, 1e-9) / 60
                print(f"{done:,}/{len(texts):,}  {rate:.1f} ex/s  ~{left:.0f} min left", flush=True)
    df["prob_supported"] = probs

    df.drop(columns=["abstract"]).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  rows={len(df):,}  elapsed={(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
