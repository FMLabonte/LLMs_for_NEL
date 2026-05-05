import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from eval_finetuning_pipeline import PipelineData, evaluate_finetuning, prepare_data

MODEL_NAME = "bioformers/bioformer-8L"
PUBTATOR_FILE = Path("/Users/chris/git/LLMs_for_NEL/Data/BioRED/Dev.PubTator")
PROMPT_TEMPLATE_FILE = Path(__file__).parent / "prompting" / "bioformer-prompting.yaml"

# ── 1. Prepare data ────────────────────────────────────────────────────────────
data: PipelineData = prepare_data(PUBTATOR_FILE, MODEL_NAME, PROMPT_TEMPLATE_FILE)

# ── 2. Load model ──────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=data.id2label,
    label2id=data.label2id,
).to(data.device)

# ── 3. Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Compute accuracy and weighted F1 from Trainer eval output."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# ── 4. Training arguments (MPS-optimized) ─────────────────────────────────────
args = TrainingArguments(
    output_dir="./bioformer-finetuned",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=False,           # MPS does not support fp16 training
    logging_steps=50,
    dataloader_num_workers=0,  # MPS + multiprocessing can deadlock
)

# ── 5. Train ───────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=data.train_ds,
    eval_dataset=data.val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# ── 6. Save ────────────────────────────────────────────────────────────────────
trainer.save_model("./bioformer-finetuned-final")
data.tokenizer.save_pretrained("./bioformer-finetuned-final")

# ── 7. Evaluate base vs. fine-tuned ───────────────────────────────────────────
evaluate_finetuning(
    val_ds=data.val_ds,
    val_labels=data.val_df["label"].tolist(),
    base_model_name=MODEL_NAME,
    finetuned_model_path=Path("./bioformer-finetuned-final"),
    tokenizer=data.tokenizer,
    device=data.device,
    id2label=data.id2label,
    label2id=data.label2id,
)
