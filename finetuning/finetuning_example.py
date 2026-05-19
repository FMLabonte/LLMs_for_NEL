# %%
# https://learnopencv.com/fine-tuning-bert/
# %%
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)

import evaluate
import glob
import numpy as np

BATCH_SIZE = 16
NUM_PROCS = 32
LR = 0.00005
EPOCHS = 5
MODEL = 'bert-base-uncased'
OUT_DIR = 'arxiv_bert'

# %%

# %%

# %%
train_dataset = load_dataset("ccdv/arxiv-classification", split='train')
valid_dataset = load_dataset("ccdv/arxiv-classification", split='validation')
test_dataset = load_dataset("ccdv/arxiv-classification", split='test')
print(train_dataset)
print(valid_dataset)
print(test_dataset)
# %%

# %%
train_dataset[0]
# %%

# %%
train_dataset[0].keys()
# %%
id2label = {
    0: "math.AC",
    1: "cs.CV",
    2: "cs.AI",
    3: "cs.SY",
    4: "math.GR",
    5: "cs.CE",
    6: "cs.PL",
    7: "cs.IT",
    8: "cs.DS",
    9: "cs.NE",
    10: "math.ST"
}
label2id = {
    "math.AC": 0,
    "cs.CV": 1,
    "cs.AI": 2,
    "cs.SY": 3,
    "math.GR": 4,
    "cs.CE": 5,
    "cs.PL": 6,
    "cs.IT": 7,
    "cs.DS": 8,
    "cs.NE": 9,
    "math.ST": 10
}
# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL)
# %%
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
    )

# %%
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=BATCH_SIZE,
    num_proc=NUM_PROCS
)

tokenized_valid = valid_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=BATCH_SIZE,
    num_proc=NUM_PROCS
)

tokenized_test = test_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=BATCH_SIZE,
    num_proc=NUM_PROCS
)
# %%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# %%
tokenized_sample = preprocess_function(train_dataset[0])
print(tokenized_sample)
print(f"Length of tokenized IDs: {len(tokenized_sample.input_ids)}")
print(f"Length of attention mask: {len(tokenized_sample.attention_mask)}")
# %%
tokenized_sample = preprocess_function(train_dataset[0])
print(tokenized_sample)
# %%
accuracy = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=11,
    id2label=id2label,
    label2id=label2id,
)
# %%
model = model.to("cuda")
# %%
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=3,
    report_to='tensorboard',
    fp16=True,
    # dataloader_num_workers=0,
)
# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

history = trainer.train()

	
trainer.evaluate(tokenized_test)
