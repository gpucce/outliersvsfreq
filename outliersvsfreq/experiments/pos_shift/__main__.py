from pathlib import Path

import torch
import spacy

from transformers import AutoModelForMaskedLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset

from outliersvsfreq.mlm_analysis import MLMAnalysisTrainer

nlp = spacy.load("en_core_web_sm")


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name_or_path = "bert-base-uncased"

# %%
upos_dict = {
    "NOUN": 0,
    "PUNCT": 1,
    "ADP": 2,
    "NUM": 3,
    "SYM": 4,
    "SCONJ": 5,
    "ADJ": 6,
    "PART": 7,
    "DET": 8,
    "CCONJ": 9,
    "PROPN": 10,
    "PRON": 11,
    "X": 12,
    "_": 13,
    "ADV": 14,
    "INTJ": 15,
    "VERB": 16,
    "AUX": 17,
}

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

if model_name_or_path == "roberta-base":
    idxs_groups = [[], [77], [588], [77, 588]]
elif model_name_or_path == "bert-base-uncased":
    idxs_groups = [[], [308], [381], [308, 381]]

ud = load_dataset("wikitext", "wikitext-2-v1")
ud = ud.filter(lambda x: len(x["text"]) > 70)
ud = ud.shuffle(42)

encoded_ds = ud.map(
    lambda x: tokenizer(x["text"], padding="max_length", max_length=128, truncation=True)
)
encoded_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

batch_size = 8

args = TrainingArguments(
    "puppa",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
)

trainer = MLMAnalysisTrainer(
    model,
    args,
    train_dataset=encoded_ds["train"],
    eval_dataset=encoded_ds["validation"],
    tokenizer=tokenizer,
)

output = trainer.correlate_full_to_hidden_pos(idxs_groups)

for col in output.columns:
    output = output.loc[output.loc[:, col].str.len() > 0]
for col in output.columns:
    output["pos_" + col] = output[col].apply(lambda x: nlp(x)[0].pos_)

out_dir = Path("output/pos_shift/")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{model_name_or_path}_masked_generation.csv"
output.to_csv(out_path, index=False)
