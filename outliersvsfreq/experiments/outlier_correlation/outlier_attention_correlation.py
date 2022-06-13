
import torch
import random
from pathlib import Path
from argparse import ArgumentParser

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    logging,
)

from outliersvsfreq.outlier_correlation import OutlierAnalysisTrainer

logging.set_verbosity_error()

parser = ArgumentParser()
parser.add_argument(
    "--model_name_or_path", type=str,
    default="bert-base-uncased"
)
parser.add_argument(
    "--step", type=str,
    default="pre-training"
)
parser.add_argument(
    "--freq_file_path", type=str
)

args = parser.parse_args()
model_name_or_path = args.model_name_or_path
step = args.step
freq_file_path = args.freq_file_path

if model_name_or_path == "roberta-base":
    outliers_idxs = [77, 588]
elif model_name_or_path == "bert-base-uncased":
    outliers_idxs = [308, 381]
elif "multiberts_seed_1" in model_name_or_path:
    outliers_idxs = [218, 674]
elif "pretrained_models_drozd" in model_name_or_path:
    outliers_idxs = [468, 477]


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path, num_labels=3
)

ds_name = "wikitext"
output_path = Path("output/outlier_correlations")

def preprocess_function(examples, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

ds = load_dataset("wikitext", "wikitext-2-v1")
ds = ds.filter(lambda x: len(x["text"]) > 70)

encoded_dataset = ds.map(lambda x: preprocess_function(x, max_length=128), batched=True)

metric_name = "accuracy"
batch_size = 16
lr = 2.0e-5


args = TrainingArguments(
    output_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=6,
    logging_strategy="steps",
    logging_steps=50,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    # overwrite_output_dir=True,
    resume_from_checkpoint=True,
)


validation_key = "validation"

full_trainer = OutlierAnalysisTrainer(
    model.to("cuda"),
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
)


if model_name_or_path == "bert-base-uncased" or "multiberts" in model_name_or_path:
    freq_file_model_name = "bert-base-uncased"
elif model_name_or_path == "roberta-base" or "drozd" in model_name_or_path:
    freq_file_model_name = "roberta-base"


freqs, masks_no_special = full_trainer.get_frequency(
    [
        f"../data_experiments/output/word_counts/{freq_file_model_name}_wiki_word_counts.json",
        f"../data_experiments/output/word_counts/{freq_file_model_name}_book_corpus_word_counts.json",
    ],
    avoid_special_toks=True,
)

torch.save(freqs, output_path / "freqs.bin")

random_dims = random.sample(set(range(768)).difference(outliers_idxs), 10)

(
    hidden_states_no_special,
    attentions_no_special,
    masks_no_special,
) = full_trainer.correlate_outliers_and_attentions(
    outliers_idxs + random_dims, avoid_special_toks=True
)

hidden_states, attentions, masks = full_trainer.correlate_outliers_and_attentions(
    outliers_idxs + random_dims, avoid_special_toks=False
)

torch.save(hidden_states_no_special, output_path / "hidden_states_no_special.bin")
torch.save(attentions_no_special, output_path / "attentions_no_special.bin")
torch.save(masks_no_special, output_path / "masks_no_special.bin")

torch.save(hidden_states, output_path / "hidden_states_special.bin")
torch.save(attentions, output_path / "attentions_special.bin")
torch.save(masks, output_path / "masks_special.bin")
