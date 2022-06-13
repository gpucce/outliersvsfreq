from unittest import result
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    logging,
    DataCollatorForLanguageModeling,
    Trainer,
)
import random
from itertools import combinations

logging.set_verbosity_error()
from datasets import load_dataset
import torch
from pathlib import Path
import sys

from outlier_analysis.parameter_access import choose_outlier_for_finetuning
from outlier_analysis.gradient_analysis import *
from outlier_analysis.parameter_hiding import zero_last_param_, zero_param_


from tqdm.auto import tqdm

import json
import datetime

from argparse import ArgumentParser

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=2.0e-5)
parser.add_argument("--model_name", type=str)
parser.add_argument("--layer_range_length", type=int, default=1)
cli_args = parser.parse_args()

model_name_or_path = cli_args.model_name_or_path
output_dir = Path("output/data_experiments/mlm_loss") / "_".join(model_name_or_path.split("/")[-3:])
layer_range_length = cli_args.layer_range_length

# PARAMS
exp_params = {
    "experiment_type": "MLM_loss",
    "batch_size": 64,
    "layer_idx": 12,
    "lr": cli_args.lr,
    "mlm_loss_scale": 0.01,
    "constrained_layers": 10,
    "constrained_heads": 10,
    "n_epochs": 1,
    "topk": 10000,
    "max_length": 128,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

### add subdir to output_dir path

output_dir.mkdir(exist_ok=True, parents=True)
with open(output_dir / "experiment_params.json", "w") as exp_param_out_file:
    json.dump(exp_params, exp_param_out_file)

output_dir /= "results_by_layer"
output_dir.mkdir(exist_ok=True, parents=True)

ds = load_dataset("wikitext", "wikitext-2-v1",)
ds = ds.filter(lambda x: len(x["text"].split()) > 70)

exp_params["n_samples"] = exp_params["n_epochs"] * ds["train"].num_rows

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast=True)
model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

encoded_dataset = ds.map(
    lambda x: tokenizer(
        x["text"],
        padding="max_length",
        truncation=True,
        max_length=exp_params["max_length"],
    ),
    batched=True,
)

encoded_dataset = encoded_dataset.shuffle(42)
encoded_dataset.set_format(type="pytorch", columns=["input_ids", "attention_mask"])
encoded_dataset = encoded_dataset["train"]

if model_name_or_path == "bert-base-uncased":
    outliers_idxs = [[], [308], [381], [308, 381]]
elif model_name_or_path == "roberta-base":
    outliers_idxs = [[], [77], [588], [77, 588]]
elif "multiberts_seed_1" in model_name_or_path:
    outliers_idxs = [[], [218], [674], [218, 674]]
elif "bert_medium" in model_name_or_path:
    outliers_idxs = [[]] + [
        i for i in choose_outlier_for_finetuning(model, model_type="LM", n_std=2)
    ]
print(outliers_idxs)

args = TrainingArguments(
    output_dir,
    per_device_train_batch_size=exp_params["batch_size"],
    per_device_eval_batch_size=exp_params["batch_size"],
    dataloader_drop_last=True,
    overwrite_output_dir=True,
    learning_rate=exp_params["lr"],
)

for layer_start in range(0, 13 - layer_range_length):
    layer_idx = f"layers_{layer_start}-{layer_start + layer_range_length}"
    for outlier_idx in outliers_idxs:
        new_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        zero_last_param_(
            new_model, [outlier_idx], layer_start, layer_start + layer_range_length
        )
        trainer = Trainer(
            new_model,
            args,
            train_dataset=encoded_dataset,
            eval_dataset=encoded_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer),
        )

        results = trainer.evaluate()

        with open(
            output_dir / f"mlm_loss_{layer_idx}_{outlier_idx}.json", "w"
        ) as output_file:
            json.dump(results, output_file)
