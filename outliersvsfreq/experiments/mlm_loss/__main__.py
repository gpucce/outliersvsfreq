import json
from pathlib import Path
from argparse import ArgumentParser

from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    logging,
    DataCollatorForLanguageModeling,
    Trainer,
)

import torch
from outliersvsfreq.parameter_access import choose_outlier_for_finetuning
from outliersvsfreq.parameter_hiding import zero_last_param_

logging.set_verbosity_error()

parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=2.0e-5)
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--layer_range_length", type=int, default=1)
cli_args = parser.parse_args()


def main():

    model_name_or_path = cli_args.model_name_or_path
    output_dir = Path("output/mlm_loss") / "_".join(model_name_or_path.split("/")[-3:])
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

    ### add subdir to output_dir path

    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / "experiment_params.json", "w") as exp_param_out_file:
        json.dump(exp_params, exp_param_out_file)

    ds = load_dataset(
        "wikitext",
        "wikitext-2-v1",
    )
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
    encoded_dataset = encoded_dataset["validation"]

    if model_name_or_path == "bert-base-uncased":
        outliers_idxs = [[], [308], [381], [308, 381]]
    elif "roberta-base" in model_name_or_path:
        outliers_idxs = [[], [77], [588], [77, 588]]
    elif "multiberts_seed_1" in model_name_or_path:
        outliers_idxs = [[], [218], [674], [218, 674]]
    elif "bert_medium" in model_name_or_path:
        outliers_idxs = [[]] + list(choose_outlier_for_finetuning(model, model_type="LM", n_std=2))

    args = TrainingArguments(
        output_dir,
        per_device_train_batch_size=exp_params["batch_size"],
        per_device_eval_batch_size=exp_params["batch_size"],
        dataloader_drop_last=True,
        overwrite_output_dir=True,
        learning_rate=exp_params["lr"],
    )
    out = {}
    for layer_start in range(0, 13 - layer_range_length):
        layer_id = f"{layer_start}_{layer_start + layer_range_length}"
        out[layer_id] = {}
        layer_id_out = out[layer_id]
        for outlier_idx in outliers_idxs:
            outlier_idx_string = (
                str(outlier_idx).replace("[", "").replace("]", "").replace(", ", "_")
            )
            # layer_id_out[outlier_idx_string] = layer_id_out.get(outlier_idx_string, [])
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
            layer_id_out[outlier_idx_string] = results["eval_loss"]

            with open(output_dir / "results_by_layer.json", "w") as output_file:
                json.dump(out, output_file)


if __name__ == "__main__":
    main()
