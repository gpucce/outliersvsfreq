

import json
import numpy as np
import regex as re

from pathlib import Path
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from argparse import ArgumentParser
from itertools import combinations

from outliersvsfreq.parameter_access import choose_outlier_for_finetuning
from outliersvsfreq.parameter_hiding import zero_param_, zero_last_param_

parser = ArgumentParser()

parser.add_argument(
    "--step", type=str, help="Choose between train and eval", choices=["train", "test"]
)

parser.add_argument(
    "--task",
    type=str,
    help="Choose task to train or evaluate",
    choices=[
        "cola",
        "mnli",
        "mnli-mm",
        "mrpc",
        "qnli",
        "qqp",
        "rte",
        "sst2",
        "stsb",
        "wnli",
    ],
)

parser.add_argument(
    "--hiding_type",
    type=str,
    help="Choose the type of hiding",
    choices=["norm", "none"],
)

parser.add_argument("--train_batch_size", type=int, help="set batch size", default=256)
parser.add_argument(
    "--random_seed", type=int, help="set dataset random seed", default=43
)
parser.add_argument("--eval_batch_size", type=int, help="set batch size", default=256)
parser.add_argument(
    "--model_checkpoint", type=str, help="Choose a specific model checkpoint to use"
)

parser.add_argument("--max_length", type=int, default=256, help="Choose the max length")
parser.add_argument("--lr", type=float, default=2.0e-5)
parser.add_argument("--layer_range_length", type=int, default=12)
parser.add_argument("--check_all_idxs", type=bool, default=False)

args = parser.parse_args()

do_train = args.step == "train"
hiding_type = args.hiding_type
max_length = args.max_length
random_seed = args.random_seed
special_sep = args.special_sep
check_all_idxs = args.check_all_idxs
lr = args.lr
task = args.task
layer_range_length = args.layer_range_length
model_checkpoint = args.model_checkpoint
if model_checkpoint == "drozd":
    model_checkpoint = "../pretrained_models/pretrained_models_drozd/sl250.m.gsic.titech.ac.jp:8000/21.11.17_06.30.32_roberta-base_a0057/checkpoints/smpl_400M/hf/"
    model_name = "drozd_model"
elif "../pretrained_models/" in model_checkpoint:
    model_name = re.sub("_$", "", "_".join(model_checkpoint.split("/")[2:])).replace(
        "-", "_"
    )
else:
    model_name = model_checkpoint
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size

actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric("glue", actual_task)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

##### tokenization_parameters
sentence1_key, sentence2_key = task_to_keys[task]

model_and_score_file_info = actual_task
model_and_score_file_info += f"_max_length_{max_length}"
model_and_score_file_info += f"_lr_{lr}"
model_and_score_file_info += f"_batch_size_{train_batch_size}"
model_and_score_file_info += f"_random_seed_{random_seed}"
model_last_dir = f"{model_name}_" + model_and_score_file_info


output_path = Path("output/baselines")
modeldir = output_path / "models" / model_last_dir
scoresdir = output_path / "scores" / model_name


def preprocess_function(examples, max_length):
    if sentence2_key is None:
        tokenized = tokenizer(
            examples[sentence1_key], truncation=True, max_length=max_length
        )
    else:
        tokenized = tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    return tokenized


encoded_dataset = dataset.map(
    lambda x: preprocess_function(x, max_length=max_length), batched=False
)


encoded_dataset = encoded_dataset.shuffle(random_seed)

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=num_labels
)

metric_name = (
    "pearson"
    if task == "stsb"
    else "matthews_correlation"
    if task == "cola"
    else "accuracy"
)


args = TrainingArguments(
    modeldir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    overwrite_output_dir=True,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


validation_key = (
    "validation_mismatched"
    if task == "mnli-mm"
    else "validation_matched"
    if task == "mnli"
    else "validation"
)


if do_train:
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model()
    trainer.save_state()
elif not do_train:
    for layer_range_start in range(0, 13 - layer_range_length):
        if check_all_idxs:
            known_idxs = range(model.config.hidden_size)
        elif model_name == "bert-base-uncased":
            known_idxs = [308, 381]
        elif model_name == "roberta-base":
            known_idxs = [77, 588]
        elif "multiberts_seed_0" in model_name:
            known_idxs = [263, 628]
        elif "multiberts_seed_1" in model_name:
            known_idxs = [218, 674]
        else:
            idxs = choose_outlier_for_finetuning(
                model, n_std=1.5, model_type="LM", topk=10,
            )

            for i in known_idxs:
                if not i in idxs:
                    idxs.append(i)

        param_groups = [[]] + list(combinations(idxs, 1)) + list(combinations(idxs, 2))
        if len(idxs) > 1:
            param_groups += [idxs]

        for param_group in param_groups:

            model = model.from_pretrained(modeldir)
            model = zero_last_param_(
                model,
                param_group,
                layer_range_start,
                layer_range_start + layer_range_length,
            )

            trainer = Trainer(
                model,
                args,
                train_dataset=encoded_dataset["train"],
                eval_dataset=encoded_dataset[validation_key],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
            results = trainer.evaluate()
            scores_filename = model_and_score_file_info.replace(actual_task, task)
            layers_dir = ""
            if layer_range_length < 12:
                layers_dir = f"layers_window_size_{layer_range_length}"
                scores_filename += f"_layers_{layer_range_start}-{layer_range_start+layer_range_length}"
            scores_filename += f"_{list(param_group)}_scores.json"

            scores_dir = scoresdir / layers_dir
            scores_dir.mkdir(exist_ok=True, parents=True)
            print(scores_filename)
            with open(scores_dir / scores_filename, "w") as scores_save_file:
                json.dump(results, scores_save_file)

