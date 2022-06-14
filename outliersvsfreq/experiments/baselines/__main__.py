import json
from pathlib import Path
from argparse import ArgumentParser
from itertools import combinations

import numpy as np

from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)


from outliersvsfreq.parameter_access import choose_outlier_for_finetuning
from outliersvsfreq.parameter_hiding import zero_last_param_


def main():

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
    parser.add_argument("--random_seed", type=int, help="set dataset random seed", default=43)
    parser.add_argument("--eval_batch_size", type=int, help="set batch size", default=256)
    parser.add_argument(
        "--model_name_or_path", type=str, help="Choose a specific model checkpoint to use"
    )

    parser.add_argument("--max_length", type=int, default=256, help="Choose the max length")
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--layer_range_length", type=int, default=12)
    parser.add_argument("--check_all_idxs", type=bool, default=False)

    args = parser.parse_args()

    do_train = args.step == "train"
    max_length = args.max_length
    random_seed = args.random_seed
    set_seed(random_seed)
    check_all_idxs = args.check_all_idxs
    lr = args.lr
    task = args.task
    layer_range_length = args.layer_range_length
    model_name_or_path = args.model_name_or_path

    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric("glue", actual_task)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

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
    output_path = Path("output/baselines")
    if not Path(model_name_or_path).exists():

        model_and_score_file_info = actual_task
        model_and_score_file_info += f"_max_length_{max_length}"
        model_and_score_file_info += f"_lr_{lr}"
        model_and_score_file_info += f"_batch_size_{train_batch_size}"
        model_and_score_file_info += f"_random_seed_{random_seed}"
        model_last_dir = f"{model_name_or_path}_" + model_and_score_file_info
        modeldir = output_path / "models" / model_last_dir
        scoresdir = output_path / "scores" / model_last_dir
    else:
        modeldir = model_name_or_path
        scoresdir = Path(str(model_name_or_path).replace("/models/", "/scores/"))

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples, max_length):
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        else:
            return tokenizer(
                examples[sentence1_key],
                examples[sentence2_key],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

    encoded_dataset = dataset.map(
        lambda x: preprocess_function(x, max_length=max_length), batched=True
    )
    encoded_dataset.set_format(type="torch")
    encoded_dataset = encoded_dataset.shuffle(random_seed)

    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels
    )

    metric_name = (
        "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
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
        idxs = choose_outlier_for_finetuning(
            model,
            n_std=2,
            model_type="LM",
            topk=10,
        )

        model_name_or_path_str = str(model_name_or_path)
        if check_all_idxs:
            known_idxs = range(model.config.hidden_size)
        elif model_name_or_path_str == "bert-base-uncased":
            known_idxs = [308, 381]
        elif "roberta-base" in model_name_or_path_str:
            known_idxs = [77, 588]
        elif "multiberts_seed_0" in model_name_or_path_str:
            known_idxs = [263, 628]
        elif "multiberts_seed_1" in model_name_or_path_str:
            known_idxs = [218, 674]
        else:
            for i in known_idxs:
                if not i in idxs:
                    idxs.append(i)

        param_groups = [[]] + list(combinations(idxs, 1))  # + list(combinations(idxs, 2))
        if len(idxs) > 1:
            param_groups += [idxs]

        param_groups = [[i] for i in known_idxs]

        out = {}
        for layer_range_start in range(0, 13 - layer_range_length):
            layer_id = f"{layer_range_start}_{layer_range_start + layer_range_length}"
            out[layer_id] = {}
            layer_id_out = out[layer_id]
            for param_group in param_groups:
                param_group_string = (
                    str(param_group).replace("[", "").replace("]", "").replace(", ", "_")
                )
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
                layer_id_out[param_group_string] = results

        scoresdir.mkdir(exist_ok=True, parents=True)
        scores_filename = f"layer_size_{layer_range_length}_results.json"
        with open(scoresdir / scores_filename, "w") as scores_save_file:
            json.dump(out, scores_save_file)


if __name__ == "__main__":
    main()
