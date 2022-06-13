from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    logging,
)

logging.set_verbosity_error()

from outlier_analysis.parameter_hiding import zero_param_, zero_inner_attention_
from outlier_analysis.parameter_access import choose_outlier_for_finetuning
from outlier_analysis.outlier_correlation import OutlierAnalysisTrainer
from outlier_analysis.plotting import (
    stack_plot_corr,
    stack_plot_freq,
    embellish_corr_to_att,
    embellish_corr_to_freq,
)

import random
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np


from scipy.stats import pearsonr
from argparse import ArgumentParser

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

plot_output_path = (
    Path("output")
    / Path("outlier_correlation")
    / model_name_or_path
    / ds_name
)

sentence1_key = "text"

def preprocess_function(examples, max_length):
    return tokenizer(
        examples[sentence1_key],
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
    plot_output_path,
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


freqs, _ = full_trainer.get_frequency(
    [
        f"../mlm_tests/output/word_counts/{freq_file_model_name}_wiki_word_counts.json",
        f"../mlm_tests/output/word_counts/{freq_file_model_name}_book_corpus_word_counts.json",
    ],
    avoid_special_toks=True,
)

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

makfig, makax = stack_plot_corr(
    [
        {i: [k.abs() for k in j] for i, j in hidden_states.items()},
        {i: [k.abs() for k in j] for i, j in hidden_states_no_special.items()},
    ],
    [attentions, attentions_no_special],
    [masks, masks_no_special],
    outliers_idxs=outliers_idxs,
    random_dims=random_dims,
)


loc_path = plot_output_path / "pre_trained_outlier_correlation_with_attention"
loc_path.mkdir(exist_ok=True, parents=True)

embellish_corr_to_att(
    makfig, makax, str(loc_path / "subplot.png"), outliers_idxs=outliers_idxs
)

makfig1, makfig2, makax = stack_plot_freq(
    freqs,
    [
        {i: [k.abs() for k in j] for i, j in hidden_states.items()},
        {i: [k.abs() for k in j] for i, j in hidden_states_no_special.items()},
    ],
    [masks.bool(), masks_no_special.bool()],
    corr_func=pearsonr,
    outliers_idxs=outliers_idxs,
    random_dims=random_dims,
)

loc_path = plot_output_path / f"pre_trained_outlier_correlation_with_frequency"
loc_path.mkdir(parents=True, exist_ok=True)

embellish_corr_to_freq(makfig1, makfig2, makax, str(loc_path / "subplot.png"))
