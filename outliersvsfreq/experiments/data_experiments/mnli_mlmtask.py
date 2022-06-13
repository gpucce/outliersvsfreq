

import json

import numpy as np
import pandas as pd

from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from outliersvsfreq.mlm_analysis import MLMAnalysisTrainer

parser = ArgumentParser()
parser.add_argument(
    "--model_name",
    choices=["bert-base-uncased", "roberta-base"],
    default="bert-base-uncased",
)
args = parser.parse_args()

model_name = args.model_name
device = "cuda"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
idxs_groups = (
    [[308], [381], [308, 381]]
    if model_name == "bert-base-uncased"
    else [[77], [588], [77, 588]]
)

# Params
batch_size = 8
max_length = 256
out_dir = Path("output/data_experiments")
out_dir.mkdir(parents=True, exist_ok=True)
generation_out_dir = out_dir / "generation_output"
generation_out_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset("glue", "mnli")
encoded_ds = ds.map(
    lambda row: tokenizer(
        row["premise"], padding="max_length", truncation=True, max_length=max_length
    ),
    batched=True,
)
encoded_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

args = TrainingArguments(
    generation_out_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
)

set_seed(42)
trainer = MLMAnalysisTrainer(
    model,
    args,
    train_dataset=encoded_ds["train"],
    eval_dataset=encoded_ds["validation_matched"],
    tokenizer=tokenizer,
)

trainer.get_full_generation_output(idxs_groups, max_length=max_length)

word_counts_path = Path("output/data_experiments/word_counts/")
with open(word_counts_path / f"{model_name}_wiki_word_counts.json") as wiki_counts_file:
    wiki_counts = json.load(wiki_counts_file)
with open(word_counts_path / f"{model_name}_book_corpus_word_counts.json") as book_counts_file:
    book_counts = json.load(book_counts_file)

future_table = dict()
similarity_out_dir = out_dir / "similarity_tables"
similarity_out_dir.mkdir(parents=True, exist_ok=True)

for idxs in idxs_groups:
    idxs_name = "_".join([str(i) for i in idxs])

    full_counts_file = (
        f"../data_study/output/{model_name}_wiki&book_wordcount_dict.json"
    )

    wiki_dict = {i: j for i, j in wiki_counts}
    book_dict = {i: j for i, j in book_counts}

    full_counts = dict()
    toks = set([i[0] for i in wiki_counts] + [i[0] for i in book_counts])
    for i in tqdm(toks):
        if i in wiki_dict and i in book_dict:
            full_counts[i] = wiki_dict[i] + book_dict[i]
        elif i in wiki_dict:
            full_counts[i] = wiki_dict[i]
        elif i in book_dict:
            full_counts[i] = book_dict[i]

    def g(x):
        return full_counts[x] if x in full_counts else "0"

    full_counts_dict = {tokenizer(i).input_ids[1]: j for i, j in full_counts.items()}

    def h(x):
        return full_counts_dict[x] if x in full_counts_dict else 0

    generated = pd.read_csv(
        generation_out_dir / f"{model_name}_{idxs_name}_tok_counts.csv"
    )
    realtoks = generated.loc[:, generated.columns.str.contains("real")]
    hidgenerated = generated.loc[:, generated.columns.str.contains("hid")]
    generated = generated.loc[
        :,
        (
            ~generated.columns.str.contains("real")
            & ~generated.columns.str.contains("hid")
        ),
    ]

    newrows = []
    newtoksrows = []
    generated_toks = []
    for i in tqdm(range(generated.shape[0])):
        lastval = np.where(realtoks.iloc[i, :] == tokenizer.pad_token_id)
        if len(lastval[0]) > 0:
            lastval = lastval[0][0]
        else:
            lastval = generated.shape[1]

        generated_frequencies = generated.iloc[i, :lastval].apply(h).to_numpy()
        hidgenerated_frequencies = hidgenerated.iloc[i, :lastval].apply(h).to_numpy()
        loc_errors = hidgenerated_frequencies - generated_frequencies
        mask = np.isfinite(loc_errors)
        toks = realtoks.iloc[i, :lastval].to_numpy()[mask]
        newrows.append(loc_errors[mask])
        newtoksrows.append(toks)
        generated_toks.append(hidgenerated.iloc[i, :lastval].to_numpy())

    errors = np.hstack(newrows)
    toks = np.hstack(newtoksrows)
    gentoks = np.hstack(generated_toks)

    toks_err_df = pd.DataFrame(
        {"tokens": toks, "gen_toks": gentoks, "freq_change": errors}
    )
    toks_err_df["abs_change"] = toks_err_df["freq_change"].abs()
    toks_err_df["tokens"] = toks_err_df["tokens"].apply(tokenizer.decode)
    toks_err_df["gen_toks"] = toks_err_df["gen_toks"].apply(tokenizer.decode)
    toks_err_df["real_freq"] = toks_err_df["tokens"].apply(g)
    toks_err_df["gen_freq"] = toks_err_df["gen_toks"].apply(g)
    toks_err_df.to_csv(
        generation_out_dir / f"{model_name}_{idxs}_all_tokens_in_a_row.csv"
    )
