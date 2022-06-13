# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
# ---

# %%
from inspect import ArgInfo
from datasets import load_dataset, load_metric, ClassLabel
import random
import torch
import pandas as pd
import numpy as np
import json
import re
from copy import deepcopy
from pathlib import Path
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)

# %%
from itertools import combinations

# %%
from argparse import _ArgumentGroup, ArgumentParser

# %%
import sys

# %%
from outlier_analysis.parameter_access import choose_outlier_for_finetuning
from outlier_analysis.parameter_hiding import zero_wav2_vec_param_

# %%
parser = ArgumentParser()

# %%
parser.add_argument("--step", type=str, choices=["train", "test"])

# %%
args = parser.parse_args()
do_train = args.step == "train"
output_dir = Path("output")
output_dir.mkdir(exist_ok=True, parents=True)

# %%
timit = load_dataset("timit_asr")

# %%
timit = timit.remove_columns(
    [
        "phonetic_detail",
        "word_detail",
        "dialect_region",
        "id",
        "sentence_type",
        "speaker_id",
    ]
)

# %%
chars_to_ignore_regex = '[\,\?\.\!\-\;\:"]'


# %%
def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, "", batch["text"]).lower()
    return batch


# %%
timit = timit.map(remove_special_characters)


# %%
def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


# %%
vocabs = timit.map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=timit.column_names["train"],
)

# %%
vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

# %%
vocab_dict = {v: k for k, v in enumerate(vocab_list)}

# %%
## CHANGES TO VOCAB
vocab_dict["|"] = vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
del vocab_dict[" "]

# %%
with open("vocab.json", "w") as vocab_file:
    try:
        vocab_dict = json.load(vocab_file)
    except:
        json.dump(vocab_dict, vocab_file)

# %%
try:
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./output/processors/")
except:
    tokenizer = Wav2Vec2CTCTokenizer(
        "./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )

# %%
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False,
)

# %%
try:
    processor = Wav2Vec2Processor.from_pretrained("./output/processors/")
except:
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

# %% [markdown]
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large")


# %%
def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


# %%
timit = timit.map(
    prepare_dataset, remove_columns=timit.column_names["train"], num_proc=4
)


# %%
class DataCollatorCTCWithPadding:
    def __init__(
        self,
        processor,
        padding=True,
        max_length=None,
        max_length_labels=None,
        pad_to_multiple_of=None,
        pad_to_multiple_of_labels=None,
    ):
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

    def __call__(self, features):
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


# %%
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# %%
wer_metric = load_metric("wer")


# %%
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# %%
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

# %%
model.freeze_feature_extractor()


# %%
training_args = TrainingArguments(
    output_dir="output/training_output",
    group_by_length=True,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    num_train_epochs=50,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=100,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
    resume_from_checkpoint=False,
    overwrite_output_dir=True,
)

# %%
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=timit["train"],
    eval_dataset=timit["test"],
    tokenizer=processor.feature_extractor,
)

# %%
if do_train:
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    tokenizer.save_pretrained("./output/processors/")
    processor.save_pretrained("./output/processors/")


# %%
def map_to_result(newmodel, batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = newmodel(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.batch_decode(batch["labels"], group_tokens=False)

    return batch


# %%
model.to("cpu")
outliers_idxs = choose_outlier_for_finetuning(model, model_type="wav2vec", n_std=3.0)

# %%
output = dict()

# %%
for idxs in (
    [[]] + list(combinations(outliers_idxs, 1)) + list(combinations(outliers_idxs, 2))
):

    model = Wav2Vec2ForCTC.from_pretrained(
        f"./output/training_output/",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    zero_wav2_vec_param_(model, idxs)
    model.to("cuda")
    # results = timit["test"].map(
    #     lambda x: map_to_result(model, x), remove_columns=timit["test"].column_names
    # )
    # out_wer = wer_metric.compute(
    #     predictions=results["pred_str"], references=results["text"]
    # )

    trainer = Trainer(
        model=model,
        # data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=timit["train"],
        eval_dataset=timit["test"],
        tokenizer=processor.feature_extractor,
    )
    results = trainer.evaluate()

    output[str(idxs)] = results

# %%
with open("output/results.json", "w") as results_output_file:
    json.dump(output, results_output_file)
