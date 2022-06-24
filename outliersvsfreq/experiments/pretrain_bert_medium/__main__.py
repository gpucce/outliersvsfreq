# pylint: disable=line-too-long
import logging
import math
import sys
import json
import datetime
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt

import transformers
import datasets
from datasets import load_dataset, load_metric, concatenate_datasets
from spacy.lang.en import English

from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

nlp = English()
nlp.add_pipe("sentencizer")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are
    going to fine-tune, or train from scratch.
    """

    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained"
            " models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer "
            "(backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running "
            "`transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    path_to_freqs: bool = field(
        default=None, metadata={"help": "Path to a token frequency json file."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    randomize_tokens: bool = field(
        default=False, metadata={"help": "Tries to remove token distribution from the input."}
    )
    is_test: bool = field(
        default=False, metadata={"help": "Runs a test run instead of a full one."},
        dest="feature", action="store_true"
    )
    do_split_in_sentences: bool = field(
        default=True, metadata={"help": "Use or not spacy sentencizer to split in sentences."}
    )
    few_special_tokens: bool = field(
        default=False, metadata={"help": "If true only add [SEP] at the end of sequence."},
    )


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


metric = load_metric("accuracy")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


def split_in_sentences(text_list):
    docs = [nlp(text) for text in text_list]
    return [str(sent).strip() for doc in docs for sent in doc.sents if len(str(sent)) > 10]


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    is_test = data_args.is_test
    randomize_tokens = data_args.randomize_tokens
    max_seq_length = data_args.max_seq_length
    few_special_tokens = data_args.few_special_tokens
    if randomize_tokens:
        training_args.output_dir += "_randomized_tokens"
    if few_special_tokens:
        training_args.output_dir += "_fewspecialtoks"
    today = datetime.datetime.now()
    date_time = today.strftime("%m.%d.%Y_%H:%M:%S")
    training_args.date_time = date_time
    training_args.output_dir += f"_{training_args.date_time}"
    if is_test:
        training_args.output_dir = "test_runs/" + training_args.output_dir
    output_dir_path = Path(training_args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    with open(output_dir_path / "data_args.json", "w") as data_args_out_file:
        json.dump(vars(data_args), data_args_out_file)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # CUSTOM DATASET SETTING
    joint_datasets = [
        load_dataset("wikipedia", "20220301.en", cache_dir=model_args.cache_dir)[
            "train"
        ].remove_columns(["id", "url", "title"]),
        load_dataset("bookcorpus", cache_dir=model_args.cache_dir)["train"],
    ]

    raw_datasets = concatenate_datasets(joint_datasets).train_test_split(test_size=0.1)
    raw_datasets["validation"] = raw_datasets["test"]
    del raw_datasets["test"]

    for i in raw_datasets:
        raw_datasets[i] = raw_datasets[i].shuffle(42)

    do_split_in_sentences = data_args.do_split_in_sentences
    if do_split_in_sentences:
        with training_args.main_process_first(desc="split sentences"):
            splat_raw_datasets = raw_datasets.map(
                lambda batch: {"text": split_in_sentences(batch["text"])},
                remove_columns=[i for i in raw_datasets["train"].column_names if i != "text"],
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                desc="Splitting on sentences.",
            )
    else:
        splat_raw_datasets = raw_datasets

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    model_name = "nreimers/BERT-Medium_L-8_H-512_A-8"
    config = AutoConfig.from_pretrained(model_name, **config_kwargs)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", **tokenizer_kwargs)
    model = AutoModelForMaskedLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = splat_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    def tokenize_function(examples):
        tokenized = {
            k: i
            for k, i in tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            ).items()
        }
        return tokenized

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = splat_raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on every text in dataset",
        )

    def _join_separate_sentences(concatenated_examples, pad):
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        local_max_seq_length = max_seq_length - pad
        if total_length >= local_max_seq_length:
            total_length = (total_length // local_max_seq_length) * local_max_seq_length
        # Split by chunks of max_len.
        return {
            k: [
                t[i : i + local_max_seq_length]
                for i in range(0, total_length, local_max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }

    def group_texts_plus_cls(examples):
        # Concatenate all texts. After removing first token ([CLS]/<s>).
        concatenated_examples = {
            k: list(chain(*[i[1:] for i in examples[k]])) for k in examples.keys()
        }
        result = _join_separate_sentences(concatenated_examples, 1)
        # Add back [CLS]/<s> only at the beginning of the first sentence (e.g. only at the beginnning
        # of the whole sequence)
        return {
            "input_ids": [[tokenizer.cls_token_id] + i for i in result["input_ids"]],
            "attention_mask": [[1] + i for i in result["attention_mask"]],
            "token_type_ids": [[0] + i for i in result["token_type_ids"]],
            "special_tokens_mask": [[1] + i for i in result["special_tokens_mask"]],
        }

    def group_texts_plus_cls_and_sep(examples):
        # Concatenate all texts. After removing first token ([CLS]/<s>).
        concatenated_examples = {
            k: list(chain(*[i[1:-1] for i in examples[k]])) for k in examples.keys()
        }
        result = _join_separate_sentences(concatenated_examples, 2)
        # Add back [CLS]/<s> only at the beginning of the first sentence (e.g. only at the beginnning
        # of the whole sequence) and [SEP]/</s> only at the end of the whole sequence.
        return {
            "input_ids": [
                [tokenizer.cls_token_id] + i + [tokenizer.sep_token_id] for i in result["input_ids"]
            ],
            "attention_mask": [[1] + i + [1] for i in result["attention_mask"]],
            "token_type_ids": [[0] + i + [0] for i in result["token_type_ids"]],
            "special_tokens_mask": [[1] + i + [1] for i in result["special_tokens_mask"]],
        }

    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    with training_args.main_process_first(desc="grouping texts together"):
        joint_tokenized_datasets = tokenized_datasets.map(
            group_texts_plus_cls if not few_special_tokens else group_texts_plus_cls_and_sep,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )

    train_dataset = joint_tokenized_datasets["train"]
    eval_dataset = joint_tokenized_datasets["validation"]

    if is_test:
        df_test_sample = joint_tokenized_datasets["train"]["input_ids"][:10000]
        all_toks = []
        for i in df_test_sample:
            all_toks += i
        n_sep = sum([sum([102 == j for j in i]) for i in df_test_sample])
        n_cls = sum([sum([101 == j for j in i]) for i in df_test_sample])
        n_final_sep = sum([i[-1] == 102 for i in df_test_sample])
        avg_sent_length = sum([len(i) for i in df_test_sample]) / len(df_test_sample)
        n_samples = len(df_test_sample)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.hist(all_toks)
        fig.savefig(output_dir_path / "token_dist.png", bbox_inches="tight")

        with open(
            output_dir_path / f"py_log_split_in_sentences_{do_split_in_sentences}.txt", "w"
        ) as log_file:
            log_file.write(f"do split in sentences: {do_split_in_sentences}\n")
            log_file.write(f"few special tokens: {few_special_tokens}\n")
            log_file.write(f"n_samples: {n_samples}\n")
            log_file.write(f"n_sep: {n_sep}\n")
            log_file.write(f"n_cls: {n_cls}\n")
            log_file.write(f"n_final_sep: {n_final_sep}\n")
            log_file.write(f"seq length: {avg_sent_length}\n")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.save_model(Path(trainer.args.output_dir) / "checkpoint-0")

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
