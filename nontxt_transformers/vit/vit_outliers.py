
import random
import os
import json
from copy import deepcopy
from itertools import accumulate, combinations

from datasets import load_metric
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import (
    ViTForImageClassification,
    ViTFeatureExtractor,
    Trainer,
    TrainingArguments,
)

from outliersvsfreq.parameter_hiding import zero_vit_param_
from outliersvsfreq.parameter_access import choose_outlier_for_finetuning

import numpy as np
import torch
from torch.optim import SGD
import torchvision
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from argparse import ArgumentParser
from datetime import datetime

torch.manual_seed(42)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--step", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--cifar_data", type=str, default="10", choices=["10", "100"])
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    outdir = args.outdir
    do_train = args.step == "train"
    cifar_data = args.cifar_data
    n_epochs = args.n_epochs
else:
    do_train = False
    cifar_data = "100"
    n_epochs = 3
    outdir = "puppa"


DOWNLOAD_PATH = "data"
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 256


if cifar_data == "100":
    with open("./cifar100_labels2idx.json") as labelsfile:
        labels2idx = json.load(labelsfile)
elif cifar_data == "10":
    labels2idx = {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }

idx2labels = {j: i for i, j in labels2idx.items()}
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k", idx2labels=idx2labels, labels2idx=labels2idx
)

device = "cuda"
metric = load_metric("accuracy")

normalize = Normalize(
    mean=feature_extractor.image_mean, std=feature_extractor.image_std
)
_train_transforms = Compose(
    [
        RandomResizedCrop(feature_extractor.size),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)
_val_transforms = Compose(
    [
        Resize(feature_extractor.size),
        CenterCrop(feature_extractor.size),
        ToTensor(),
        normalize,
    ]
)

if cifar_data == "100":

    train_set = torchvision.datasets.CIFAR100(
        DOWNLOAD_PATH,
        train=True,
        download=True,
        # transform=transform_mnist,
        transform=_train_transforms,
        target_transform=lambda x: {"labels": x},
    )

    test_set = torchvision.datasets.CIFAR100(
        DOWNLOAD_PATH,
        train=False,
        download=True,
        transform=_val_transforms,
        target_transform=lambda x: {"labels": x},
    )

    mymodel = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=100,
    )

elif cifar_data == "10":

    train_set = torchvision.datasets.CIFAR10(
        DOWNLOAD_PATH,
        train=True,
        download=True,
        transform=_train_transforms,
        target_transform=lambda x: {"labels": x},
    )

    test_set = torchvision.datasets.CIFAR10(
        DOWNLOAD_PATH,
        train=False,
        download=True,
        transform=_val_transforms,
        target_transform=lambda x: {"labels": x},
    )

    mymodel = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=10,
    )

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=BATCH_SIZE_TEST, shuffle=True
)

test_set.map(lambda x: {**x[0], **x[1]})

train_set.map(lambda x: {**x[0], **x[1]})

dsname = str(train_set).split("\n")[0].split()[1].lower()

def mycollate(x):
    i1 = {
        "pixel_values": torch.cat(
            # [sample[0]["pixel_values"].unsqueeze(0) for sample in x], axis=0
            [sample[0].unsqueeze(0) for sample in x],
            axis=0,
        )
    }
    i2 = {"labels": torch.tensor([sample[1]["labels"] for sample in x])}
    return {**i1, **i2}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


if outdir == None:
    outdir = f"{dsname}_trainer_output"
    outdir += f"_nepochs_{n_epochs}"
    outdir += f"_{datetime.now()}"

args = TrainingArguments(
    output_dir=outdir,
    per_device_train_batch_size=BATCH_SIZE_TRAIN,
    per_device_eval_batch_size=BATCH_SIZE_TEST,
    save_steps=2000,
    logging_steps=10,
    metric_for_best_model="accuracy",
    overwrite_output_dir=True,
    num_train_epochs=n_epochs,
    warmup_ratio=0.1,
    learning_rate=0.0001,
    lr_scheduler_type="cosine",
    gradient_accumulation_steps=1,
    max_grad_norm=1,
    weight_decay=0,
    evaluation_strategy="epoch",
    remove_unused_columns=False,
)

decay_parameters = [n for n, p in mymodel.named_parameters() if not "layernorm" in n]
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in mymodel.named_parameters() if n in decay_parameters],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [
            p for n, p in mymodel.named_parameters() if n not in decay_parameters
        ],
        "weight_decay": 0.0,
    },
]

optimizer = SGD(
    optimizer_grouped_parameters,
    args.learning_rate,
    momentum=0.9,
    weight_decay=args.weight_decay,
)

n_steps = args.num_train_epochs * len(train_loader)

scheduler = get_cosine_schedule_with_warmup(
    optimizer, int(n_steps * args.warmup_ratio), int(n_steps)
)

trainer = Trainer(
    mymodel,
    args=args,
    train_dataset=train_set,
    eval_dataset=test_set,
    data_collator=mycollate,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
    optimizers=(optimizer, scheduler),
)

if do_train:
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model()
    trainer.save_state()
else:
    # trainer.model.load_state_dict(torch.load(os.path.join(outdir, "pytorch_model.bin")))
    trainer.model = trainer.model.from_pretrained(outdir)

basemodel = deepcopy(trainer.model)

idxs = choose_outlier_for_finetuning(basemodel.to("cpu"), is_vit=True)

basemodel.to("cuda")

settings_accuracies = dict()

param_settings = list(combinations(idxs, 1)) + list(combinations(idxs, 2)) + [[]]
if len(idxs) > 1:
    param_settings += [idxs]
param_settings = [[j for j in i] for i in param_settings]
param_settings += ["random"]

# %%
for param_setting in param_settings:
    if param_setting == "random":
        for i in range(len(idxs)):
            accs = []
            for _ in range(5):
                trainer.model = deepcopy(basemodel)
                zero_vit_param_(
                    trainer.model,
                    idxs=random.sample(set(range(768)).difference(idxs), i + 1),
                )
                outdict = trainer.evaluate()
                accs.append(outdict["eval_accuracy"])
            settings_accuracies[param_setting + f"_len_{i+1}"] = np.mean(accs).item()
        print(
            param_setting + f"_len_{i+1}",
            settings_accuracies[param_setting + f"_len_{i+1}"],
        )
    else:
        trainer.model = deepcopy(basemodel)
        zero_vit_param_(trainer.model, idxs=param_setting)
        outdict = trainer.evaluate()
        settings_accuracies["_".join([str(i) for i in param_setting])] = outdict[
            "eval_accuracy"
        ]
        print(
            "_".join([str(i) for i in param_setting]),
            settings_accuracies["_".join([str(i) for i in param_setting])],
        )

outeval_file_path = os.path.join(outdir, "eval_output.json")
with open(outeval_file_path, "w") as outeval_file:
    json.dump(settings_accuracies, outeval_file)
