

from transformers import AutoTokenizer
import json
import os
import os.path as osp
from tqdm.auto import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--wiki_path", type=str)
parser.add_argument("--model_name", type=str)
args = parser.parse_args()


wikidir = args.wiki_path
model_name = args.model_name

datadirs = [osp.join(wikidir, i) for i in os.listdir(wikidir)]
datafiles = [
    osp.join(directory, file)
    for directory in datadirs
    for file in os.listdir(directory)
]

fastok = AutoTokenizer.from_pretrained(model_name, fast=True)

counts = dict()
for file in tqdm(datafiles):
    with open(file) as datafile:
        lines = datafile.readlines()
    for line in lines:
        if "<doc" in line or "</doc" in line:
            continue
        for item in fastok(line).input_ids:
            token = fastok.decode(item)
            if token in counts:
                counts[token] += 1
            else:
                counts[token] = 1

sorted_counts = sorted(
    [(i, j) for i, j in counts.items()], key=lambda x: x[1], reverse=True
)

with open(f"word_counts/{model_name}_wiki_word_counts.json", "w") as output_file:
    json.dump(sorted_counts, output_file)
