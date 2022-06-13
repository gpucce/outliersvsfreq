
import json
import os
import os.path as osp

from tqdm.auto import tqdm

from blingfire import text_to_sentences
from transformers import AutoTokenizer

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--bookcorpus_path", type=str)
parser.add_argument("--model_name", type=str)
args = parser.parse_args()

wikidir = args.wiki_path
model_name = args.model_name

fastok = AutoTokenizer.from_pretrained("model_name", fast=True)

toronto_path = "toronto_story_books"
files = [
    osp.join(toronto_path, directory, file)
    for directory in os.listdir(toronto_path)
    if not directory.endswith(".txt")
    for file in os.listdir(osp.join(toronto_path, directory))
]

counts = dict()
n_missed = 0
for filepath in tqdm(files):
    with open(filepath, encoding="utf8") as file:
        try:
            filetext = file.read()
        except:
            n_missed += 1
            continue
        sentences = text_to_sentences(filetext).splitlines()
        for sentence in sentences:
            for j in fastok(sentence).input_ids:
                token = fastok.decode(j)
                if token in counts:
                    counts[token] += 1
                else:
                    counts[token] = 1

sorted_counts = sorted(
    [(i[0], i[1]) for i in counts.items()], key=lambda x: x[1], reverse=True
)
with open(f"output/data_experiments/word_counts/{model_name}_book_corpus_wordcount.json", "w") as book_corpus_count:
    json.dump(sorted_counts, book_corpus_count)
