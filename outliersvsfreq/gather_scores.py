import pandas as pd
import regex as re
import os
import os.path as osp
import json

__all__ = [
    "gather_scores",
    "free_gather_scores_with_layers",
    "free_gather_scores",
    "gather_scores_in_df",
]


def free_gather_scores(path, exclude=[], include=[], include_or=[], name=""):
    file_paths = sorted([i for i in os.listdir(path)])

    indices = dict()
    scores = dict()
    for file_path in file_paths:
        there_must_not_be = [i in file_path for i in exclude]
        if any(there_must_not_be):
            continue
        there_must_be = [i in file_path for i in include]
        if not all(there_must_be):
            continue
        there_must_be_some = [i in file_path for i in include_or]
        if len(there_must_be_some) > 0 and not any(there_must_be_some):
            continue
        task = file_path.split("_")[0]
        task += f"_{name}" if name else ""
        ids = re.search("[[(](.*)[\)\]]", file_path).group(1)
        if not task in indices:
            indices[task] = []
        if not task in scores:
            scores[task] = []

        if not ids in indices[task]:
            indices[task].append(ids)
        with open(osp.join(path, file_path)) as file:
            out = json.load(file)
        if "eval_accuracy" in out:
            scores[task].append(out["eval_accuracy"])
        elif "eval_matthews_correlation" in out:
            scores[task].append(out["eval_matthews_correlation"])
        elif "eval_spearmanr" in out:
            scores[task].append(out["eval_spearmanr"])

    return scores, indices


def free_gather_scores_with_layers(
    path, exclude=[], include=[], include_or=[], name=""
):
    file_paths = sorted([i for i in os.listdir(path)])

    indices = dict()
    scores = dict()
    losses = dict()
    for file_path in file_paths:
        there_must_not_be = [i in file_path for i in exclude]
        if any(there_must_not_be):
            continue
        there_must_be = [i in file_path for i in include]
        if not all(there_must_be):
            continue
        there_must_be_some = [i in file_path for i in include_or]
        if len(there_must_be_some) > 0 and not any(there_must_be_some):
            continue
        task = file_path.split("_")[0]
        task += f"_{name}" if name else ""
        ids = re.search("[[(](.*)[\)\]]", file_path).group(1)
        if not task in indices:
            indices[task] = dict()
        if not task in scores:
            scores[task] = dict()
        if not task in losses:
            losses[task] = dict()

        layer = re.search("layers_\d+", file_path).group(0)
        if not layer in indices[task]:
            indices[task][layer] = []
        if not layer in scores[task]:
            scores[task][layer] = []
        if not layer in losses[task]:
            losses[task][layer] = []

        if not ids in indices[task][layer]:
            indices[task][layer].append(ids)
        with open(osp.join(path, file_path)) as file:
            out = json.load(file)
        if "eval_accuracy" in out:
            scores[task][layer].append(out["eval_accuracy"])
        elif "eval_matthews_correlation" in out:
            scores[task][layer].append(out["eval_matthews_correlation"])
        elif "eval_spearmanr" in out:
            scores[task][layer].append(out["eval_spearmanr"])
        if "eval_loss" in out:
            losses[task][layer].append(out["eval_loss"])

    return scores, indices, losses


def gather_scores(path, exclude=[], include=[], include_or=[], name=""):

    scores, indices = free_gather_scores(
        path=path, exclude=exclude, include=include, include_or=include_or, name=name
    )

    dfs = dict()

    for task in scores:
        dfs[task] = pd.DataFrame(scores[task], index=indices[task], columns=[task])
    return dfs


def gather_scores_in_df(path, exclude=[], include=[], include_or=[], name=""):
    dfs = gather_scores(
        path=path, exclude=exclude, include=include, include_or=include_or, name=name
    )

    onetask = list(dfs.keys())[0]
    outdf = dfs[onetask]
    for task, df in dfs.items():
        if task == onetask:
            continue
        outdf = outdf.join(df, rsuffix="_" + task)

    return outdf
