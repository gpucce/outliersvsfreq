from enum import unique
import torch
import numpy as np


__all__ = [
    "get_layers",
    "get_layernorm_layers",
    "get_vit_layernorm_layers",
    "check_out_of_sigma",
    "get_all_outliers",
    "get_outliers",
    "pinpoint_outliers_by_type",
    "pinpoint_outliers",
    "choose_outlier_for_finetuning",
    "get_param_layer",
]


def get_layers(model, present, absent):

    return [
        i[0]
        for i in model.named_parameters()
        if any([there in i[0] for there in present])
        and all([not there in i[0] for there in absent])
    ]


def get_layernorm_layers(model, wob="weight", idxs=range(12)):
    idxs = [str(i) for i in idxs]
    layer_names = get_layers(model, ["LayerNorm." + wob], ["attention", "embeddings"])
    layer_names = [i for i in layer_names if any(j + "." in i for j in idxs)]
    return torch.cat(
        [i[1].unsqueeze(0) for i in model.named_parameters() if i[0] in layer_names],
        axis=0,
    ).detach()


def get_vit_layernorm_layers(model, wob="weight", idxs=range(12)):
    idxs = [str(i) for i in idxs]
    layer_names = get_layers(model, ["layernorm_after." + wob], [])
    layer_names = [i for i in layer_names if any(j + "." in i for j in idxs)]
    return torch.cat(
        [model.get_parameter(i).data.detach().unsqueeze(0) for i in layer_names],
        axis=0,
    ).detach()


def get_wav2vec_layernorm_layers(model, wob="weight", idxs=range(12)):
    idxs = [str(i) for i in idxs]
    layer_names = get_layers(model, ["final_layer_norm." + wob], [])
    layer_names = [i for i in layer_names if any(j + "." in i for j in idxs)]
    return torch.cat(
        [model.get_parameter(i).data.detach().unsqueeze(0) for i in layer_names],
        axis=0,
    ).detach()


def check_out_of_sigma(tens, n_std=3):
    means = tens.mean(1).reshape(-1, 1)
    stds = tens.std(1).reshape(-1, 1)
    return torch.where((tens - means).abs() > n_std * stds)


def get_all_outliers(model, wob, n_std=3):
    return check_out_of_sigma(get_layernorm_layers(model, wob), n_std=n_std)


def get_outliers(model, model_type, wob="weight", n_std=3):
    if model_type == "vit":
        tens = get_vit_layernorm_layers(model, wob)
    elif model_type == "wav2vec":
        tens = get_wav2vec_layernorm_layers(model, wob)
    elif model_type == "LM":
        tens = get_layernorm_layers(model, wob)

    all_outs = check_out_of_sigma(tens, n_std=n_std)
    which, howmany = np.unique(all_outs[1], return_counts=True)
    order = np.argsort(howmany)[::-1]
    return which[order], howmany[order]


def pinpoint_outliers_by_type(
    model, model_type, wob, n_std=3,
):
    which, howmany = get_outliers(model, model_type=model_type, wob=wob, n_std=n_std)
    n2c = {i: j for i, j in zip(which, howmany) if j > 3}
    return sorted([(i, j) for i, j in n2c.items()], key=lambda x: x[1], reverse=True)


def pinpoint_outliers(model, model_type, n_std=3):
    w_n2c = pinpoint_outliers_by_type(
        model, model_type=model_type, wob="weight", n_std=n_std
    )
    b_n2c = pinpoint_outliers_by_type(
        model, model_type=model_type, wob="bias", n_std=n_std,
    )

    min_len = min(len(w_n2c), len(b_n2c))
    count = []
    for i in range(min_len):
        count.append(w_n2c[i])
        count.append(b_n2c[i])
    return count


def choose_outlier_for_finetuning(model, model_type, n_std=3, topk=6):
    out = []
    for i in pinpoint_outliers(model, model_type=model_type, n_std=n_std)[:topk]:
        if not i[0] in out:
            out.append(i[0])
    return out


def get_param_layer(x, param_steps):
    for idx, i in enumerate(param_steps):
        if i[1] > x:
            if idx == 0:
                return i[0], x
            else:
                return i[0], x - param_steps[idx - 1][1]
