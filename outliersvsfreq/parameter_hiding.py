import torch
from .parameter_access import get_layers


__all__ = [
    "zero_param_",
    "zero_last_param_",
    "scale_param_",
    "zero_word_embeddings_",
    "zero_positional_embeddings_",
    "zero_positional_embeddings_complement_",
    "zero_vit_param_",
    "zero_inner_attention_",
]


def zero_param_(model, idxs=[], show_zeroed_layers=False):

    layers2zero = get_layers(
        model, present=["LayerNorm"], absent=["attention", "embeddings"]
    )

    if show_zeroed_layers:
        print(layers2zero)

    for layername in layers2zero:
        for idx in idxs:
            with torch.no_grad():
                model.get_parameter(layername).data[idx] = 0

    return model


def zero_last_param_(model, idxs, fromk, toh=12):

    layers2zero = get_layers(
        model, present=["LayerNorm"], absent=["attention", "embeddings"]
    )

    layers2zero = [
        i
        for i in layers2zero
        if any(["." + str(j) + "." in i for j in range(fromk, toh)])
    ]

    for layername in layers2zero:
        for idx in idxs:
            with torch.no_grad():
                model.get_parameter(layername).data[idx] = 0

    return model


def scale_param_(model, idxs=[], scaling=1 / 2):
    layers2zero = get_layers(
        model, present=["LayerNorm"], absent=["attention", "embeddings"]
    )

    for layername in layers2zero:
        for idx in idxs:
            with torch.no_grad():
                val = model.get_parameter(layername).data[idx]
                model.get_parameter(layername).data[idx] = val * scaling

    return model


def zero_word_embeddings_(model, idxs):
    for idx in idxs:
        with torch.no_grad():
            model.get_parameter("embeddings.word_embeddings.weight").data[:, idx] = 0

    return model


def zero_positional_embeddings_(model, idxs):
    with torch.no_grad():
        for idx in idxs:
            model.get_parameter("embeddings.position_embeddings.weight").data[
                :, idx
            ] = 0

    return model


def zero_positional_embeddings_complement_(model, idxs):
    for idx in set(range(768)).difference(idxs):
        with torch.no_grad():
            model.get_parameter("embeddings.position_embeddings.weight").data[
                :, idx
            ] = 0

    return model


def zero_vit_param_(model, idxs=[]):
    layers2zero = get_layers(model, present=["layernorm"], absent=[])

    for layername in layers2zero:
        data = model.get_parameter(layername).data
        with torch.no_grad():
            data[idxs] = 0

    return model


def zero_wav2_vec_param_(model, idxs):
    layers2zero = get_layers(model, present=["final_layer_norm"], absent=[])

    for layername in layers2zero:
        data = model.get_parameter(layername).data
        with torch.no_grad():
            data[torch.tensor(idxs, dtype=torch.long)] = 0

    return model


def zero_inner_attention_(model, idxs):
    assert all([i in range(12) for i in idxs])
    queries = [
        model.get_parameter(i).data
        for i in get_layers(model, present=["query"], absent=["bias"])
    ]
    keys = [
        model.get_parameter(i).data
        for i in get_layers(model, present=["key"], absent=["bias"])
    ]
    for i in idxs:
        for query in queries:
            query[:, i * 64 : (i + 1) * 64] = 0
        for key in keys:
            key[:, i * 64 : (i + 1) * 64] = 0
    return model
