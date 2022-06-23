from copy import deepcopy
from functools import reduce
import datetime


import torch
from torch.nn import Module, Linear, CrossEntropyLoss

from transformers import ViTModel

from .parameter_access import get_layers


device = "cuda" if torch.cuda.is_available() else "cpu"

__all__ = [
    "MASK_STRTG_EXCEPT",
    "MyViT",
    "replace_weight_name",
    "smart_init_",
    "self_generate_sents",
    "adapt2hans",
    "get_all_layers_bounds",
    "get_time_str",
    "get_module_by_name",
]

MASK_STRTG_EXCEPT = Exception('Masking stategy can only be "random" or "deterministic".')


class MyViT(Module):
    def __init__(self, nclasses):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classification_head = Linear(768, nclasses)
        self.loss = CrossEntropyLoss()

    def forward(self, pixel_values, labels):
        y = self.classification_head(self.vit(pixel_values).pooler_output)
        return (self.loss(y, labels), y)


def replace_weight_name(stringa):
    if stringa == "encoder.sentence_encoder.embed_tokens.weight":
        return "embeddings.word_embeddings.weight"
    elif stringa == "encoder.sentence_encoder.embed_positions.weight":
        return "embeddings.position_embeddings.weight"
    elif stringa == "encoder.sentence_encoder.emb_layer_norm.weight":
        return "embeddings.LayerNorm.weight"
    elif stringa == "encoder.sentence_encoder.emb_layer_norm.bias":
        return "embeddings.LayerNorm.bias"

    mapper = [
        ("embed_postions", "position_embeddings"),
        ("sentence_encoder.", ""),
        ("layers", "layer"),
        ("self_attn_layer_norm", "attention.output.LayerNorm"),
        ("self_attn.out_proj", "attention.output.dense"),
        ("final_layer_norm", "output.LayerNorm"),
        ("k_proj", "key"),
        ("q_proj", "query"),
        ("v_proj", "value"),
        ("self_attn", "attention.self"),
        ("fc1", "intermediate.dense"),
        ("fc2", "output.dense"),
    ]
    mapped_stringa = stringa
    for strfrom, strto in mapper:
        mapped_stringa = mapped_stringa.replace(strfrom, strto)
    return mapped_stringa


def smart_init_(model, model_from, idxs):

    layers2smartinit = get_layers(
        model,
        present=["LayerNorm", "position_embeddings"],
        absent=["attention", "classifier"],
    )

    for layer_name in layers2smartinit:
        for idx in idxs:
            layer = model.get_parameter(layer_name).data
            if len(layer.shape) > 1:
                layer[:, idx] = model_from.get_parameter(layer_name).data[:, idx]
            elif len(layer.shape) == 1:
                layer[idx] = model_from.get_parameter(layer_name).data[idx]

    return model


def self_generate_sents(model, loc_tokenizer, sents, start_idx=0, end_idx=None):
    outs = []
    atts = []
    model.to(device)
    for sent in sents:
        linput = {
            i: j[:, start_idx:end_idx].to(device)
            for i, j in loc_tokenizer(sent, return_tensors="pt", padding=True).items()
        }
        out = model(**linput, output_attentions=True)
        out_logit = out.logits[linput["attention_mask"] != 0].argmax(-1).detach().cpu()
        outs.append(loc_tokenizer.decode(out_logit))
        atts.append([i.detach().cpu() for i in out.attentions])
    model.to("cpu")
    return outs, atts


def adapt2hans(x):
    newx = deepcopy(x)
    mask = newx >= 1
    newx[mask] = 1
    return newx


def get_all_layers_bounds(model):
    param_steps = []
    for idx, (i, j) in enumerate(model.named_parameters()):
        if idx == 0:
            param_steps.append((i, j.numel()))
        else:
            param_steps.append((i, j.numel() + param_steps[idx - 1][1]))
    return param_steps


def get_time_str():
    d = datetime.datetime.now()
    s = d.strftime("%y.%m.%d_%H.%M.%S")
    return s


def get_module_by_name(module, access_string):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)
