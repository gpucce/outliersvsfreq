from scipy.stats import pearsonr, spearmanr

import torch
import numpy as np

import matplotlib.pyplot as plt

__all__ = [
    "stack_plot_corr",
    "stack_plot_freq",
]


def _corr(hidden_states, attentions, masks):
    atts = []
    for att_idx in range(12):
        to_plot = []
        for idx in range(12):
            corr = torch.corrcoef(
                torch.cat(
                    [
                        hidden_states[idx][masks.bool()].unsqueeze(0),
                        attentions[idx][:, att_idx, :][masks.bool()].unsqueeze(0),
                    ]
                )
            )[0, 1].item()
            to_plot.append(corr)
        atts.append(torch.tensor(to_plot).unsqueeze(0))
    return torch.cat(atts).T


def stack_plot_corr(
    hidden_states,
    attentions,
    masks,
    corr_func=_corr,
    outliers_idxs=[],
    random_dims=None,
):
    fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    
    idxs = outliers_idxs + ["random"]
    for idx in range(3):
        if idx <= 1:
            to_plot = corr_func(
                hidden_states[idxs[idx]],
                attentions,
                masks,
            )
        else:
            loc_hidden_states = []
            for random_idx in random_dims:
                newcorr = corr_func(
                    hidden_states[random_idx],
                    attentions,
                    masks,
                )
                loc_hidden_states.append(newcorr)
            to_plot = torch.cat([i.unsqueeze(0) for i in loc_hidden_states]).mean(0)
        for attention_head_idx in range(12):
            ax[idx].plot(to_plot.numpy()[:, attention_head_idx],)
            ax[idx].scatter(range(12), to_plot.numpy()[:, attention_head_idx],)

    return fig, ax

def corr_freq(freqs, hidden_states, masks):
    to_plot = []
    for idx in range(12):
        corr = pearsonr(freqs[masks.bool()], hidden_states[idx][masks.bool()])[0]
        to_plot.append(corr)
    return torch.tensor(to_plot)

def stack_plot_freq(
    freqs, hidden_states, masks, corr_func=pearsonr, outliers_idxs=[], random_dims=None
):
    labels = [
        f"Outlier {outliers_idxs[0]}",
        f"Outlier {outliers_idxs[1]}",
        "Random Dimension",
    ]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharey=True)

    idxs = outliers_idxs + ["random"]
    for idx in range(3):
        if idx <= 1:
            to_plot = corr_freq(
                freqs, hidden_states[idxs[idx]], masks
            ).numpy()
        else:
            loc_hidden_states = []
            for random_idx in random_dims:
                newcorr = corr_freq(
                    freqs, hidden_states[random_idx], masks
                )
                loc_hidden_states.append(newcorr)
            to_plot = (
                torch.cat([i.unsqueeze(0) for i in loc_hidden_states])
                .mean(0)
                .numpy()
            )
        ax.plot(to_plot, markersize=10)
        ax.scatter(range(12),to_plot,)
    return fig, ax
