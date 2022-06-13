import numpy as np
import matplotlib.pyplot as plt

def stack_plot_perf(df1, df2):
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
    ax = [ax1, ax2]
    for idx, df in enumerate([df1, df2]):
        for col_idx, col in enumerate(df.columns):
            ax[idx].plot(df[col].values)
            ax[idx].scatter(range(12), df[col].values)
    return fig1, fig2, ax
