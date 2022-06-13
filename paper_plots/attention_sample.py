from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from torch.optim import SGD
import sys

from outlier_analysis.parameter_access import *
from outlier_analysis.parameter_hiding import zero_param_
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--out_path", type=str, default = "output")
parser.add_argument("--model_name_or_path", type=str, default = "bert-base-uncased")
parser.add_argument("--fig_nrows", type=int, default = 1)
args = parser.parse_args()


from copy import deepcopy


model_name_or_path = args.model_name_or_path
output_path = Path(args.out_path)
output_path /= model_name_or_path
output_path /= "attention_sample"
fig_nrows = args.fig_nrows
fig_ncols = 12 // fig_nrows

if model_name_or_path == "bert-base-uncased":
    outliers = [308, 381]
elif model_name_or_path == "roberta-base":
    outliers = [77, 588]

model = AutoModel.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
newmodel = deepcopy(model)
newmodel = zero_param_(newmodel, outliers)


ds = load_dataset("glue", "mnli")

sent_idx = 13
sentence1 = ds["train"]["premise"][sent_idx]
sentence2 = ds["train"]["hypothesis"][sent_idx]

print(sentence1)
print(sentence2)

tokenized = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True,)

out = [
    i.detach().clone()
    for i in model(**tokenized, output_hidden_states=True).hidden_states
]

hidden_out = [
    i.detach().clone()
    for i in newmodel(**tokenized, output_hidden_states=True).hidden_states
]

att_plot_idx = 10

palette = sns.color_palette("flare", as_cmap=True)

att_fig1, att_ax1 = plt.subplots(fig_nrows, fig_ncols, figsize=(20, 2.5))
att_ax1 = att_ax1.reshape(-1)
atts = model(**tokenized, output_attentions=True).attentions[att_plot_idx]
for i in range(12):
    sns.heatmap(atts[0, i, :].detach(), ax=att_ax1[i], vmin=0, vmax=1, cmap=palette, cbar=i==11)
    att_ax1[i].axis('off')
    att_ax1[i].set_title(f"Head n. {i + 1}")
plt.tight_layout()

output_path.mkdir(exist_ok=True, parents=True)

att_fig1.savefig(output_path / f"full_model_Thebes_held_onto_attentions_nrow_{fig_nrows}_ncol_{fig_ncols}.png")


att_fig2, att_ax2 = plt.subplots(fig_nrows, fig_ncols, figsize=(20, 2.5))
att_ax2 = att_ax2.reshape(-1)
atts = newmodel(**tokenized, output_attentions=True).attentions[att_plot_idx]
for i in range(12):
    sns.heatmap(atts[0, i, :].detach(), ax=att_ax2[i], vmin=0, vmax=1, cmap=palette, cbar=i==11)
    att_ax2[i].axis('off')
# att_fig2.suptitle("Attention computed with removed outlier dimensions", fontsize=20)
plt.tight_layout()
att_fig2.savefig(output_path / f"hidden_outlier_Thebes_held_onto_attentions_nrow_{fig_nrows}_ncol_{fig_ncols}.png")
