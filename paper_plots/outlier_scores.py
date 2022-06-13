
import pandas as pd
import regex as re

from outlier_analysis.gather_scores import free_gather_scores_with_layers
from outlier_analysis.plotting import stack_plot_perf, embellish_scores_plot

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--nli_input_dir", type=str)
parser.add_argument("--mlm_input_dir", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

nli_input_dir = args.nli_input_dir
mlm_input_dir = args.mlm_input_dir
output_file = args.output_file

scores_mlm, indices_mlm, losses_mlm = free_gather_scores_with_layers(mlm_input_dir)
scores_nli, indices_nli, losses_nli = free_gather_scores_with_layers(nli_input_dir)

if "bert-base-uncased" in nli_input_dir:
    col_order = ["308", "381", "308, 381", "Full Model"]
elif "roberta-base" in nli_input_dir:
    col_order = ["77", "588", "77, 588", "Full Model"]

df_score = pd.concat({i: pd.DataFrame(j).T for i, j in scores_nli.items()}).sort_index(
    level=1, key=lambda x: [int(re.search("\d+", i).group(0)) for i in x]
)
df_score.columns = indices_nli["mnli"]["layers_0"]
df_score.rename({"": "Full Model"}, axis=1, inplace=True)
df_score = df_score.groupby(level=0).get_group("mnli")

df_loss = pd.concat({i: pd.DataFrame(j).T for i, j in losses_mlm.items()}).sort_index(
    level=1, key=lambda x: [int(re.search("\d+", i).group(0)) for i in x]
)
df_loss.columns = indices_mlm["mlm"]["layers_0"]
df_loss.rename({"": "Full Model"}, axis=1, inplace=True)
df_loss = df_loss.loc[:, col_order].groupby(level=0).get_group("mlm")
df_score = df_score.loc[:, [i for i in df_loss.columns if i in df_score.columns]]

f1, f2, ax = stack_plot_perf(df_score, df_loss)
