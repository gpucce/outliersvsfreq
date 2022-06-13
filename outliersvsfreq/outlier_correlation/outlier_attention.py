from transformers import Trainer
from tqdm.auto import tqdm
import numpy as np
import torch
import json


__all__ = ["OutlierAnalysisTrainer"]


class OutlierAnalysisTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def correlate_outliers_and_attentions(self, outlier_idxs, avoid_special_toks=False):
        eval_loader = self.get_eval_dataloader()
        hidden_states = {outlier_idx: [] for outlier_idx in outlier_idxs}
        attentions = []
        masks = []
        for idx, batch in enumerate(tqdm(eval_loader)):
            batch_out = self.model(
                **{i: j.to(self.model.device) for i, j in batch.items()},
                output_hidden_states=True,
                output_attentions=True
            )

            for outlier_idx in hidden_states.keys():
                new_hidden_states = [
                    hidden.detach().clone().to("cpu")[:, :, outlier_idx]
                    for hidden in [i for i in batch_out.hidden_states[1:]]
                ]

                hidden_states[outlier_idx] = (
                    [
                        torch.cat([hidden_state, new_hidden_states[hidden_idx]], axis=0)
                        for hidden_idx, hidden_state in enumerate(
                            hidden_states[outlier_idx]
                        )
                    ]
                    if idx > 0
                    else new_hidden_states
                )

            new_attentions = [
                att.detach().clone().to("cpu").mean(-2) for att in batch_out.attentions
            ]

            attentions = (
                [
                    torch.cat([attention, new_attentions[att_idx]], axis=0)
                    for att_idx, attention in enumerate(attentions)
                ]
                if idx > 0
                else new_attentions
            )

            new_masks = batch["attention_mask"].to("cpu")
            if avoid_special_toks:
                new_masks = (
                    new_masks
                    & ~batch["input_ids"]
                    .clone()
                    .cpu()
                    .apply_(lambda x: x in self.tokenizer.all_special_ids)
                    .bool()
                )
            masks = torch.cat([masks, new_masks]) if idx > 0 else new_masks

        return hidden_states, attentions, masks.bool()

    def _init_freq_file(self, freq_files):
        if isinstance(freq_files, str):
            freq_files = [freq_files]
        self.freqs = dict()
        for freq_file in freq_files:
            with open(freq_file, "r") as freq_file_path:
                new_freqs = json.load(freq_file_path)
                for key, val in new_freqs:
                    if key in self.freqs:
                        self.freqs[key] += val
                    else:
                        self.freqs[key] = val
        self.idx_freqs = {
            self.tokenizer(i).input_ids[1]: j for i, j in self.freqs.items()
        }

    def _get_freq(self, x):
        if x in self.idx_freqs:
            return self.idx_freqs[x]
        else:
            return 0

    def get_freq(self, x):
        return np.array([self._get_freq(i) for i in x], dtype=float)

    def get_frequency(self, freq_files, avoid_special_toks=False):
        self._init_freq_file(freq_files)
        eval_loader = self.get_eval_dataloader()
        freqs = []
        masks = []
        for idx, batch in enumerate(tqdm(eval_loader)):
            new_freqs = batch["input_ids"].clone().cpu().apply_(self._get_freq)
            freqs = torch.cat([freqs, new_freqs]) if idx > 0 else new_freqs

            new_masks = batch["attention_mask"].to("cpu")
            if avoid_special_toks:
                new_masks = (
                    new_masks
                    & ~batch["input_ids"]
                    .clone()
                    .cpu()
                    .apply_(lambda x: x in self.tokenizer.all_special_ids)
                    .bool()
                )
            masks = torch.cat([masks, new_masks]) if idx > 0 else new_masks

        return freqs, masks.bool()
