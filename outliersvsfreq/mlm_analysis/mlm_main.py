from transformers import Trainer, DataCollatorForLanguageModeling, set_seed
from torch.utils.data import DataLoader
import torch

import pandas as pd

from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm

from outliersvsfreq.parameter_hiding import zero_param_

__all__ = ["MLMAnalysisTrainer"]


class MLMAnalysisTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer)

    def get_train_dataloader(self, masked=True):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.data_collator if masked else None,
            batch_size=self.args.train_batch_size,
        )

    def get_eval_dataloader(self, masked=True):
        return DataLoader(
            self.eval_dataset,
            collate_fn=self.data_collator if masked else None,
            batch_size=self.args.eval_batch_size,
        )

    def correlate_full_to_hidden_pos(self, idxs_groups):

        clean_model = deepcopy(self.model)
        do_full_model = True
        for idxs in idxs_groups:
            model = zero_param_(deepcopy(self.model), idxs=idxs)

            # model.to(self.device)
            # clean_model.to(self.device)

            count = 0
            if do_full_model:
                real = []
                generated_out = []

            outliers_out = []
            set_seed(42)
            masked_loader = self.get_train_dataloader(masked=True)
            unmasked_loader = self.get_train_dataloader(masked=False)
            for masked_batch, batch in tqdm(zip(masked_loader, unmasked_loader)):
                mask = masked_batch["labels"] != -100
                broken_masked_output = model(
                    **{i: j.to(model.device) for i, j in masked_batch.items()}
                ).logits
                broken_masked_generated = (
                    broken_masked_output[mask, :].argmax(-1).detach().cpu()
                )
                if do_full_model:
                    masked_output = clean_model(
                        **{i: j.to(clean_model.device) for i, j in masked_batch.items()}
                    ).logits
                    masked_generated = masked_output[mask, :].argmax(-1).detach().cpu()
                    original = batch["input_ids"][mask]

                    real += [self.tokenizer.decode(tok) for tok in original]
                    generated_out += [
                        self.tokenizer.decode(tok) for tok in masked_generated
                    ]

                outliers_out += [
                    self.tokenizer.decode(tok) for tok in broken_masked_generated
                ]
                count += 1
            outlier_col_name = f"outlier_{idxs}_generated_tokens"
            if do_full_model:
                out = pd.DataFrame.from_dict(
                    {
                        "real_tokens": real,
                        "generated_tokens": generated_out,
                        outlier_col_name: outliers_out,
                    }
                )
                del real
                del generated_out
            else:
                out[outlier_col_name] = outliers_out

            do_full_model = False

        return out

    def get_full_generation_output(self, idxs_groups, max_length):

        masked_loader = self.get_train_dataloader(masked=False)
        unmasked_loader = self.get_train_dataloader(masked=False)
        all_out = dict()
        for idxs in idxs_groups:
            hidden_model = zero_param_(deepcopy(self.model), idxs)
            sentences = []
            out_toks = []
            locout_toks = []
            for masked_batch, batch in zip(tqdm(masked_loader), unmasked_loader):
                out = self.model(
                    **{i: j.to(self.model.device) for i, j in masked_batch.items()}
                ).logits.argmax(-1)
                out_toks.append(out.detach().cpu())
                locout = hidden_model(
                    **{i: j.to(self.model.device) for i, j in masked_batch.items()}
                ).logits.argmax(-1)
                locout_toks.append(locout.detach().cpu())
                sentences.append(batch["input_ids"])

            sentences_names = ["real_tok_" + str(i) for i in range(max_length)]
            normal_names = ["tok_" + str(i) for i in range(max_length)]
            hidden_names = ["hid_tok_" + str(i) for i in range(max_length)]
            names = sentences_names + normal_names + hidden_names

            outdf = pd.DataFrame(
                torch.cat(
                    [
                        torch.cat(sentences, axis=0),
                        torch.cat(out_toks, axis=0),
                        torch.cat(locout_toks, axis=0),
                    ],
                    axis=1,
                ).numpy(),
                columns=names,
            )

            model_name = self.model.config.name_or_path
            idxs_name = "_".join([str(i) for i in idxs])
            outdf.to_csv(
                Path(self.args.output_dir) / f"{model_name}_{idxs_name}_tok_counts.csv",
                index=False,
            )
