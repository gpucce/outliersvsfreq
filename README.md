# Repository for the paper [Outliers Dimensions that Disrupt Transformers Are Driven by Frequency](https://arxiv.org/abs/2205.11380)


Create and activate a conda environment with 
```
conda env creat -f environment.yaml
conda activate outliersvsfreq
```

To replicate the results in the paper first run:

```
bash run_glue.sh bert-base-uncased
```

> **_NOTE:_** This will run all the fine-tuning for bert-base-uncased and all the experiment zeroing outliers out, will take a long time to run.

To then run all experiments in the paper except the bert-medium pretraining use
```bash
python -m outliersvsfreq.experiments.outlier_correlation --model_name_or_path bert-base-uncased
python -m outliersvsfreq.experiments.mlm_loss --model_name_or_path bert-base-uncased
python -m outliersvsfreq.experiments.data_experiments --model_name_or_path bert-base-uncased
```

All the plots in the paper should be replicable using the `paper_plots.ipynb` notebook

Finally to run `bert_medium` pretraining the following should run the _SPLIT_ case
```
cd pre_training_bert_medium
python -m outliersvsfreq.experiments.pretrain_bert_medium \
    --output_dir mlm_run \
    --preprocessing_num_workers 8 \
    --do_train true \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --warmup_step 30000 \
    --weight_decay 0.01 \
    --num_train_epochs 4 \
    --do_eval true \
    --validation_split_percentage 1 \
    --learning_rate 1.e-4 \
    --max_seq_length 256 \
    --do_split_in_sentences true \
    --is_test false \
    --randomize_tokens false \
    --few_special_tokens false \
```
changing the last two flags runs the different data preparation.

For _RANDOMIZE\_TOKENS_ they should be set to 
```
    --randomize_tokens true
    --few_special_tokens false
```
and for _ONE\_SEP_ to
```
    --randomize_tokens false
    --few_special_tokens true
```

>**_NOTE:_** adjust the checkpointing using huggingface settings as this is too variable to be fixed for all machines.