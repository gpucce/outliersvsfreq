# Repository for the paper [Outliers Dimensions that Disrupt Transformers Are Driven by Frequency](https://arxiv.org/abs/2205.11380)


Create a conda environment with `conda env creat -f environment.yaml` this should allow to activate 
```
conda activate outliersvsfreq
```

All the following commands assume you start in the home folder of this project and include `cd` statements to move to the right folder.

## 1) GLUE Baseline

To run the GLUE benchmarks the following should work
```
cd baselines
bash run_glue.sh bert-base-uncased
```
if this is run as is will run several trainings and validations zeroing out outliers one by one in all layers and then one layer at a time (might take some time) otherwise one can comment out some parts to perform less evaluations.

To run the multiberts, download the checkpoints to a folder and then edit the `run_multiberts.sh` inside `baselines/` to add them to the model list replacing `... ADD MODEL HERE ...` after that is done running
```
cd baselines
bash run_multiberts.sh
``` 
should run all the experiments the outputs should be in `baeselines/output`

## 2) MLM dependency on outlier

To run the experiments on MLM the following should work
```
cd mlm_tests
python wikitext_mlm_loss.py --model_name bert-base-uncased
```

## 3) NLI & MLM

After running the two above one should be able to run
```
cd paper_plots
python outlier_scores.py \
    --nli_input_dir ../baselines/output/scores/bert-base-uncased/layers_windows_size_1 \
    --mlm_input_dir ../data_experiments/output/mlm_loss_computation/bert-base-uncased/output \
    --output_file ./output/bert-base-uncased_nli_mlm_figure.png
```

## 4) Word frequencies

To compute the word frequency download the wikipedia corpus in a folder inside the folder `data_experiments` after that run
```
cd data_experiments
python wiki_word_freqs.py --wiki_path path_to_folder \
    --model_name name_or_path_of_model_with_tokenizer
python book_corpus_word_freqs.py --bookcorpus_path path_to_folder \
    --model_name name_or_path_of_model_with_tokenizer
```
and the frequency files should be inside `data_experiments/output/word_counts`. 
For convenience the files used in the paper are there already.

## 5) Correlation to frequency

To compute the correlation to frequency, after the word frequency files are created or using the ones provided the following should do:
```
cd outlier_correlation
python outlier_attention_correlation.py --freq_file_path ../data_experiments/word_counts
```

## 6) Pre-training Bert-medium

To run `bert_medium` pretraining the following would run the _SPLIT_ case
```
cd pre_training_bert_medium
python compact_run_mlm.py --output_dir mlm_run \
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
changing the last two flags runs the different tokenizations.

For _RANDOMIZE_TOKENS_ they should be set to 
```
--randomize_tokens true
--few_special_tokens false
```
and for _ONE_SEP_ to
```
--randomize_tokens false
--few_special_tokens true
```
