#!/bin/bash

model_dir="/data/evaled/mberts_1/checkpoints/"
tasks=(mnli-mm)

models=()
for i in $(ls $model_dir)
do
models+=" $model_dir/$i"
done

for task in ${tasks[@]}
do
    for model in ${models[@]}
    do
        if [[ $task != "mnli-mm" ]]
        then
            echo "train ${task}"
            python -m outliersvsfreq.experiments.baselines --task $task \
                --step "train" \
                --model_name_or_path $model \
                --max_length 256 \
                --train_batch_size 16 \
                --layer_range_length 12 \
                --random_seed 42 \
                --lr 2.e-5 \
                --check_all_idxs false
        fi

        python -m outliersvsfreq.experiments.baselines --task $task \
            --step "test" \
            --model_name_or_path $model/NLI/ep_003_smpl_1M/hf \
            --max_length 256 \
            --train_batch_size 32 \
            --layer_range_length 12 \
            --random_seed 42 \
            --lr 2.e-5 \
            --check_all_idxs false,
    done
done