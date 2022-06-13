#!/bin/bash

model_dir="./multiberts_ckpts/seed_1"
tasks=(mnli mnli-mm)
models=()
for i in $(ls $model_dir)
do
models+=" $model_dir/$i"
done

for i in ${models[@]}
do
echo $i
done

for task in ${tasks[@]}
do
    for model in ${models[@]}
    do
        if [[ $task != "mnli-mm" ]]
        then
            echo "train ${task}"
            python glue_remake.py --task $task \
            --step "train" \
            --model_checkpoint $model \
            --max_length 256 \
            --train_batch_size 16 \
            --layer_range_length 12 \
            --random_seed 42 \
            --lr 2.e-5 \
            --check_all_idxs false
        fi

        python glue_remake.py --task $task \
        --step "test" \
        --model_checkpoint $model \
        --max_length 256 \
        --train_batch_size 32 \
        --layer_range_length 1 \
        --random_seed 42 \
        --lr 2.e-5 \
        --check_all_idxs false
    done
done