tasks=(mrpc sst2 qqp qnli mnli mnli-mm stsb cola rte)

for task in ${tasks[@]}
do
    if [[ $task != "mnli-mm" ]]
    then
        echo "train ${task}"
        python glue_remake.py --task $task \
        --step "train" \
        --model_checkpoint $1 \
        --max_length 256 \
        --train_batch_size 32 \
        --layer_range_length 12 \
        --random_seed 42 \
        --lr 2.e-5 \
        --check_all_idxs true
    fi

    echo "test ${task}"
    python glue_remake.py --task $task \
    --step "test" \
    --model_checkpoint $1 \
    --max_length 256 \
    --train_batch_size 32 \
    --layer_range_length 12 \
    --random_seed 42 \
    --lr 2.e-5 \
    --check_all_idxs true

    python glue_remake.py --task $task \
    --step "test" \
    --model_checkpoint $1 \
    --max_length 256 \
    --train_batch_size 32 \
    --layer_range_length 1 \
    --random_seed 42 \
    --lr 2.e-5 \
    --check_all_idxs false
done