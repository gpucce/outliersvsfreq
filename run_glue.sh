tasks=(mnli mnli-mm mrpc sst2 qqp qnli stsb cola rte)

for task in ${tasks[@]}
do
    if [[ $task != "mnli-mm" ]]
    then
        echo "train ${task}"
        python -m outliersvsfreq.experiments.baselines --task $task \
            --step "train" \
            --model_name_or_path $1 \
            --output_path $1 \
            --max_length 256 \
            --train_batch_size 32 \
            --random_seed 42 \
            --lr 2.e-5 \
            --output_path $1
    fi

    echo "test ${task}"
    python -m outliersvsfreq.experiments.baselines --task $task \
        --step "test" \
        --model_name_or_path $1 \
        --output_path $1 \
        --max_length 256 \
        --train_batch_size 32 \
        --layer_range_length 12 \
        --random_seed 42 \
        --lr 2.e-5

    if [[ $task == *"mnli"* ]]
    then
        python -m outliersvsfreq.experiments.baselines --task $task \
            --step "test" \
            --model_name_or_path $1 \
            --output_path $1 \
            --max_length 256 \
            --train_batch_size 32 \
            --layer_range_length 1 \
            --lr 2.e-5 \
            --random_seed 42
    fi
done
